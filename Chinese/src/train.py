# ==============================================================================
# 基础依赖库：模型训练的指挥部
# ==============================================================================
import torch                      # 核心深度学习框架 
import numpy as np                # 数值计算库 
from torch.utils.data import Dataset, DataLoader # 数据流水线组件 
from torch.utils.data.distributed import DistributedSampler # DDP: 数据集切成 N 份，每卡只看自己那份
from torch.nn.parallel import DistributedDataParallel       # DDP: 模型包装器，自动同步多卡梯度
import torch.distributed as dist                            # DDP: 进程组通信，AllReduce 梯度同步的底层接口
from model import Transformer, ModelArgs # 导入我们手搓的模型结构 (Ep1-Ep18)
import os # 用于增删改查文件
import math # 用于手写 warmup + cosine decay 的数学计算
import yaml # 引入 yaml 解析库

# ==============================================================================
# 训练配置：定义训练的"超参数"
# ==============================================================================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_BASE_DIR, "../config_zh.yaml"), "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)
    _train_cfg = _cfg["training"]
    _model_cfg = _cfg["model"]

DATA_PATH     = _train_cfg["data_path"] # 语料
BATCH_SIZE    = _train_cfg["batch_size"] # 批次大小
SEQ_LEN       = _train_cfg["seq_len"] # 序列长度
LR_MAX        = _train_cfg["learning_rate"] # 学习率最大值
EPOCHS        = _train_cfg["epochs"] # 轮次
VOCAB_SIZE    = _model_cfg["vocab_size"] # 词表大小
WARMUP_STEPS  = _train_cfg["warmup_steps"] # 热身步数
LR_MIN        = _train_cfg["lr_min"] # 学习率最小值

# ==============================================================================
# 学习率调度：手写 warmup + cosine decay，替代 CosineAnnealingLR
# ==============================================================================
# ---- 核心思想 ----
# 学习率 (LR) 就像你骑自行车的速度：
#   1. Warmup 阶段：刚起步时慢慢加速，避免猛踩踏板把链条蹬飞（梯度爆炸）
#   2. Cosine Decay 阶段：到达巡航速度后，用余弦曲线平滑减速，让模型在 loss 低谷处"精细调整"而不是猛冲过头
# 为什么不用 PyTorch 自带的 CosineAnnealingLR？
#   因为它要求你提前填死 T_max（总步数），但 DDP 多卡训练时每张卡分到的 batch 数会变，
#   T_max 一旦填错，余弦曲线就对不齐实际训练进度，LR 要么提前触底要么还没降完训练就结束了。
#   手写就一行 math.cos，反而更灵活、更不容易出 bug。

def get_lr(step: int) -> float:
    """
    手写学习率调度器：给定当前 step，返回此刻应该使用的学习率。
    整条曲线长这样：
      LR
      ▲
      │      ╭──╮                ← 峰值 LR_MAX (3e-4)
      │     ╱    ╲
      │    ╱      ╲              ← 余弦平滑下降
      │   ╱        ╲
      │  ╱          ╲────────    ← 谷底 LR_MIN (3e-5)
      │ ╱  warmup
      └──────────────────────→ step
        0   400            T_MAX
    """
    # Warmup 阶段：从 0 线性增长到 LR_MAX
    if step < WARMUP_STEPS:
        return LR_MAX * step / WARMUP_STEPS
    # Cosine decay 阶段：从 LR_MAX 平滑衰减到 LR_MIN
    # progress ∈ [0, 1]，代表"warmup 结束后你走了全程的百分之几"
    # cos(0) = 1 → 刚开始衰减时 LR 最高
    # cos(π) = -1 → 衰减结束时 LR 最低
    # 套一个 0.5*(1+cos) 把范围压到 [0, 1]，再乘上 (峰值-谷底) + 谷底，完美映射
    progress = (step - WARMUP_STEPS) / (T_MAX_STEPS - WARMUP_STEPS)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))

# ==============================================================================
# 数据集类：定义如何读取和切分数据
# ==============================================================================
class PretokenizedDataset(Dataset):
    # PretokenizedDataset这个类继承 Dataset，它要求你必须写__len__和__getitem__方法
    def __init__(self, data_path, seq_len):
        super().__init__()
        # 使用 np.fromfile 瞬间加载整个二进制语料库到内存，直接将磁盘上的二进制数据映射到内存中
        self.data = np.fromfile(data_path, dtype=np.uint32)
        # uint32表示告诉 NumPy 如何"解释"这串二进制 0/1 序列，32 表示每个数字占 32 个 bit，也就是 4Byte 字节
        # 这一步直接把data_path里的所有二进制码全部装到 self.data，每 8 个字节切一刀变成一个数组（连续内存）
        self.seq_len = seq_len
        # 存储序列长度

    def __getitem__(self, idx):
        # 注意，凡是__xxx__的方法，称为魔术方法，会在触发条件时自动被调用。
        # 这里__getitem__的触发条件就是 obj[i]（如果实例起名叫 obj）
        # 这个[i]中的 i 就是第几块的意思

        # 传入的参数 idx 表示第几块 seq，idx=0 表示第 1 块 1024 个 token，idx=1 就是第 2 块 1024  个 token

        start_idx = idx * self.seq_len
        # 计算起始点索引
        end_idx = start_idx + self.seq_len
        # 计算结束点索引
        
        chunk = np.array(self.data[start_idx : end_idx + 1])
        # 创建块，存储取出的 1025 个 token（多取 1 个是因为 y 要向右平移一格）
        
        # 前 1024 个作为输入，后 1024 个作为标签
        # .astype(np.int64)：PyTorch 的 CrossEntropyLoss 要求标签是 int64 (Long)
        # 输入也转 int64 是为了与 nn.Embedding 层的索引类型对齐
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        # 举例：seq_len=4 时，chunk = [床, 前, 明, 月, 光]
        #   x = [床, 前, 明, 月]   ← chunk[:-1]，模型看到的输入
        #   y = [前, 明, 月, 光]   ← chunk[1:]，模型被要求预测的答案

        return (x, y)
        # 返回一个元组，包含两个列表，输入列表 x 和标签列表 y，
        # 现在里面的 token 都是词表数字，还没有变成向量
    
    def __len__(self):
        return (len(self.data) - 1) // self.seq_len
        # 统计总共有多少块
        # 减 1 是因为每个 chunk 需要多取 1 个 token 给 y，所以 data 总 token 数要减 1

# ==============================================================================
# 训练主循环：大模型的"进化"过程
# ==============================================================================
def train():
    # ==============================================================================
    # DDP 初始化：建立多卡进程组，每张卡跑一个独立进程
    # ==============================================================================
    # ---- DDP 是什么？ ----
    # DistributedDataParallel：PyTorch 的多卡并行训练方案
    # 核心思路：每张 GPU 上跑一个完全独立的 Python 进程，每个进程持有一份完整的模型副本
    # 训练时：
    #   1. 数据被 DistributedSampler 切成 N 份，每张卡只看自己那份（数据并行）
    #   2. 各卡独立做 forward + backward，算出各自的梯度
    #   3. 通过 NCCL（NVIDIA 的高速卡间通信库）做 AllReduce，也就是把所有卡的梯度加权平均
    #   4. 每张卡用相同的平均梯度更新参数 → 保证所有卡的模型权重始终一致
    # 好处：相比 DataParallel，没有"主卡瓶颈"，通信和计算可以重叠，效率高得多

    use_ddp = "RANK" in os.environ
    if use_ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        # backend='nccl'：指定用 NCCL 协议通信，这是 NVIDIA GPU 多卡训练的事实标准
        # 这一句会阻塞等待，直到所有进程（所有卡）都调用了它，才会一起放行
        local_rank = int(os.environ["LOCAL_RANK"])
        # 当前进程对应哪张卡（比如 0/1/2/3）
        # LOCAL_RANK 由 torchrun 自动注入环境变量，不需要手动设置
        world_size = dist.get_world_size()
        # 总共几张卡参与训练
        # 每张卡绑定自己的 GPU，避免所有进程都往 GPU 0 上堆
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        local_rank = 0
        world_size = 1
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps" # 苹果M芯片走这里！
        else:
            device = "cpu" # 要是用上这步了，兄弟你蛮牛逼的
            
    is_master = (local_rank == 0)
    # 只让 rank 0（主进程）打印和保存，避免 3 个进程同时输出造成日志混乱


    # ==============================================================================
    # 1. 准备数据流：从磁盘到 GPU 的完整管道
    # ==============================================================================
    # 数据流向：磁盘二进制文件 → Dataset (切块) → Sampler (分卡) → DataLoader (批处理) → GPU
    dataset = PretokenizedDataset(DATA_PATH, SEQ_LEN)
    # dataset是PretokenizedDataset这个类的实例
    # 你需要用第i块就用 dataset[i] 就行，返回一个元组(x, y)
    # x 是输入，y 是标签（也就是理想输出）

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True) if use_ddp else None
    # DistributedSampler：DDP 的核心组件之一
    # 它负责把 dataset 的索引 [0, 1, 2, ..., N-1] 均匀分给 world_size 张卡
    # 比如 3 张卡、9 个样本：卡0拿[0,3,6]，卡1拿[1,4,7]，卡2拿[2,5,8]
    # shuffle=True 保证每个 epoch 的分配顺序不同，避免模型"记住"数据顺序
    # sampler最终就是一个整数列表（一堆 idx），告诉每张卡的 loader 用取 dataset 里面的第几块 idx
    # 搬运指定数据给当前这张卡去进行计算

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                        shuffle=(not use_ddp),
                        num_workers=_train_cfg["num_workers"],
                        pin_memory=(device != "mps"),
                        persistent_workers=(_train_cfg["num_workers"] > 0))
    # DataLoader：数据的"传送带"
    # batch_size=14：每次从 sampler 取 14 个样本打包成一个 batch
    # dataset[0] → x shape=(1024,)
    # dataset[3] → x shape=(1024,)
    # dataset[6] → x shape=(1024,)
    # ...取够14个...
    # → 拼成一个大X batch，shape=(14, 1024)，同样大Y也是这个 size

    # num_workers=8：开 8 个子进程并行从磁盘预加载下一批数据，避免 GPU 等 CPU
    # pin_memory=True：把数据预先钉在 CPU 的"锁页内存"中，加速 CPU→GPU 的传输（DMA 直通）
    # 所谓锁页内存就是说这些内存不许放回到磁盘里面去，给我锁在内存 RAM 里面！

    # persistent_workers=True：worker 进程在 epoch 之间不销毁重建，省去反复 fork 的开销
    # fork就是创建新进程的方式，要是一轮 epoch 结束杀掉老 worker 再去创建会很浪费时间

    # 注意：这里没有传 shuffle=True，因为 DistributedSampler 已经接管了打乱逻辑
    
    # 填坑：这里要动态计算真实的最大 Step 数量，让 Cosine 衰减曲线完美贴合实际训练进度
    # 为什么要动态算？因为 len(loader) 取决于 数据总量 / (batch_size × world_size)
    # 你换了数据集或者加减 GPU，步数都会变，硬编码 T_MAX_STEPS 迟早翻车
    global T_MAX_STEPS
    # global 声明 T_MAX_STEP不仅仅输入 train 这个函数，全局变量

    steps_per_epoch = len(loader)
    # len(loader)是 pytorch 提供的方法，len(loader)=⌊每张卡的样本数/batch_size⌋
    # 这样直接计算出了每轮训练多少步，每一步模型其实对 batch_size * seqlen 这么多 token 进行了
    # 一次：{前向传播，计算损失函数，反向传播，维护优化器，更新模型参数}

    T_MAX_STEPS = steps_per_epoch * EPOCHS
    # 最大步数等于每轮步数乘上轮数

    if is_master:
        # 第一张卡打印一次步数
        print(f"动态计算完毕：每 Epoch 步数 {steps_per_epoch}，总步数 T_MAX_STEPS = {T_MAX_STEPS}")

    # ==============================================================================
    # 2. 初始化模型：按照"tiny"预设构建模型
    # ==============================================================================
    # ModelArgs.get_args("tiny", ...) 是我们在 Ep1-Ep18 手搓模型时定义的配置工厂
    # "tiny" 预设对应一组较小的超参（层数、头数、hidden_dim），适合在有限显存下实验
    args = ModelArgs.from_yaml(os.path.join(_BASE_DIR, "../config_zh.yaml"))

    # .to(device)：把模型的所有参数搬到当前进程绑定的那张 GPU 上
    model = Transformer(args).to(device)

    # ---- 断点续训（Resume）机制 ----
    # 训练大模型动辄几天几周，中途断电、OOM、手抖 Ctrl+C 都是家常便饭
    # 所以我们需要一个"存档/读档"系统：每隔一段时间把模型权重 + 优化器状态 + 训练进度存下来
    # 下次启动时检测到存档文件就自动从断点继续，而不是从头再来
    start_epoch = 0      # 默认从第 0 个 epoch 开始
    global_step = 0      # 全局步数计数器，跨 epoch 累加，用于 LR 调度和日志
    RESUME_FILE = "model_latest.pth"  # 存档文件名，每次保存都会覆盖这个文件

    # 准备一个空篮子，暂时存放优化器的状态
    temp_opt_state = None 

    if os.path.exists(RESUME_FILE):
        if is_master:
            print(f"正在尝试从 {RESUME_FILE} 恢复训练...")
        # map_location=device：加载时直接映射到当前卡，避免所有进程都往 GPU 0 上加载再挪过去
        checkpoint = torch.load(RESUME_FILE, map_location=device)
        
        # 兼容两种存档格式：
        #   新格式（dict）：包含 model_state_dict、optimizer_state_dict、epoch、global_step
        #   旧格式（纯权重）：只有 model.state_dict()，没有优化器和进度信息
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # isinstance判断checkpoint是不是dict
            model.load_state_dict(checkpoint['model_state_dict'])
            # 从 checkpoint 字典读取模型状态字典
            temp_opt_state = checkpoint['optimizer_state_dict']
            # 从 checkpoint 字典读取优化器状态字典
            start_epoch = checkpoint['epoch']
            # 从 checkpoint 字典读取轮数
            global_step = checkpoint['global_step']
            # 从 checkpoint 字典读取全局步数（也就是与轮数无关的全局总步数）
        else:
            model.load_state_dict(checkpoint)
            # 不是字典就直接读权重本身，这个是为了兼容我的老版本 checkpoint，其实可以删掉了
        
        if is_master:
            print(f"✅ 恢复成功！从 Epoch {start_epoch+1}, Step {global_step} 继续训练")

    # ---- torch.compile：PyTorch 2.0 的编译加速 ----
    # 原理：把 Python 写的模型"编译"成优化过的 CUDA 内核，减少 Python 解释器的开销
    # 类比：你写了一篇中文文章，compile 相当于把它翻译成机器码直接执行，跳过逐字翻译的过程
    model = torch.compile(model, fullgraph=False)

    # ---- DDP(DistributedDataParallel分布式数据并行) 包装：让模型获得多卡同步能力 ----
    # 这一步做了三件事：
    #   1. 给模型的每个参数注册 "梯度钩子"（gradient hook）
    #   2. 每当某个参数的梯度算完，钩子自动触发 AllReduce，跟其他卡交换梯度
    #      AllReduce说白了就是把梯度每个数字除以当前显卡总数
    #   3. 通信和计算重叠进行（overlap），所以你几乎感觉不到通信开销
    # device_ids=[local_rank]：告诉 DDP 这个进程的模型在哪张卡上
    model = DistributedDataParallel(model, device_ids=[local_rank]) if use_ddp else model
    
    # ==============================================================================
    # 3. 定义优化器与损失函数
    # ==============================================================================
    # AdamW：目前大模型训练的标准优化器 (Ep13)
    # 相比 SGD：自带动量（momentum）和自适应学习率，收敛更快更稳
    # 相比 Adam：W = Weight Decay，把 L2 正则化从梯度里解耦出来，避免自适应学习率干扰正则效果
    # lr=LR_MAX：初始学习率，后面会被 get_lr() 动态覆盖
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX)

    # 把之前暂存的优化器状态灌回去
    # 为什么优化器状态很重要？因为 Adam 维护了每个参数的一阶动量（均值）和二阶动量（方差）
    # 如果不恢复，等于告诉优化器"我对这些参数一无所知"，它会像刚开始训练一样乱走几百步才能稳定
    if temp_opt_state is not None:
        optimizer.load_state_dict(temp_opt_state)
        if is_master:
            print("✅ 优化器动量状态也已完美恢复！")

    # 统计可训练参数总量，方便确认模型规模是否符合预期
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # .numel()返回 tensor 中的元素个数，比如 p.shape = (1024, 512) 的 p.numel() = 1024 * 512 = 524288
    # 统计总参数量，注意只有需要梯度的才是参数，会被更新

    if is_master:
        print(f"我的模型现在有 {total_params} 个参数在等待训练！")

    # CrossEntropyLoss：分类任务的标准损失函数 (Ep13)
    # 本质：衡量模型预测的概率分布和真实答案之间的"距离"
    # 对于语言模型：每个位置都是一个 vocab_size 类的分类问题（"下一个 token 是词表中的哪个？"）
    criterion = torch.nn.CrossEntropyLoss()

    # ==============================================================================
    # 4. 开始训练循环
    # ==============================================================================
    if is_master:
        print(f"开始训练，设备: {device}，总卡数: {world_size}...")
        print(f"注意：torch.compile 第一个 step 需要 1~2 分钟编译，请耐心等待...")
    model.train()

    SAVE_INTERVAL = 1000  # 每 1000 步保存一次权重
    LOG_INTERVAL = 100    # 每 100 步打印一次 Loss
    saved_checkpoints = []  # 记录已保存的 checkpoint 文件名，用于滚动清理（只保留最近 3 个）

    # ---- 混合精度训练 (Mixed Precision) ----
    # 核心思想：前向传播和部分反向传播用半精度（bf16/fp16），省显存、提速度
    # 但梯度更新和 loss 计算保持 float32，保证数值稳定性
    # bf16 vs fp16：（现在几乎全面采用 bf16，fp16 已经是历史遗留问题了）
    #   bf16（bfloat16）：指数位和 float32 一样宽（8 bit），不容易溢出，Google TPU / NVIDIA A100+ 原生支持
    #   fp16（float16）：精度更高但指数位窄（5 bit），大数容易变成 inf，需要 GradScaler 动态缩放
    
    # 使用 try-except-finally 结构，确保哪怕训练崩溃或被 Ctrl+C，NCCL 进程组也能被正确释放，避免显存卡死成为"僵尸进程"
    try:
        for epoch in range(start_epoch, EPOCHS):
            # ---- 重要：每个 epoch 必须调用 sampler.set_epoch(epoch) ----
            # 如果不调，DistributedSampler 每个 epoch 用相同的随机种子打乱
            # 结果：每张卡每轮看到的数据顺序完全一样 → 模型学到的是"顺序"而不是"内容"
            sampler.set_epoch(epoch)
            # total_loss 用于累加整个 epoch 的 loss，最后算平均
            # 注意：这里累加的是 GPU 上的 tensor，不会每步都做 GPU→CPU 同步
            total_loss = 0
            for X, Y in loader:
            # 这里触发了 loader 这个实例的__iter__魔术方法
            # DataLoader 的 __iter__ 内部做的事情是：每次迭代从 sampler 拿一批 idx，调用 dataset[idx] 取数据，
            # 拼成 (batch_size, seq_len) 的 batch，然后把 (X, Y) 这个元组交给你。
            
                # ---- 一个 training step 的完整流程 ----
                # 1. 更新学习率 → 2. 数据搬到 GPU → 3. 清零梯度 → 4. 前向传播
                # → 5. 算 loss → 6. 反向传播 → 7. 梯度裁剪 → 8. 更新参数
                global_step += 1

                # Step 1：根据当前 step 从我们手写的调度器拿到学习率，注入 optimizer
                # 为什么不在 optimizer 里直接改？因为 AdamW 初始化时只接受一个固定 lr
                # 我们需要每一步都手动覆盖 param_group 里的 lr 值
                current_lr = get_lr(global_step)
                for param_group in optimizer.param_groups:
                    # 优化器可定制学习率，optimizer.param_groups有embedding，Transformer，head等
                    param_group['lr'] = current_lr

                # Step 2：把数据从 CPU（pin_memory）搬到 GPU
                # non_blocking=True：异步传输，不等传完就继续执行下面的代码
                # 配合 pin_memory 使用效果最佳：CPU 端锁页内存 → GPU 显存，走 DMA 直通，不占 CPU
                X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)

                # Step 3：清零上一步残留的梯度，因为 pytorch 默认梯度是累加的
                # set_to_none=True：不是把梯度填 0，而是直接把 .grad 设为 None
                # 比 .zero_grad() 快，因为省了一次"写 0"的显存操作
                # 代价是如果你后面代码里有 if param.grad is not None 的判断，需要注意
                optimizer.zero_grad(set_to_none=True)
                
                # Step 4：前向传播（在混合精度上下文中）
                # torch.amp.autocast：自动把算子（矩阵乘法、线性层等）降到半精度执行
                # 但注意：不是所有算子都会降精度，PyTorch 内部维护了一张"白名单/黑名单"
                # 比如矩阵乘法会降到 bf16，但 softmax、layer_norm 会保持 fp32
                autocast_device = "cuda" if "cuda" in device else "cpu"
                use_bf16 = _train_cfg["use_bf16"]
                with torch.amp.autocast(autocast_device, enabled=("cuda" in device), dtype=torch.bfloat16 if use_bf16 else torch.float16):
                    logits, aux_loss = model(X)
                    # logits: [batch_size, seq_len, vocab_size]，模型对每个位置预测的词表概率分布
                    # aux_loss: MoE 的辅助损失（负载均衡 loss），防止所有 token 都涌向同一个专家
                    # DDP: model.module 才是原始 Transformer，DDP 包装层没有 get_aux_loss 方法
                    # aux_loss = model.module.get_aux_loss()
                    
                    ## main_loss = criterion(logits.view(-1, args.vocab_size), Y.view(-1))
                    ## loss = main_loss + 0.01 * aux_loss

                # .float() 就是 .to(torch.float32) 的简写
                logits = logits.float()

                main_loss = criterion(logits.view(-1, args.vocab_size), Y.view(-1))
                # .view(-1, args.vocab_size)：把 [batch, seq_len, vocab] 展平成 [batch*seq_len, vocab]
                # .view(-1)：把 [batch, seq_len] 展平成 [batch*seq_len]
                # 这样 CrossEntropyLoss 就能逐 token 计算 loss 了
                # 总 loss = 主 loss + 0.01 × MoE 辅助 loss

                loss = main_loss + _train_cfg["aux_loss_weight"] * aux_loss.float()
                # 权重从 YAML 中读取：太大会让模型把注意力放在"平衡专家负载"上而忽略语言建模
                # 太小则 MoE 的负载均衡形同虚设，某些专家被饿死
                # float(aux_loss)：确保 aux_loss 也被转成 fp32 参与计算

                # --- Step 5-7: 反向传播 + 梯度裁剪 + 参数更新 ---
                loss.backward()
                # .backward()靠链式法则反向传播，计算出损失函数对于每个参数的偏导数∂Loss/∂W

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # 梯度裁剪 (Gradient Clipping)：把所有参数的梯度的全局 L2 范数限制在 1.0 以内
                # L2范数说白了就是梯度每个元素的平方和，广义距离。
                # 如果梯度范数超过 1.0，就按比例缩小所有梯度，防止某一步"迈太大步"导致训练崩溃
                # g_new = g_old / (L2)
                # 这在 Transformer 训练早期尤其重要，attention 的梯度经常突然飙大

                optimizer.step()
                # 优化器对参数进行更新，对于每个参数 Wt 有
                # Wt+1 = Wt - η(mt/(√vt + ε)-λWt)

                total_loss += loss.detach()
                # detach()：切断 loss 和计算图的联系，防止累加的 total_loss 也被纳入反向传播
                
                # ---- 日志打印：每 LOG_INTERVAL 步输出一次 ----
                if global_step % LOG_INTERVAL == 0 and is_master:
                    current_lr = optimizer.param_groups[0]['lr'] 
                    # ⚠️ 只有在真要打印日志的这一刻（比如每 100 步），才使用 .item() 让 GPU 传数据给 CPU
                    # 每 100 步同步一次的代价可以忽略不计，但如果每步都 .item() 就会显著拖慢训练
                    current_loss_val = loss.item()
                    print(f"Step {global_step} | Epoch {epoch+1} | Loss: {current_loss_val:.4f} | LR: {current_lr:.2e}")

                # ---- Checkpoint 保存：每 SAVE_INTERVAL 步存一次档 ----
                if global_step % SAVE_INTERVAL == 0:
                    
                    # 只有 rank 0 负责写文件，避免多个进程同时写同一个文件导致损坏
                    if is_master:
                        # ---- 从 DDP + compile 的"俄罗斯套娃"中取出原始模型权重 ----
                        # model                          → DistributedDataParallel 包装层
                        # model.module                   → torch.compile 生成的 OptimizedModule
                        # model.module._orig_mod         → 我们手搓的原始 Transformer
                        # 如果没有 compile，model.module 直接就是 Transformer
                        # 为什么要剥到最里面？因为保存/加载时我们只想要干净的权重键名
                        # 带 DDP 前缀（module.xxx）或 compile 前缀的键名，恢复时容易对不上
                        raw_model = model.module._orig_mod if hasattr(model.module, "_orig_mod") else model.module
                        checkpoint_data = {
                            'epoch': epoch,               # 当前 epoch 编号，恢复时从下一个 epoch 继续
                            'global_step': global_step,   # 全局步数，恢复时 LR 调度器能接上
                            'model_state_dict': raw_model.state_dict(),     # 模型权重
                            'optimizer_state_dict': optimizer.state_dict(), # 优化器状态（Adam 动量）
                        }
                        ckpt_name = f"model_step_{global_step}.pth"
                        # checkpoint名字 model_step_多少步.pth
                        torch.save(checkpoint_data, ckpt_name)
                        # 保存成一个字典
                        saved_checkpoints.append(ckpt_name)

                        # 同时覆盖 model_latest.pth，这样下次启动时无脑加载最新的即可
                        torch.save(checkpoint_data, "model_latest.pth")
                        print(f"💾 Step {global_step} 进度(含优化器状态)已保存: {ckpt_name}")

                        # 滚动清理：只保留最近 3 个 checkpoint，避免磁盘被撑爆
                        # 大模型一个 checkpoint 动辄几个 GB，15 个 epoch 下来能存几十个
                        if len(saved_checkpoints) > 3:
                            oldest_ckpt = saved_checkpoints.pop(0) 
                            # saved_checkpoints 是一个队列，超过 3 后每次删除最左侧的元素
                            if os.path.exists(oldest_ckpt):
                                os.remove(oldest_ckpt) 
                                print(f"清理旧权重: {oldest_ckpt}")
                    if use_ddp:
                        dist.barrier()
                        # 这是同步屏障，所有 GPU 跑到这里都必须停下，等 Rank 0 的 Master 把文件彻底写入硬盘。
                        # 否则其他显卡已经冲去算下一个 Batch 了，会导致 DataLoader 脱节甚至显存 OOM。

            # ---- Epoch 结束：打印整个 epoch 的平均 loss ----
            if is_master:
                # 这里才调用 .item()，一个 epoch 只同步一次，开销可以忽略
                avg_loss = (total_loss / len(loader)).item()
                # .item()是把张量转为普通的浮点数
                print(f"--- Epoch {epoch+1} 完成，平均 Loss: {avg_loss:.4f} ---")

    except KeyboardInterrupt:
        # ---- 紧急存档：Ctrl+C 中断时的安全网 ----
        # 训练到一半发现 loss 不对想停下来调参？没问题，按 Ctrl+C 不会丢失进度
        if is_master:
            print("\n检测到训练中断，正在紧急保存当前权重...")
            raw_model = model.module._orig_mod if hasattr(model.module, "_orig_mod") else model.module
            checkpoint_data = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': raw_model.state_dict(), # 存干净的权重
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint_data, "model_interrupted.pth")
            torch.save(checkpoint_data, "model_latest.pth") 
            print("✅ 状态已安全保存。")
    
    finally:
        if use_ddp:
            dist.destroy_process_group()
            # .destroy_process_group()释放进程组的通信资源
            # 无论什么情况，这一句都会执行，确保显卡间的 NCCL 通信矩阵被释放。
            # 不加这个，下次跑必报 Address already in use。

# ==============================================================================
# 程序入口
# ==============================================================================
if __name__ == "__main__":
    # 这个脚本不能直接 python train.py 运行！
    # 必须用 torchrun 启动，它会自动帮你：
    #   1. 启动 world_size 个进程（每张 GPU 一个）
    #   2. 设置 LOCAL_RANK、WORLD_SIZE 等环境变量
    #   3. 建立进程间通信
    # 启动命令示例（3 卡）：
    #   torchrun --nproc_per_node=3 train.py
    train()
