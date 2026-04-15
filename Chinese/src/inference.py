# ==============================================================================
# 推理脚本：让训练好的大模型"开口说话"
# ==============================================================================
# 训练阶段我们做的事情是：喂数据 → 算 Loss → 反向传播 → 更新参数（Ep22）
# 推理阶段完全不同：没有 Loss，没有梯度，没有参数更新
# 我们只做一件事——第2集讲的"文字接龙"：
# 给模型一段开头，让它一个字一个字地往后"猜"，猜到停为止
# ==============================================================================

import torch
import torch.nn.functional as F
import custom_bpe  # 导入我们用 Rust 手搓的 BPE 引擎编译产物（Ep21）
from model import Transformer, ModelArgs  # 导入我们手搓的模型结构（Ep1-Ep18 理论 + Ep19 实现）
import yaml
import os

# ---- 加载推理配置 ----
# 把超参数（温度、top_k、top_p 等）都丢进 YAML 配置文件里
# 好处：改参数不用动代码，直接改配置文件就行
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_BASE_DIR, "../config_zh.yaml")

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)
    _inf_cfg = _cfg["inference"] # 推理相关配置（模型路径、采样参数等）
    _model_cfg = _cfg["model"] # 模型结构配置（层数、头数、维度等）


# ==============================================================================
# 1. 采样算法：Top-K + Top-P 联合过滤
# ==============================================================================

# ---- 背景回顾 ----
# 第2集我们讲过：大模型的输出是一个概率分布（经过 Softmax 后）
# 最简单的策略是"贪心"：每次选概率最大的那个 token
# 但贪心生成的文本非常死板、重复，像复读机一样
#
# Top-K + Top-P 的思想是：不只看概率最大的，而是从"比较靠谱的候选"中随机抽样
# 这样生成的文本既不会胡说八道（排除了低概率的垃圾选项），又保留了创造力

def top_k_top_p_filtering(logits, top_k=50, top_p=0.9, filter_value=-float('Inf')):
    """
    对 logits（原始打分，还没过 Softmax）进行过滤，只保留"靠谱"的候选词
    logits 是我们的输入内容经过前向传播结束后尚未进行归一化为概率的原始得分

    
    参数：
    - logits: 一维张量，形状 (vocab_size,)，每个词的原始得分
    - top_k: 只保留得分最高的 K 个词（其余设为 -∞）
    - top_p: 在 top_k 筛选后，按概率从高到低累加，累加超过 p 的部分砍掉
    - filter_value: 被砍掉的词设为这个值（-∞ 经过 Softmax 后概率为 0）
    
    两道筛子的组合：
      第一道（Top-K）：粗筛，只留前 K 名，淘汰绝大多数不靠谱的选项
      第二道（Top-P）：精筛，在前 K 名中按概率累加到超过一个阈值 P，然后砍掉尾部"凑数的"
    """
    assert logits.dim() == 1

    # ---- 第一道筛子：Top-K ----
    # torch.topk 返回最大的 K 个值及其索引
    # [0] 取值，[-1] 取第 K 大的值（也就是门槛）
    # 所有低于这个门槛的 logit 都被设为 -∞
    if top_k > 0:
        threshold = torch.topk(logits, top_k).values[-1]
        # threshold 表示门槛，torch.topk(logits, top_k) 返回一个由两个数字列表构成的元组(values, indices)
        # 其中 values 是从大到小排好序的前 top_k 个值，比如：[12.3, 10.1, 8.7, ..., 3.2]  # 长度 50，降序排列
        # 咱们直接用-1从右往左取第一个，也就是取最后一个，也就是最小值，把它设为门槛threshold
        logits[logits < threshold] = filter_value
        # logits < threshold 会生成一个列表，里面每个值都是布尔类型，也就是 True or False
        # 然后logits[这个布尔列表]会根据这个布尔列表的 True 或者 False 却执行赋值，如果是 True 就赋值，如果不是就不赋值
        # 这样就只保留了 top_k 个 logits 得分，其他全被置为 -inf 了，其实是 -11111111111111111111111111111111（32 个 1）

    # ---- 第二道筛子：Top-P（核采样, Nucleus Sampling）----
    # 思路：把剩余候选按概率从高到低排列，逐个累加概率
    # 一旦累加值超过 p（比如 0.9），后面的全部砍掉
    # 这样保证我们只从"占据 90% 概率质量"的核心候选中采样
    if top_p > 0.0:
        # 按得分从高到低排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # 计算排序后的累积概率
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # 它具体怎么实现，我们来举个例子感受一下，就取4个吧
        # 第一步：排序后的 softmax 累积概率
        # sorted_probs     = [0.50, 0.30, 0.15, 0.05]
        # cumulative_probs = [0.50, 0.80, 0.95, 1.00]

        sorted_indices_to_remove = cumulative_probs > top_p
        # 第二步：cumulative_probs > top_p 标记哪些位置"已经超过了"
        # sorted_probs             = [0.50, 0.30, 0.15, 0.05]
        # cumulative_probs         = [0.50, 0.80, 0.95, 1.00]
        # > 0.9?                   = [False, False, True, True]
        # sorted_indices_to_remove = [False, False, True, True]
        # 注意：index=2 的 token（概率 0.15）本身让累积值从 0.80 跳到 0.95，越过了阈值。
        # 但我们其实想保留它，因为是它完成了 90% 覆盖的最后一块。

        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        # 这里我们让 sorted_indices_to_remove第二个到最后一个元素 变为 第一个到倒数第二个元素
        # 相当于整体向右移动了一个单位
        sorted_indices_to_remove[0] = False 
        # 此时第一个元素值直接改成 False，因为重复了
        # [False, False, True, True] 变为：
        # [False, False, False, True]

        logits[sorted_indices[sorted_indices_to_remove]] = filter_value
        # sorted_indices         = [4, 2, 0, 1, 3]   # 排序后每个位置对应原始的哪个下标
        # sorted_indices_to_remove = [F, F, T, T, T]  # 哪些位置要被移除
        # 结果 = [0, 1, 3] 
        # logits[[0, 1, 3]] = filter_value

    return logits


# ==============================================================================
# 2. 初始化加载：模型 + Rust BPE 引擎
# ==============================================================================
# 推理的第一步：把训练好的模型从磁盘上"请"出来
# 训练时我们存了 checkpoint（Ep22），现在要把权重灌回模型骨架里
# ==============================================================================


# ---- 设备选择 ----
# "auto" 模式下优先用 NVIDIA GPU，其次检测 Apple M 系列芯片的 MPS 加速，最后退回 CPU
_device_setting = _inf_cfg["device"]
if _device_setting == "auto":
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
else:
    device = _device_setting

# ---- 构建模型骨架 ----
# 这一步只是搭了个"空壳"——所有参数都是随机初始化的
# 就像 Ep22 开头说的"一堆随机参数的橡皮泥"，还不能用
args = ModelArgs.from_yaml(_CONFIG_PATH)
model = Transformer(args).to(device)

# ---- 加载训练好的权重（读档） ----
# 训练时我们用 torch.save 存了一个字典（Ep22 第五块），
# 里面包含 model_state_dict（模型权重）、optimizer_state_dict（优化器状态）等
# 推理时只需要模型权重，优化器状态不用管了——推理不更新参数
CHECKPOINT_PATH = os.path.join(_BASE_DIR, _inf_cfg["checkpoint_path"])

try:
    print(f"正在加载权重文件: {CHECKPOINT_PATH}")

    # map_location=device：加载时直接把张量映射到当前设备
    # 如果模型是在 GPU 上训练的，但你现在只有 CPU，这个参数会自动帮你搬过来
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
     # 兼容多种保存格式：
    #   格式1: {"model": state_dict, ...}
    #   格式2: {"model_state_dict": state_dict, ...}（Ep22 的训练脚本用这种）
    #   格式3: 直接就是 state_dict（最简陋的保存方式）
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # 把权重灌进空壳模型——从"橡皮泥"变成"捏好的形状"
    model.load_state_dict(state_dict)

    print("✅ 权重加载成功！")
except Exception as e:
    print(f"❌ 权重加载失败: {e}")
    exit()

# ---- 推理优化 ----
if _inf_cfg["use_fp16"]:
    model.half()
    # .half() = .to(torch.float16)，把所有参数从 32 位浮点数压缩到 16 位
    # 训练时我们用 bf16 混合精度（Ep22），推理时直接全部转 fp16
    # 好处：显存占用减半，推理速度翻倍
    # 为什么推理可以比训练更激进地用低精度？
    # 因为推理不需要算梯度！梯度对精度敏感，但前向传播的数值误差可以容忍

model.eval()
# .eval() 模式：告诉模型"现在是考试，不是训练"
# 它会关闭 Dropout（训练时随机丢弃神经元防过拟合，推理时不需要）
# 以及切换 BatchNorm/LayerNorm 的行为（用全局统计量而非批次统计量）


# ---- 加载 Rust BPE 统一引擎 ----
# ==============================================================================
# 这些全部内化到 Rust 的 BpeEngine 里了：
#   - vocab.json 由 serde_json 直接读取解析（自动处理嵌套格式）
#   - merges.txt 由 Rust 逐行 split + 查 vocab 拿 ID（比 Python 快一个数量级）
#   - 特殊 token 判断在 Rust 内部一次遍历完成
#   - 编码侧和解码侧共享同一份 vocab 解析结果，零冗余
# ==============================================================================
print("正在装载 Rust 高性能 BPE 引擎...")
vocab_path = os.path.join(_BASE_DIR, _inf_cfg["vocab_path"])
# 词表路径

merges_path = os.path.join(_BASE_DIR, _inf_cfg["merges_path"])
#合并规则路径

engine = custom_bpe.BpeEngine(vocab_path, merges_path)
# 执行这一行我们会得到 BpeEngine 对象实例，同时终端会打印三行日志：
# Rust BpeEngine: vocab 加载完成，共 XXXXX 个词条
# Rust BpeEngine: merges 加载完成，共 XXXXX 条规则，XX 个特殊词元
# Rust BpeEngine 初始化完成！

# 这个对象上有五个可调用的方法：
# engine.encode(text) → List[int] 字符串->向量 比如 "床前明月光" → [1234, 567, 89, ...]
# engine.decode(ids) → ids->字符串 [1234, 567, 89, ...] -> "床前明月光"
# engine.decode_stream(token_id) → str # 逐个 token 解码，专门处理推理时逐 token 生成的场景。
# 内部维护一个字节缓冲区，遇到汉字这种多字节字符还没凑齐时返回空字符串 ""，凑齐了再一起吐出来。
# engine.reset_stream() → None 用来清空 decode_stream 的内部字节缓冲区。
# engine.token_to_id(token_str) → Optional[int]
# 用来拿特殊 token 的 ID，比如 engine.token_to_id("<eos>") 拿到终止符 ID，推理循环里用来判断停止条件。

# 拿到 <eos>（End of Sequence）的 ID，后面生成时遇到它就停下
eos_id = engine.token_to_id("<eos>")
if eos_id is None:
    raise ValueError("词表中未找到 <eos>，请检查 vocab.json 里特殊 token 的写法")

print(f"\n=============================================")
print(f"{_inf_cfg['model_name']} 已上线！(输入 'quit' 退出)")
print(f"=============================================")


# ==============================================================================
# 3. 推理主循环：文字接龙的工程实现
# ==============================================================================
# 回顾 Ep2 的伪代码：
#   input_text = "床前明月光"
#   while 没写完:
#       next_word = 大模型.predict(input_text)
#       input_text = input_text + next_word
#
# 下面就是这段伪代码的真实版本。
# 区别在于：真实版本需要处理分词、采样策略、重复惩罚、流式输出等工程细节。
# ==============================================================================

while True:
    prompt = input("\n输入开头 (输入 'quit' 退出): ")
    if prompt.lower() in ['quit', 'exit', '退出']:
        # .lower()的作用是把 prompt 字符串的大写字母转换成小写，兼容大小写
        print("👋 拜拜！")
        break
    
    # ---- 编码：自然语言 → Token ID 序列 ----
    # engine.encode 内部完成：正则预分词 → 字节化 → BPE 合并 → 特殊 token 匹配
    input_ids = engine.encode(prompt)
    # 输入一个 Python 字符串，比如 "床前明月光"，交给 Rust 根据词表和合并规则切分、合并，然后吐出数字
    # 输出一个 Python 列表，里面都是 Token ID，比如 [2847, 1156, 9634, 7789, 4815]

    x = torch.tensor([input_ids], device=device)
    # 把 Token ID 列表包装成 PyTorch 张量，送入 GPU
    # 形状：(1, seq_len)——batch_size=1（推理时一次只处理一个样本）

    print("🥧派派续写内容: ", end="", flush=True)
    # end=""让 print 打印完不换行，否则你将看到一行一行的 token
    # flush=True 的意思是：写完立刻强制刷新，别等。

    engine.reset_stream()
    # 重置 Rust 解码器的流式缓冲区
    # 每次新对话都要 reset，清掉上一轮可能残留的字节碎片
    # 只需调用 Rust 端的 reset_stream()，零分配，零开销
    generated_ids = []
    # 记录生成的所有 token ID

    # ---- 推理核心：文字接龙生成循环 ----
    with torch.no_grad():
    # torch.no_grad()：告诉 PyTorch 不要构建计算图、不要算梯度
    # 推理时不需要反向传播（Ep13），关掉计算图能省大量显存和计算量
    # 如果不加 no_grad 跑推理，模型会老老实实把所有中间激活值都存着，等一个永远不会来的 .backward()。
    # 开了之后显存开销巨大，关掉大幅降低显存
        for step in range(_inf_cfg["max_new_tokens"]):
            # ---- Step 1: 前向传播 ----
            # 把当前的 token 序列喂给模型，得到 logits
            # logits 形状：(1, seq_len, vocab_size)
            # logits 就是模型对"下一个字是什么"的原始打分，词表里的每个字都有一个 logit 得分
            # x = torch.tensor([input_ids], device=device)
            outputs = model(x)
            if isinstance(outputs, tuple):
                logits = outputs[0] # 如果模型返回的是 (logits, aux_loss) 元组，取第一个
            else:
                logits = outputs # 如果不是那直接拿来用就完事了
            
            next_token_logits = logits[0, -1, :].clone()
            # 第 1 个批次，最后一个 seqlen 的 logits 分布才是下一个词的 logits 分布
            # 假设你输入了 "我 爱 吃 火锅"，4 个 token，那输出 (bsz, 4, vocab_size) 里：
            # logits[0, 0, :] → 看了"我"，预测位置 1（即"爱"）的分布
            # logits[0, 1, :] → 看了"我 爱"，预测位置 2（即"吃"）的分布
            # logits[0, 2, :] → 看了"我 爱 吃"，预测位置 3（即"火锅"）的分布
            # logits[0, 3, :] → 看了"我 爱 吃 火锅"，预测位置 4——这才是还没出现的下一个新 token
            # 比方说吐出来一个句号"。" 因为一句话讲完了

            # 所以 logits[0, -1, :] 是最后一个位置的输出，它看到了整个输入序列，
            # 预测的是序列之后的第一个新 token。这正是推理时我们需要的东西。
            # 训练时所有位置的 logits 都有用（因为每个位置都和它的下一个 token 构成一个训练样本）
            # 但推理生成时，你只关心最后一个位置的输出，因为只有它在预测真正未知的下一个词。 

            # ---- Step 2: 重复惩罚 ----
            # 没有这一步，模型很容易陷入"复读机"模式：
            #   "我喜欢你我喜欢你我喜欢你我喜欢你..."
            # 惩罚方法：回头看最近 repetition_window 个 token，
            # 如果某个 token 已经出现过，就把它的 logit 减掉一个惩罚值
            # 这样它在后续 Softmax 中的概率就会降低，不容易被再次选中
            window = generated_ids[-_inf_cfg["repetition_window"]:]
            for token_id in set(window):
                next_token_logits[token_id] -= _inf_cfg["repetition_penalty"]

            # ---- Step 3: Top-K + Top-P 过滤 ----
            # 把不靠谱的候选词砍掉，只保留核心候选
            next_token_logits = top_k_top_p_filtering(
                next_token_logits, top_k=_inf_cfg["top_k"], top_p=_inf_cfg["top_p"]
            )
            # 经过这一步，我们只保留前50个最大概率的候选词（TopK）
            # 并且前五十个词中加到0.9位置概率就结束，可能只有 7-8 个词（TopP）

            # ---- Step 4: 温度缩放 + Softmax → 概率分布 ----
            probs = F.softmax(next_token_logits / _inf_cfg["temperature"], dim=-1)
            # probs是 probabilities 的缩写
            # 还记得 Ep2 的 Softmax 吗？把原始打分变成合法的概率分布（全为正，加起来=1）
            # temperature（温度）控制"创造力"：
            #   温度 < 1：分布变尖锐，高概率词更突出 → 输出更保守、更确定
            #   温度 = 1：原始分布，不做调整
            #   温度 > 1：分布变平缓，低概率词也有机会 → 输出更"天马行空"
            # 数学上就是在 Softmax 前把 logits 除以温度：
            #   Softmax(logits / T)
            #   T 越大，logits 差距被压缩，分布越均匀
            # 假设词表只有三个词，logits 是 [10, 9, 1]：
            # 原始（T=1）：Softmax([10, 9, 1])   → [0.731, 0.269, 0.000]  # 第一个词几乎垄断
            # 除以T=0.1： Softmax([100, 90, 10]) → [≈1.0,  ≈0.0,  ≈0.0]  # 更极端，赢家通吃
            # 你观察到了 100 和 90 不是才差10吗，怎么 softmax 完了 90 那一项怎么也变成路边的一条野狗了
            # 那我问你，e^100和e^90相差的 e^10大概有多大？e^10 ≈ 22026，
            # 其实相差两万多倍，是不是确实就是路边的一条了？
            # 除以T=10：Softmax([1, 0.9, 0.1]) → [0.433, 0.391, 0.176]  # 差距缩小，小词进入视野
            # e^1和e^0.9和e^0.1次方e^1 ≈ 2.718 e^0.9 ≈ 2.460 e^0.1 ≈ 1.105
            # 三个值比较接近，所以 Softmax 之后概率分布接近均匀——这就是高温的效果。
            # 这时候模型说好听点叫有了发散性思维，说难听点就可能会发癫
            # 所以在比如写诗、脑洞创作用高温，做题、写代码用低温

            # ---- Step 5: 随机抽样 ----
            next_id = torch.multinomial(probs, num_samples=1).item()
            # torch.multinomial：根据概率分布随机抽一个 token
            # 注意：这不是"选概率最大的"（那是贪心），而是"按概率随机抽"
            # 概率越大越容易被抽中，但小概率词也有机会，这就是生成文本多样性的来源
            
            # ---- Step 6: 终止判断 ----
            # 如果模型生成了 <eos>（End of Sequence），说明它认为话说完了该停了
            if next_id == eos_id:
                break
            
            # ---- Step 7: 拼接 ----
            x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
            generated_ids.append(next_id)
            # 把刚生成的 token 拼到输入序列末尾，作为下一次前向传播的输入
            # 这就是 Ep2 讲的"Output 变成下一次的 Input"
            # x 的长度每一步 +1：(1, n) → (1, n+1) → (1, n+2) → ...
            
            # ---- Step 8: 流式解码输出 ----
            text_chunk = engine.decode_stream(next_id)
            if text_chunk:
                print(text_chunk, end="", flush=True)
            # engine.decode_stream(token_id) → str # 逐个 token 解码，专门处理推理时逐 token 生成的场景。
            # 内部维护一个字节缓冲区，遇到汉字这种多字节字符还没凑齐时返回空字符串 ""，凑齐了再一起吐出来。
        # 我输入"床前"，然后开始计算
        # 第一次计算，得到明，输入变为床前明
        # 第二次计算，得到月，输入变为床前明月
        # 第三次计算，得到光，输入变为床前明月光
        # 第四次计算，得到eos，模型觉得话说完了直接结束运行
        # 这就是这台文字接龙机的全貌，他们管这叫做“自回归“
            
    print("\n" + "-"*30)
    # 换行并打印一行横杠区分每一轮对话