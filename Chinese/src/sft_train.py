# ==============================================================================
# SFT 训练脚本（修复版，DDP 多卡）
# ==============================================================================
# 与预训练脚本的关键区别：
#   1. 数据集：SFTDataset 同时读两个 bin（input_ids + labels），而非滑窗切片
#   2. 损失：labels 里 -100 的位置被 CrossEntropyLoss 自动忽略（ignore_index=-100）
#   3. 模型起点：首次 SFT 要加载预训练好的 dense checkpoint（PRETRAIN_CKPT）
# ==============================================================================

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler  # DDP: 把数据集切成 N 份，每卡只看自己那份
from torch.nn.parallel import DistributedDataParallel         # DDP: 模型包装器，自动同步多卡梯度
import torch.distributed as dist                              # DDP: 进程组通信
from model import Transformer, ModelArgs                      # 手搓的模型结构 (Ep1-Ep18)
import os
import math
import yaml

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_BASE_DIR, "../config_zh.yaml")

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)
    _sft_cfg = _cfg["sft_data"]
    _training_cfg = _cfg["training"]

# ==============================================================================
# 训练配置
# ==============================================================================

# 数据路径（两个 bin 文件，由 sft_dataprocess.py 生成）
INPUT_IDS_PATH = os.path.join(_sft_cfg["output_dir"], "sft_input_ids.bin")
LABELS_PATH    = os.path.join(_sft_cfg["output_dir"], "sft_labels.bin")
SEQ_LEN = _sft_cfg["max_seq_len"]    # 和预处理时保持严格一致！否则 reshape 会错位

BATCH_SIZE    = 20
LEARNING_RATE = 5e-5    # SFT 学习率建议比预训练低一个量级
EPOCHS        = 3       # 小样本 SFT，3 轮让模型充分见到每条数据
VOCAB_SIZE    = 32128   # 词表大小（和 tokenizer 训练时保持一致）

# Checkpoint 路径（统一用绝对路径，避免相对路径因工作目录变化而失效）
CKPT_DIR       = _sft_cfg["output_dir"]
PRETRAIN_CKPT = os.path.join(CKPT_DIR, "PIE-0.2B-dense.pth")       # 首次 SFT 的起点权重
RESUME_FILE   = os.path.join(CKPT_DIR, "sft_latest.pth")           # SFT 断点续训的恢复文件

# ==============================================================================
# 学习率调度：手写 warmup + cosine decay
# T_MAX_STEPS 会在 train() 里根据真实 DataLoader 长度动态覆盖，这里只是占位
# ==============================================================================
WARMUP_STEPS = 50
T_MAX_STEPS  = 16820    # 占位值，train() 中会覆盖
LR_MIN       = 5e-6     # LR 衰减下界（≈ 峰值的 1/10）


def get_lr(step: int) -> float:
    # Warmup 阶段：从 0 线性增长到 LEARNING_RATE
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    # Cosine decay 阶段：从 LEARNING_RATE 平滑衰减到 LR_MIN
    progress = (step - WARMUP_STEPS) / max(T_MAX_STEPS - WARMUP_STEPS, 1)
    progress = min(progress, 1.0)  # 兜底：防止 step 超出 T_MAX_STEPS 时 cos 越界
    return LR_MIN + 0.5 * (LEARNING_RATE - LR_MIN) * (1 + math.cos(math.pi * progress))


# ==============================================================================
# SFT 数据集：直接读两个并行 bin 文件
# ==============================================================================
class SFTDataset(Dataset):
    """
    input_ids.bin : dtype=uint32，shape = [N, SEQ_LEN]
    labels.bin    : dtype=int32 （要容纳 -100），shape = [N, SEQ_LEN]
    两文件行数严格相等，用同一个 idx 分别取即可。
    """
    def __init__(self, input_ids_path, labels_path, seq_len):
        super().__init__()
        # 🌟 关键：input_ids 写入时是 uint32，读取也必须用 uint32，不能写 int32
        # （虽然 token id < 2^31 时两者数值等价，但保持 dtype 一致是好习惯，避免未来踩坑）
        self.input_ids = np.fromfile(input_ids_path, dtype=np.uint32).reshape(-1, seq_len)
        self.labels    = np.fromfile(labels_path,    dtype=np.int32 ).reshape(-1, seq_len)
        self.seq_len   = seq_len

        assert len(self.input_ids) == len(self.labels), \
            f"input_ids({len(self.input_ids)}) 与 labels({len(self.labels)}) 行数不一致！"

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # 转 int64：PyTorch 的 Embedding / CrossEntropyLoss 都要求 LongTensor
        x = torch.from_numpy(self.input_ids[idx].astype(np.int64))
        y = torch.from_numpy(self.labels[idx].astype(np.int64))
        return x, y


# ==============================================================================
# 训练主循环
# ==============================================================================
def train():
    # ── DDP 初始化 ──────────────────────────────────────────────────────────────
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])  # 当前进程对应哪张卡（0/1/2）
    world_size = dist.get_world_size()          # 总共几张卡参与训练
    torch.cuda.set_device(local_rank)           # 绑定当前进程到指定 GPU，避免全挤到 GPU 0
    device = f"cuda:{local_rank}"
    is_master = (local_rank == 0)               # 只让 rank 0 打印和保存，避免日志混乱

    # ── 创建 checkpoint 目录（所有 rank 都执行，exist_ok=True 保证幂等）──────────
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── 1. 准备数据流 ───────────────────────────────────────────────────────────
    dataset = SFTDataset(INPUT_IDS_PATH, LABELS_PATH, SEQ_LEN)

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=local_rank, shuffle=True
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # 动态计算真实的最大 Step 数量，让 Cosine 衰减完美贴合实际训练进度
    global T_MAX_STEPS
    steps_per_epoch = len(loader)
    T_MAX_STEPS = steps_per_epoch * EPOCHS
    if is_master:
        print(f"📊 数据统计：共 {len(dataset)} 条样本")
        print(f"📊 每 Epoch 步数 {steps_per_epoch}，总步数 T_MAX_STEPS = {T_MAX_STEPS}")

    # ── 2. 初始化模型 ───────────────────────────────────────────────────────────
    args = ModelArgs.get_args("tiny", vocab_size=VOCAB_SIZE)
    model = Transformer(args).to(device)

    start_epoch = 0
    global_step = 0
    temp_opt_state = None  # 暂存优化器状态，等 optimizer 创建后再灌进去

    if os.path.exists(RESUME_FILE):
        # SFT 中断续训：优先级最高
        if is_master:
            print(f"🔄 检测到 SFT 断点，从 {RESUME_FILE} 恢复训练...")
        checkpoint = torch.load(RESUME_FILE, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        temp_opt_state = checkpoint['optimizer_state_dict']
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        if is_master:
            print(f"✅ 恢复成功！从 Epoch {start_epoch+1}, Step {global_step} 继续训练")
    else:
        # 首次 SFT：加载预训练底座权重
        if is_master:
            print(f"🚀 首次 SFT，加载预训练权重: {PRETRAIN_CKPT}")
        checkpoint = torch.load(PRETRAIN_CKPT, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        if is_master:
            print("✅ 预训练权重加载完成，开始 SFT")

    # torch.compile：fullgraph=False 容错性更好，仍有 20%+ 提速
    model = torch.compile(model, fullgraph=False)
    # DDP 包装：此时层次为 DDP(CompiledModule(OriginalTransformer))
    # 后续取原模型：model.module._orig_mod
    model = DistributedDataParallel(model, device_ids=[local_rank])

    # ── 3. 优化器与损失函数 ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if temp_opt_state is not None:
        optimizer.load_state_dict(temp_opt_state)
        if is_master:
            print("✅ 优化器动量状态也已完美恢复！")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_master:
        print(f"📦 模型参数量: {total_params:,}")

    # ignore_index=-100：labels 中被 mask 为 -100 的位置自动不计入 loss
    # 这是 SFT 只学回答部分的关键机制
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    if is_master:
        print(f"🎬 开始训练，设备: {device}，总卡数: {world_size}...")
        print("   注意：torch.compile 第一个 step 需要 1~2 分钟编译，请耐心等待...")
    model.train()

    # ── 日志与保存配置 ──────────────────────────────────────────────────────────
    SAVE_INTERVAL = 1000    # 每 1000 步保存一次权重
    LOG_INTERVAL  = 100     # 每 100 步打印一次 Loss
    saved_checkpoints = []

    use_bf16 = True
    scaler = torch.amp.GradScaler('cuda', enabled=(not use_bf16))

    # 🌟 在进入 for 之前先绑定 epoch，避免 Ctrl+C 在第一个 step 之前触发时
    # 异常分支里 `epoch` 还未定义而抛 UnboundLocalError
    epoch = start_epoch

    # 用 try-except-finally 确保异常/中断也能释放 NCCL 资源
    try:
        for epoch in range(start_epoch, EPOCHS):
            sampler.set_epoch(epoch)    # 让每个 epoch 的 shuffle 不一样
            total_loss = 0
            for X, Y in loader:
                global_step += 1

                # 手动调整学习率
                current_lr = get_lr(global_step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                # autocast 只包 forward；CrossEntropy 在外面用 fp32 算，避免数值问题
                with torch.amp.autocast(
                    'cuda', enabled=True,
                    dtype=torch.bfloat16 if use_bf16 else torch.float16,
                ):
                    logits, aux_loss = model(X)

                # 🌟 把 logits 拉回 fp32 再算 loss：
                # CrossEntropy 需要 softmax→log，bf16/fp16 下 logits 偏大会炸
                # --- 修改后的 sft_train.py (推荐写法) ---
                logits = logits.float()

                # 核心对齐逻辑：
                # shift_logits 取从第 0 到倒数第 2 个位置 (预测下一个)
                # shift_labels 取从第 1 到最后一个位置 (真实下一个)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = Y[:, 1:].contiguous()

                main_loss = criterion(
                    shift_logits.view(-1, args.vocab_size), 
                    shift_labels.view(-1)
                )

                # 🌟 aux_loss 兼容处理：
                # 你的 dense 模型（MoE 禁用）可能返回 None 或 0 标量。
                # 只有当它是一个真正需要梯度的 tensor 时才加进总 loss。
                if (aux_loss is not None
                        and torch.is_tensor(aux_loss)
                        and aux_loss.requires_grad):
                    loss = main_loss + 0.01 * aux_loss.float()
                else:
                    loss = main_loss

                # 反向传播
                if use_bf16:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                # 🌟 .detach() 在 GPU 上累加，避免每步 .item() 造成 CPU-GPU 同步阻塞
                total_loss += loss.detach()

                if global_step % LOG_INTERVAL == 0 and is_master:
                    # 只在打日志这一刻才 .item()，把 GPU 数据拉到 CPU
                    current_loss_val = loss.item()
                    print(f"Step {global_step} | Epoch {epoch+1} | "
                          f"Loss: {current_loss_val:.4f} | LR: {current_lr:.2e}")

                # ── 定期保存 checkpoint ──
                if global_step % SAVE_INTERVAL == 0:
                    if is_master:
                        # 剥掉 DDP 和 compile 两层包装，拿到原始 Transformer 权重
                        raw_model = (
                            model.module._orig_mod
                            if hasattr(model.module, "_orig_mod")
                            else model.module
                        )
                        checkpoint_data = {
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': raw_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }
                        # 🌟 统一用绝对路径保存，和 RESUME_FILE 对齐，保证能 resume
                        ckpt_name = f"sft_step_{global_step}.pth"
                        ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
                        torch.save(checkpoint_data, ckpt_path)
                        saved_checkpoints.append(ckpt_path)
                        # 覆写 sft_latest.pth —— 这个文件名必须和 RESUME_FILE 一致！
                        torch.save(checkpoint_data, RESUME_FILE)
                        print(f"💾 Step {global_step} 进度(含优化器状态)已保存: {ckpt_path}")

                        # 只保留最新的 3 个 step ckpt，清理旧文件防止磁盘爆炸
                        if len(saved_checkpoints) > 3:
                            oldest_ckpt = saved_checkpoints.pop(0)
                            if os.path.exists(oldest_ckpt):
                                os.remove(oldest_ckpt)
                                print(f"🧹 清理旧权重: {oldest_ckpt}")

                    # ⚠️ 核心屏障：所有 GPU 在此等待 Rank 0 把文件落盘完成
                    # 否则其他 rank 可能已经冲进下一个 batch，造成 DataLoader 脱节
                    dist.barrier()

            if is_master:
                avg_loss = (total_loss / len(loader)).item()
                print(f"--- Epoch {epoch+1} 完成，平均 Loss: {avg_loss:.4f} ---")

    except KeyboardInterrupt:
        if is_master:
            print("\n⚠️  检测到训练中断，正在紧急保存当前权重...")
            raw_model = (
                model.module._orig_mod
                if hasattr(model.module, "_orig_mod")
                else model.module
            )
            checkpoint_data = {
                'epoch': epoch,  # 有上面的 epoch = start_epoch 兜底，这里不会 UnboundLocalError
                'global_step': global_step,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            interrupted_path = os.path.join(CKPT_DIR, "sft_interrupted.pth")
            torch.save(checkpoint_data, interrupted_path)
            torch.save(checkpoint_data, RESUME_FILE)
            print(f"✅ 状态已安全保存到 {interrupted_path} 和 {RESUME_FILE}")

    finally:
        # 无论成功、报错还是 Ctrl+C，都要释放 NCCL 进程组
        # 否则下次启动会报 "Address already in use"
        dist.destroy_process_group()


# ==============================================================================
# 程序入口
# ==============================================================================
if __name__ == "__main__":
    train()
