# --- START OF FILE sft_dataprocess.py ---
#
# SFT 数据预处理脚本（修复版）
# ==============================
#
#
#
# Part 1 (L23-32)  : 环境与依赖              → 跳过
# Part 2 (L34-98)  : 配置参数 + 特殊 token   → 重点 ⭐⭐⭐
# Part 3 (L101-139): 加载 tokenizer          → 一笔带过
# Part 4 (L142-228): 数据加载 + Blending     → 工程细节 (留作下集钩子)
# Part 5 (L231-287): encode_conversation    → 灵魂 ⭐⭐⭐⭐⭐
# Part 6 (L290-331): 解码工具函数            → 跳过
# Part 7 (L334-443): 主流程 + Sanity Check   → 收尾 ⭐⭐
#
# 输出：
#   sft_input_ids.bin  —— shape: [N, MAX_SEQ_LEN]，dtype: uint32
#   sft_labels.bin     —— shape: [N, MAX_SEQ_LEN]，dtype: int32（-100 需要有符号）
#
# 两个文件行数严格对齐，DataLoader 用同一个 idx 分别取就行。
# ==============================================================================

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import random
import numpy as np
from tqdm import tqdm
import custom_bpe  # 我们亲手打磨的 Rust BPE 引擎
from dataclasses import dataclass
from typing import List
import yaml

# ==============================================================================
# 配置参数 —— 全部从 ../config_zh.yaml 的 sft_data 节读取
# 改参数请直接动 yaml，不要回头改这里
# ==============================================================================

# ---- 加载 SFT 数据预处理配置 ----
# 参照 inference.py 的取法：脚本在 train/ 之类的子目录里，config 在上一级
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_BASE_DIR, "../config_zh.yaml")

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)
    _sft_cfg = _cfg["sft_data"]   # SFT 数据相关配置

# ---- 路径 ----
OUTPUT_DIR     = _sft_cfg["output_dir"]
TOKENIZER_PATH = _sft_cfg["tokenizer_path"]
MERGES_PATH    = _sft_cfg["merges_path"]

# ---- 序列与复现 ----
MAX_SEQ_LEN = _sft_cfg["max_seq_len"]   # padding / 截断的目标长度，0.2B 建议 512
RANDOM_SEED = _sft_cfg["random_seed"]

# ---- 采样规模 ----
TOTAL_SAMPLES          = _sft_cfg["total_samples"]
BELLE_OVERSAMPLE_RATIO = _sft_cfg["belle_oversample_ratio"]  # Belle 预读放大系数

# ---- Data Blending 配置（list of dicts，结构同原硬编码）----
DATA_SOURCES = _sft_cfg["data_sources"]

# ---- 特殊 token —— ChatML 风格 ----
# 用 <|user|> / <|assistant|> / <|endofturn|> 明确标记「谁在说话 + 什么时候说完」
_st = _sft_cfg["special_tokens"]
BOS_TOKEN       = _st["bos"]
EOS_TOKEN       = _st["eos"]
PAD_TOKEN       = _st["pad"]
USER_TOKEN      = _st["user"]        # 指令开始标记
ASSISTANT_TOKEN = _st["assistant"]   # 回答开始标记
EOT_TOKEN       = _st["endofturn"]   # 回答结束标记（推理时可作为停止符）

# IGNORE_INDEX 是 PyTorch CrossEntropy 的协议常量，不是"配置"，留在代码里
IGNORE_INDEX = -100

# ==============================================================================
# Data Class 定义 —— 抽象出对话实体（结构定义，不是配置，留在代码里）
# ==============================================================================
@dataclass
class Message:
    role: str       # 通常为 "user" 或 "assistant"
    content: str    # 文本内容

@dataclass
class Conversation:
    messages: List[Message]
    source: str = "unknown"  # 记录数据来源，方便追踪


# ==============================================================================
# 1. 加载 tokenizer（复用预训练脚本的逻辑，保持风格统一）
# ==============================================================================
def load_tokenizer(vocab_path, merges_path):
    """
    读取词表 JSON 和 merges.txt，构造 custom_bpe.Tokenizer。
    返回: tokenizer 实例, vocab 字典, 各特殊 token 的 id
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    merges_list = []
    with open(merges_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                t0, t1 = parts
                p0, p1 = vocab[t0], vocab[t1]
                new_id = vocab[t0 + t1]
                merges_list.append(((p0, p1), new_id))

    special_tokens_dict = {
        k: v for k, v in vocab.items()
        if k in [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN] or k.startswith("<|")
    }

    tokenizer = custom_bpe.Tokenizer(merges_list, special_tokens_dict)

    # 直接用 [] 硬取 —— 这些特殊 token 词表里必须有，没有就报 KeyError 早发现早治疗
    bos_id       = vocab[BOS_TOKEN]        # 32065
    eos_id       = vocab[EOS_TOKEN]        # 32066
    pad_id       = vocab[PAD_TOKEN]        # 32064
    user_id      = vocab[USER_TOKEN]       # 32067
    assistant_id = vocab[ASSISTANT_TOKEN]  # 32068
    eot_id       = vocab[EOT_TOKEN]        # 32070

    return tokenizer, vocab, bos_id, eos_id, pad_id, user_id, assistant_id, eot_id


# ==============================================================================
# 2. 数据加载：Data Blending 调度与组装
# ==============================================================================
def load_alpaca(path, n_sample, rng):
    """提取 Alpaca 格式，转换为标准 Conversation 多轮结构"""
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read().strip()
    try:
        data = json.loads(raw)
    except:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]

    rng.shuffle(data)
    if len(data) < n_sample:
        times = (n_sample // len(data)) + 1
        data = (data * times)[:n_sample]
    else:
        data = data[:n_sample]

    conversations =[]
    for item in data:
        inst = item.get("instruction", "").strip()
        inp  = (item.get("input") or "").strip()
        out  = item.get("output", "").strip()
        if not inst or not out: continue

        user_content = f"{inst}\n{inp}" if inp else inst
        
        # 将单步问答包装成多轮 Message 列表
        msgs =[
            Message(role="user", content=user_content),
            Message(role="assistant", content=out)
        ]
        conversations.append(Conversation(messages=msgs, source="alpaca"))

    return conversations

def load_belle(path, n_sample, rng):
    """提取 BELLE 格式，同样转换为标准 Conversation 结构"""
    raw =[]
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try: raw.append(json.loads(line.strip()))
            except: continue
            if len(raw) >= n_sample * BELLE_OVERSAMPLE_RATIO: break

    rng.shuffle(raw)
    if len(raw) < n_sample and len(raw) > 0:
        # 数据不足,循环复制补齐(允许重复采样)
        times = (n_sample // len(raw)) + 1
        raw = (raw * times)[:n_sample]
    else:
        raw = raw[:n_sample]

    conversations =[]
    for item in raw:
        inst = item.get("instruction", "").strip()
        inp  = (item.get("input") or "").strip()
        out  = (item.get("output") or "").strip()
        if not inst or not out: continue

        user_content = f"{inst}\n{inp}" if inp else inst
        msgs =[
            Message(role="user", content=user_content),
            Message(role="assistant", content=out)
        ]
        conversations.append(Conversation(messages=msgs, source="belle"))

    return conversations

def load_blended_data(sources, total_samples, rng):
    """
    根据配置的权重大盘（Data Blending），自动计算各个数据集应该采样的数量，
    并分发给对应的读取函数。
    """
    total_weight = sum(s["weight"] for s in sources)
    all_conversations =[]
    
    for s in sources:
        # 按权重分配实际采样数
        n_sample = int(total_samples * (s["weight"] / total_weight))
        print(f"\n📂 加载 {s['name']} (格式: {s['format']}, 目标采样: {n_sample} 条)...")
        
        if s["format"] == "alpaca":
            convs = load_alpaca(s["path"], n_sample, rng)
        elif s["format"] == "belle":
            convs = load_belle(s["path"], n_sample, rng)
        else:
            raise ValueError(f"未知的格式: {s['format']}")
            
        print(f"   实际读到: {len(convs)} 条")
        all_conversations.extend(convs)
        
    rng.shuffle(all_conversations)
    return all_conversations


# ==============================================================================
# 3. 核心：将 Conversation 对象编码成 input_ids 和 labels
# ==============================================================================
def encode_conversation(conv: Conversation, tokenizer, bos_id, eos_id, pad_id,
                        user_id, assistant_id, eot_id, max_seq_len):
    """
    输入: 一个 Conversation 实例 (包含任意轮数的对话)
    输出: (input_ids, labels)
    
    动态拼接机制：
        对于 user 轮次：[<|user|>, msg..., <|assistant|>] -> labels 全部填 -100
        对于 assistant 轮次：[msg..., <|endofturn|>]     -> labels 取原始 id 进行 loss 计算
    全系列首部加 bos，尾部加 eos。
    """
    full_ids = [bos_id]
    full_labels = [IGNORE_INDEX]  # bos 也不算 loss
    # full_ids[i] = 第 i 个位置喂给模型的 token id，IGNORE_INDEX就是-100
    # full_labels[i] = 第 i 个位置期望模型预测出的 token id（或者 -100 表示"这个位置不考核"）
    
    for msg in conv.messages:
        msg_ids = tokenizer.encode(msg.content)
        # msg 长这样比如<|user|>问题内容 或者 <|assistant|>content
        # .encode把字符串转变为数字列表
        
        if msg.role == "user":
            # 构建：<|user|> + 问题内容 + <|assistant|> (提示模型开始回答)
            turn_ids = [user_id] + msg_ids + [assistant_id]
            # user_id其实就是vocab[USER_TOKEN]       # 32067, 对应 <|user|>
            # 同理，assistant_id = vocab[ASSISTANT_TOKEN]  # 32068, 对应 <|assistant|>

            turn_labels = [IGNORE_INDEX] * len(turn_ids)
            # 用户输入的全部内容不参与 Loss 计算，我们创建一个等同于用户输入长度的-100 列表，等下给它加到 labels 里
            
        elif msg.role == "assistant":
            # 构建：回答内容 + <|endofturn|>
            # 注意，每当大模型回复了完整内容，末尾就会加一个 eot，表示一轮话（turn）讲完了
            # 每次跟大模型讲话基本上都是多轮对话，所以会出现多个eot
            # 多轮对话结束之后才会出现 eos，标志整个对话结束
            turn_ids = msg_ids + [eot_id]

            # 大模型的回答算 Loss
            turn_labels = turn_ids[:]
            
        else:
            continue
            
        full_ids.extend(turn_ids)
        full_labels.extend(turn_labels)

    # 整个多轮对话结束，追加 eos
    full_ids.append(eos_id)
    full_labels.append(eos_id)  # eos 算作回答的一部分，需要预测以学懂停止

    assert len(full_ids) == len(full_labels), "input_ids 和 labels 长度必须一致"
    # 这一步防止出错，要是不对就不继续了

    # ---------- 截断与 Padding ----------
    if len(full_ids) > max_seq_len:
        # 长了后续内容直接截断，max_seq_len我设置的是 512
        full_ids    = full_ids[:max_seq_len]
        full_labels = full_labels[:max_seq_len]

    pad_len = max_seq_len - len(full_ids)
    # 短了就补齐
    input_ids = full_ids    + [pad_id]        * pad_len
    labels    = full_labels + [IGNORE_INDEX]  * pad_len  # pad 位置忽略 loss

    return (
        np.array(input_ids, dtype=np.uint32),
        # 输入转换成u32即可，没有负的
        np.array(labels,    dtype=np.int32),
        # 标签有-100 要用 int32 存
    )


# ==============================================================================
# 4. Sanity Check 辅助：解码 token id 还原文本
# ==============================================================================
def get_bytes_to_unicode():
    """BPE 的 byte→unicode 映射，用于解码还原原始文本"""
    bs = (list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def decode_ids(ids, vocab):
    """把 token id 列表还原成字符串，用于 sanity check"""
    inv_vocab = {v: k for k, v in vocab.items()}
    u2b = {v: k for k, v in get_bytes_to_unicode().items()}
    text = ""
    buf  = bytearray()
    for tid in ids:
        tid_int = int(tid)
        # -100 是 labels 的 mask 值，解码时跳过
        if tid_int < 0:
            continue
        tok = inv_vocab.get(tid_int, "")
        if tok.startswith("<") and tok.endswith(">"):
            if buf:
                text += buf.decode("utf-8", errors="replace")
                buf.clear()
            text += tok
        else:
            for ch in tok:
                if ch in u2b:
                    buf.append(u2b[ch])
    if buf:
        text += buf.decode("utf-8", errors="replace")
    return text


# ==============================================================================
# 5. 主函数
# ==============================================================================
def process():
    rng = random.Random(RANDOM_SEED)
    # rng = random.Random(RANDOM_SEED)：创建一个带种子的独立随机数生成器。42 保证可复现
    # random.Random 保证是局部的，不会影响全局
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # os.makedirs(..., exist_ok=True)：建好输出目录，已存在就跳过，不报错。

    # ── 加载 tokenizer ──────────────────────────────────────────────────────────
    print("📖 加载 tokenizer...")
    tokenizer, vocab, bos_id, eos_id, pad_id, user_id, assistant_id, eot_id = load_tokenizer(
        TOKENIZER_PATH, MERGES_PATH
    )
    print(f"   词表大小: {len(vocab)}")
    print(f"   特殊 token id — bos:{bos_id}  eos:{eos_id}  pad:{pad_id}  "
          f"user:{user_id}  assistant:{assistant_id}  eot:{eot_id}")

    # ── 1. 使用 Blending 调度器加载所有数据 ─────────────────────────────────────────
    all_conversations = load_blended_data(DATA_SOURCES, TOTAL_SAMPLES, rng)
    print(f"\n✅ 合计样本数: {len(all_conversations)} 条，已根据权重混合并随机打乱")
    # 这一步把两个数据集按 1:1 权重混好打乱，得到 50000 条 Conversation 实例。

    # ── 2. 编码每条对话样本 ────────────────────────────────────────────────────────
    print(f"\n⚙️  开始编码（MAX_SEQ_LEN={MAX_SEQ_LEN}）...")

    out_ids_path = os.path.join(OUTPUT_DIR, "sft_input_ids.bin")
    out_lbl_path = os.path.join(OUTPUT_DIR, "sft_labels.bin")

    n_total = 0 # 统计总计处理的条数
    n_truncated = 0 # 统计被截断的条目
    n_skipped = 0 # 统计跳过条目
    total_resp_tokens = 0 # 统计最终条目数

    with open(out_ids_path, 'wb') as f_ids, open(out_lbl_path, 'wb') as f_lbl:
        for conv in tqdm(all_conversations, desc="编码中"):
        # 这里我们把 all_conversations 包装进 tqdm 用来显示进度条
            
            # 使用新写的 encode_conversation 处理对话类
            input_ids, labels = encode_conversation(
                conv, tokenizer,
                bos_id, eos_id, pad_id, user_id, assistant_id, eot_id,
                MAX_SEQ_LEN
            )
            
            # 统计丢弃和截断逻辑
            if (input_ids == pad_id).sum() == 0:   # 没有 pad → 说明被截断了
                n_truncated += 1
            
            # 判断有效性：如果整条 labels 都是 -100，说明模型全被截断或者这本身就是一条废数据，丢弃
            resp_tokens = int((labels != IGNORE_INDEX).sum())
            if resp_tokens == 0:
                n_skipped += 1
                continue
                
            total_resp_tokens += resp_tokens

            input_ids.tofile(f_ids)
            labels.tofile(f_lbl)
            n_total += 1

    # ── 打印统计 ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🎉 SFT 数据预处理完成！")
    print("=" * 60)
    n_all = max(len(all_conversations), 1)
    print(f"  成功写入样本数   : {n_total:,}")
    print(f"  被截断样本数     : {n_truncated:,}  ({n_truncated / n_all * 100:.1f}%)")
    print(f"  因超长丢弃样本数 : {n_skipped:,}")
    print(f"  实际参与 loss 的 token 总量: {total_resp_tokens:,}")
    print(f"  平均每条回答长度 : {total_resp_tokens / max(n_total, 1):.1f} tokens")
    print(f"\n  输出文件:")
    print(f"    input_ids → {out_ids_path}")
    print(f"             大小: {os.path.getsize(out_ids_path) / 1024 / 1024:.2f} MB")
    print(f"    labels    → {out_lbl_path}")
    print(f"             大小: {os.path.getsize(out_lbl_path) / 1024 / 1024:.2f} MB")

    # ── Sanity Check ─────────────────────────────────────────────────────────────
    print("\n🔍 Sanity Check：解码前 5 条，肉眼确认格式...")
    print("-" * 60)

    all_ids = np.fromfile(out_ids_path, dtype=np.uint32).reshape(n_total, MAX_SEQ_LEN)
    all_lbl = np.fromfile(out_lbl_path, dtype=np.int32 ).reshape(n_total, MAX_SEQ_LEN)
    # 读取刚才保存好的二进制文件抽查一下

    for i in range(min(5, n_total)):
        ids = all_ids[i]
        lbl = all_lbl[i]

        # 找到第一个 pad，只显示有效部分
        pad_positions = np.where(ids == pad_id)[0]
        # np.where 返回的是元组，([bool数组], )

        effective_len = int(pad_positions[0]) if len(pad_positions) > 0 else MAX_SEQ_LEN
        # 找到第一个 pad_id 的位置，它的索引就是有效长度

        print(f"\n【样本 {i+1}】有效长度 = {effective_len} tokens")
        print(f"  input_ids 解码: {decode_ids(ids[:effective_len], vocab)[:200]}...")

        # 找到 labels 中第一个非 -100 的位置，即回答开始处
        non_mask = np.where(lbl != IGNORE_INDEX)[0]
        if len(non_mask) == 0:
            print("  ⚠️  整条样本 labels 全为 -100，这是异常数据！请检查 encode 逻辑。")
            continue
        resp_start = int(non_mask[0])
        print(f"  labels mask 到第 {resp_start} 位，回答部分解码:")
        print(f"  {decode_ids(lbl[resp_start:effective_len], vocab)[:200]}...")

        # 验证：labels 的 mask 范围内确实全是 -100
        assert (lbl[:resp_start] == IGNORE_INDEX).all(), \
            f"❌ 样本 {i+1} 的指令部分 labels 不全是 -100，有 bug！"
        print(f"  ✅ 指令部分 labels 全为 -100，格式正确")

    print("\n" + "=" * 60)
    print("💡 如果你看到 <bos><|user|>指令内容<|assistant|>回答内容<|endofturn|><eos>，说明格式完美！")
    print("   接下来只需要 SFTDataset 读这两个 bin 文件，就可以开始训练了。")
    print("=" * 60)

    print("\n🔍 搜索 Self_cognition 数据...")
    identity_keyword = "派派"  # 换成你 identity.json 里的特征词，比如你的模型名

    found = 0
    for i in range(n_total):
        decoded = decode_ids(all_lbl[i][all_lbl[i] != IGNORE_INDEX], vocab)
        if identity_keyword in decoded:
            found += 1
            if found <= 3:  # 只打印前3条
                print(f"\n【命中样本 {i}】")
                ids = all_ids[i]
                pad_pos = np.where(ids == pad_id)[0]
                eff_len = int(pad_pos[0]) if len(pad_pos) > 0 else MAX_SEQ_LEN
                print(decode_ids(ids[:eff_len], vocab)[:300])

    print(f"\n共找到 {found} 条含 '{identity_keyword}' 的样本")


if __name__ == "__main__":
    process()