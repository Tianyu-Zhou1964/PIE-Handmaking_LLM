# Python 端：负责从多个数据源（JSONL + Parquet）读取语料，按配比采样，
# 将其拼接成一个大字符串列表，一次性传给 Rust。
# Rust 端：接收字符串，将其转化为底层的整数数组（Byte IDs），在纯 Rust 环境下完成所有的"统计 -> 寻找最高频 -> 合并 -> 更新数组"的循环，直到达到目标词表大小。
# Rust 端：将最终炼成的"词表（Vocab）"和"合并规则（Merges）"传回 Python。
# Python 端：将结果保存为 JSON 格式，并封装 encode 和 decode 方法。
import json
import time
import random
import os
import glob
import pandas as pd  # 新增：用于读取 Parquet 格式文件
import custom_bpe    # 我们的 Rust 编译产物
import yaml          # 从 config_zh.yaml 加载配置

# ==============================================================================
# 配置参数：从 config_zh.yaml 加载
# ==============================================================================
with open("config_zh.yaml", "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)
    _tok_cfg = _cfg["tokenizer"]

SAVE_PATH    = _tok_cfg["save_path"]
MERGES_PATH  = _tok_cfg["merges_path"]
VOCAB_SIZE   = _tok_cfg["vocab_size"]
TARGET_CHARS = _tok_cfg["target_chars"]


# ==============================================================================
# 多数据源配置
# 每个数据源指定：路径、格式、字段名、采样字符目标量
# 配比思路：
#   中文(SkyPile)   约 20M 字符 —— 中文是主力，要占大头
#   英文(FineWeb)   约 15M 字符 —— 英文次之
#   代码(StarCoder) 约 10M 字符 —— python/rust/markdown 均分
#   数学(Numina)    约  5M 字符 —— 数量少但领域重要，全量采
# ==============================================================================

def get_bytes_to_unicode_mapping():
    """
    GPT-2 标准的 Byte-to-Unicode 映射法则。
    返回一个字典，将 0-255 的整数字节，无损且 1:1 地映射为可见的 Unicode 字符。
    这样彻底杜绝了“控制字符投毒”和“UTF-8 解码失败导致的文本膨胀”。
    """
    # 收集本来就可见、安全的 ASCII 字符
    # 接下来说的安全就是可见的意思，不可见会在训练过程导致很多麻烦
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    # 33~126：可见ASCII，!"#$...XYZ...xyz  161~172：可见拉丁字符  174~255：可见拉丁字符
    # bs 收集的是 0-255 里本来就已经是可见字符的那些编号。
    # 被排除在外的（也就是不安全的）：
    # 0-32：控制字符（换行\n、退格、空格等）
    # 127：DEL 删除符
    # 128-160：扩展控制字符，放进 JSON 会乱掉
    # 173：软连字符，不可见
    
    cs = bs[:]
    # cs 是映射目标，一开始和 bs 完全一样——意思是安全字节映射到自己。
    # 比如 65 → 65，即字节 65 映射到字符 chr(65) = "A"，保持不变。

    n = 0
    # 对剩下的（控制字符、UTF-8残缺字节）进行偏移映射
    for b in range(256): 
        if b not in bs:  # 找到不安全的字节
            bs.append(b) # 0-187 安全字符，188 开始是不安全字符一直到 255
            cs.append(256 + n) # 加入不安全字符，但是用的是 256 以上的安全 Unicode 区域
            # 我们一共会用到68个Unicode 的拉丁扩展块，各种带帽子、带尾巴的字母，比如 Ā ā Ĉ ĉ Ġ ġ
            # 比如字节 0（空字符，不可见），就映射到 chr(256) = Ā。
            # 字节 1 映射到 chr(257) = ā，以此类推。
            n += 1
    # 组合成字典：{ 0: 'Ā', ..., 65: 'A', ..., 228: 'ä' }
    return {b: chr(c) for b, c in zip(bs, cs)}
    # 从右往左看，zip(bs, cs)把刚才两个列表一一对应做成了字典
    # b: chr(c) for 的结构是字典推导式，得到最后的字典：
    # { 33: '!',  34: '"', ... 126: '~',   # 先是安全的可见ASCII
    #   161: '¡', ... 172: '¬',            # 安全的拉丁字符
    #   174: '®', ... 255: 'ÿ',            # 安全的拉丁字符
    #   0: 'Ā',   1: 'ā',  2: 'Ă', ...    # 危险字节排在后面，映射到偏移字符
    #   10: 'Ċ',  32: 'Ġ', ...             # 换行、空格也在这里
    # }
    # 也就是 0-255 的字节序号分别是哪些字符
    # 字典的 key 是 0-255 的字节编号，value 是对应的可见字符
    # 但顺序不是 0-255，而是安全字节在前，危险字节在后
    # 例如空格(32)和换行(10)都排在靠后的位置，映射到 Ġ、Ċ 这样的偏移字符

DATASET_CONFIGS = _tok_cfg["datasets"]
# yaml 中每个条目已经包含 name, path, format, field, target_chars

# ==============================================================================
# 工具函数：从单个 JSONL 文件流式采样文本
# 保留原有的水塘抽样逻辑，只把字段名参数化了
# ==============================================================================
def sample_from_jsonl(filepath, field, target_chars):
    """
    从单个 JSONL 文件中用水塘抽样读取指定字段的文本，直到凑够 target_chars 个字符。
    """
    random.seed(42)
    k = int(target_chars / 300)
    # 估算需要抽多少条：JSONL 文本平均约 300 字符（比 wiki 短），所以除以 300
    
    reservoir = []
    total_seen = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                text = data.get(field, '').strip()
                if not text:
                    continue
                total_seen += 1
                if len(reservoir) < k:
                    reservoir.append(text)
                else:
                    j = random.randint(0, total_seen - 1)
                    if j < k:
                        reservoir[j] = text
            except Exception:
                continue

    random.shuffle(reservoir)

    texts = []
    current_chars = 0
    for text in reservoir:
        if current_chars >= target_chars:
            break
        texts.append(text)
        current_chars += len(text)

    return texts, current_chars


# ==============================================================================
# 工具函数：从单个 Parquet 文件采样文本
# Parquet 不像 JSONL 支持流式读取，整个文件一次性加载进内存
# 所以我们用 pandas 读完之后再随机打乱采样，不做水塘抽样（文件不超大，可以接受）
# ==============================================================================
def sample_from_parquet(filepath, field, target_chars):
    """
    从单个 Parquet 文件中随机采样文本，直到凑够 target_chars 个字符。
    field 如果是 'problem+solution'，则将两个字段拼接后作为文本。
    """
    df = pd.read_parquet(filepath)
    # 把整个 parquet 文件读成 DataFrame，一行就是一条数据

    if field == "problem+solution":
        # NuminaMath 的特殊处理：把题目和解答拼起来，中间加换行符分隔
        # 这样词表既能见到数学题的描述语言，也能见到解题过程的符号
        df["_text"] = df["problem"].fillna("") + "\n" + df["solution"].fillna("")
        texts_raw = df["_text"].tolist()
    else:
        texts_raw = df[field].dropna().tolist()
        # dropna() 扔掉空值，tolist() 转成 Python 列表

    random.seed(42)
    random.shuffle(texts_raw)
    # 打乱顺序，避免只采到文件开头的数据

    texts = []
    current_chars = 0
    for text in texts_raw:
        if current_chars >= target_chars:
            break
        text = str(text).strip()
        if not text:
            continue
        texts.append(text)
        current_chars += len(text)

    return texts, current_chars


# ==============================================================================
# 主采样函数：遍历一个数据源下的所有文件，汇总到目标字符量为止
# ==============================================================================
def load_source(config):
    """
    根据数据源配置，扫描目录下所有符合格式的文件，
    逐文件采样，直到凑够该数据源的目标字符量。
    """
    name         = config["name"]
    path         = config["path"]
    fmt          = config["format"]
    field        = config["field"]
    target_chars = config["target_chars"]

    print(f"\n[{name}] 开始采样，目标字符数: {target_chars:,}")
    start = time.time()

    # 扫描目录下所有对应格式的文件，排序保证每次运行顺序一致
    if fmt == "jsonl":
        files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    elif fmt == "parquet":
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    else:
        raise ValueError(f"不支持的格式: {fmt}")

    if not files:
        print(f"[{name}] ⚠️  目录下没有找到任何文件，跳过")
        return []

    all_texts    = []
    total_chars  = 0
    remaining    = target_chars  # 还差多少字符没凑够

    for filepath in files:
        if total_chars >= target_chars:
            break  # 已经凑够了，不需要再读更多文件
        
        filename = os.path.basename(filepath)
        per_file_target = min(remaining, target_chars // len(files) + 1)
        # 每个文件的采样目标：大概平摊，但不超过剩余需求
        # 这样能保证多个文件都被采到，而不是全集中在第一个文件

        if fmt == "jsonl":
            texts, chars = sample_from_jsonl(filepath, field, per_file_target)
        else:
            texts, chars = sample_from_parquet(filepath, field, per_file_target)

        all_texts   += texts
        total_chars += chars
        remaining   -= chars
        print(f"  ✓ {filename}：采样 {chars:,} 字符，累计 {total_chars:,} / {target_chars:,}")

    print(f"[{name}] 完成，共采样 {len(all_texts)} 条，{total_chars:,} 字符，耗时 {time.time()-start:.1f}s")
    return all_texts


# ==============================================================================
# 主流程
# ==============================================================================
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # 1. 多源采样：依次读取每个数据源，汇总成一个大列表
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("第一步：多源语料采样")
    print("=" * 60)

    all_texts = []
    for config in DATASET_CONFIGS:
        texts = load_source(config)
        all_texts += texts

    # 全局打乱，避免语料块状分布影响 BPE 统计
    # 比如不打乱的话前 20M 字符全是中文，Rust 统计到的前期 pair 会严重偏向中文
    random.seed(42)
    random.shuffle(all_texts)

    total_chars = sum(len(t) for t in all_texts)
    print(f"\n✅ 采样完成，共 {len(all_texts)} 条文本，{total_chars:,} 字符")

    # -------------------------------------------------------------------------
    # 2. 调用 Rust 引擎训练 BPE（这里完全不动 Rust，只是把列表传进去）
    # -------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("第二步：移交 Rust 引擎训练 BPE")
    print("=" * 60)
    print(f"🚀 将 {len(all_texts)} 条文本移交 Rust，目标词表大小: {VOCAB_SIZE}")
    start_time = time.time()

    _, merges = custom_bpe.train_bpe(all_texts, VOCAB_SIZE)
    # 返回出来的是一个元组，_ 保存空哈希表，用 merges 保存合并规则
    # 回顾一下它是 vec!(((u32, u32), u32)...)

    print(f"🎉 Rust 训练结束！总耗时 {time.time()-start_time:.2f}s。")
    print(f"获得了 {len(merges)} 条合并规则。")

    # ==============================================================================
    # 3. 后处理：利用安全映射构建真正的 Vocab 和 Merges
    # ==============================================================================
    print("\nPython: 正在构建最终 Vocab 映射表...")
    
    # 获取 1:1 字节到可见字符的安全映射
    byte_to_unicode = get_bytes_to_unicode_mapping()
    # 取得我们刚才创建的映射字典：
    # { 33: '!',  34: '"', ... 126: '~',   # 先是安全的可见ASCII
    #   161: '¡', ... 172: '¬',            # 安全的拉丁字符
    #   174: '®', ... 255: 'ÿ',            # 安全的拉丁字符
    #   0: 'Ā',   1: 'ā',  2: 'Ă', ...    # 危险字节排在后面，映射到偏移字符
    #   10: 'Ċ',  32: 'Ġ', ...             # 换行、空格也在这里
    # }
    
    # 直接用映射后的单字符作为基础词表，丢弃 raw bytes
    vocab = {byte_to_unicode[i]: i for i in range(256)}
    # 字典，单字符: 数字

    vocab_inverse = {v: k for k, v in vocab.items()}
    # 反向字典：数字 -> 单字符

    # 重建合并规则
    for (p0, p1), new_id in merges:
        s0 = vocab_inverse[p0]
        s1 = vocab_inverse[p1]
        
        # 直接字符串拼接。由于 1 个字节已被映射为 1 个字符，
        # 所以合并只会让字符串变长，绝对不会出现 <0xE4> 这种奇怪膨胀！
        combined = s0 + s1 
        
        vocab[combined] = new_id
        vocab_inverse[new_id] = combined

    # 现在 vocab 已经是纯文本的安全形式了，不需要再做 try...except 解码！
    readable_vocab = vocab

    # -------------------------------------------------------------------------
    # 特殊词元注册 (保持你的原逻辑不变)
    # -------------------------------------------------------------------------
    current_max_id = max(vocab.values())
    special_tokens = list(_tok_cfg["special_tokens"])  # 从 yaml 读取
    num_reserved = _tok_cfg["num_reserved_tokens"]
    for i in range(num_reserved):
        special_tokens.append(f"<|reserved_{i}|>")

    for i, token in enumerate(special_tokens):
        new_special_id = current_max_id + 1 + i 
        readable_vocab[token] = new_special_id 

    print(f"Python: 已添加 {len(special_tokens)} 个特殊词元")

    # 保存词表 JSON
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(readable_vocab, f, ensure_ascii=False, indent=2)

    print(f"✅ 词表已保存至: {SAVE_PATH}")
    print(f"   最终词表大小: {len(readable_vocab)}")

    # 保存 Merges.txt
    with open(MERGES_PATH, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for (p0, p1), _ in merges:
            token0 = vocab_inverse[p0]
            token1 = vocab_inverse[p1]
            f.write(f"{token0} {token1}\n")

    print(f"✅ 合并规则已保存至: {MERGES_PATH}")