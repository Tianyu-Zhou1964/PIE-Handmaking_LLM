import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import json
import multiprocessing
import ctypes
from multiprocessing import Value, Lock
from functools import partial
import pandas as pd
import tempfile
import shutil
import time
import random
import threading
from tqdm import tqdm
import yaml

# 导入我们亲手打磨的无敌 Rust 引擎
import custom_bpe

# ==============================================================================
# 🔥 新增：1B 级别语料配比与目标控制
# 目标总量：10亿 Tokens。
# SkyPile(60%), FineWeb(25%), StarCoder(10%), NuminaMath(5%)
# ==============================================================================

# 全局共享变量（供多进程初始化时继承）
global_counters = {}
global_locks = {}

def init_worker(counters, locks):
    """子进程初始化函数，继承共享的计数器和锁"""
    global global_counters, global_locks
    global_counters = counters
    global_locks = locks

# ==============================================================================
# 1. 引擎初始化：全局只运行一次，避免 IO 风暴和重复计算
# ==============================================================================
def load_global_tokenizer_data(vocab_path, merges_path):
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
        if k in ["<pad>", "<bos>", "<eos>"] or k.startswith("<|")
    }
    
    return vocab, merges_list, special_tokens_dict

# ==============================================================================
# 2. Sanity Check 解码函数
# ==============================================================================
def get_bytes_to_unicode_mapping():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

def decode_tokens(tokens, vocab):
    vocab_inverse = {v: k for k, v in vocab.items()}
    unicode_to_byte = {v: k for k, v in get_bytes_to_unicode_mapping().items()}

    text = ""
    byte_array = bytearray()
    for token in tokens:
        token_str = vocab_inverse.get(token, "")
        if token_str.startswith("<") and token_str.endswith(">"):
            if byte_array:
                text += byte_array.decode('utf-8', errors='replace')
                byte_array.clear()
            text += token_str
        else:
            for char in token_str:
                if char in unicode_to_byte:
                    byte_array.append(unicode_to_byte[char])
                    
    if byte_array:
        text += byte_array.decode('utf-8', errors='replace')
    return text

# ==============================================================================
# 3. 核心处理逻辑 (单个子进程独立运行)
# ==============================================================================
def process_single_file(file_path, merges_list, special_tokens_dict, bos_id, eos_id, temp_dir, category_targets):
    pid = os.getpid()
    
    # 识别当前文件属于哪个数据集大类
    category = None
    if "SkyPile" in file_path: category = "SkyPile"
    elif "FineWeb" in file_path: category = "FineWeb"
    elif "starcoderdata" in file_path: category = "StarCoder"
    elif "NuminaMath" in file_path: category = "NuminaMath"
    
    if not category:
        return 0, None
        
    target_for_cat = category_targets[category]
    
    # 如果该类别的 Token 已经达标，直接跳过该文件处理
    if global_counters[category].value >= target_for_cat:
        return 0, None
    
    try:
        tokenizer = custom_bpe.Tokenizer(merges_list, special_tokens_dict)
        temp_filename = os.path.join(temp_dir, f"{os.path.basename(file_path)}_{pid}.bin")
        total_tokens_in_file = 0
        
        # 减小 Buffer 以增加 flush 频率，防止超量过多
        BUFFER_LIMIT = 50000 
        buffer = []

        def flush_buffer():
            """核心拦截器：将 buffer 存入磁盘前，精确计算并扣减额度，超出的部分截断丢弃"""
            nonlocal total_tokens_in_file, buffer
            if not buffer: return
            
            with global_locks[category]:
                current_val = global_counters[category].value
                # 如果被其他进程抢先干满配额了，清空并退出
                if current_val >= target_for_cat:
                    buffer.clear()
                    return
                
                # 计算还能塞下多少 Token
                allowed = target_for_cat - current_val
                if len(buffer) > allowed:
                    buffer = buffer[:allowed] # 极限截断，保证一丝一毫都不超发！
                
                # 扣除全局额度
                global_counters[category].value += len(buffer)
                
            with open(temp_filename, 'ab') as f:
                np.array(buffer, dtype=np.uint32).tofile(f)
            total_tokens_in_file += len(buffer)
            buffer.clear()

        # --- 针对不同数据集的提取逻辑 (带全局配额中断检查) ---
        if category == "FineWeb":
            df = pd.read_parquet(file_path, engine='pyarrow', columns=['text'])
            for text in df['text'].dropna():
                if global_counters[category].value >= target_for_cat: break
                text_str = str(text).strip()
                if text_str:
                    buffer.extend([bos_id] + tokenizer.encode(text_str) + [eos_id])
                    if len(buffer) >= BUFFER_LIMIT: flush_buffer()
                
        elif category == "SkyPile":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if global_counters[category].value >= target_for_cat: break
                    try:
                        data = json.loads(line)
                        text_str = data.get('text', '').strip()
                        if text_str:
                            buffer.extend([bos_id] + tokenizer.encode(text_str) + [eos_id])
                            if len(buffer) >= BUFFER_LIMIT: flush_buffer()
                    except:
                        continue
                        
        elif category == "StarCoder":
            df = pd.read_parquet(file_path, engine='pyarrow', columns=['content'])
            for content in df['content'].dropna():
                if global_counters[category].value >= target_for_cat: break
                content_str = str(content).strip()
                if content_str and "generated by" not in content_str.lower():
                    buffer.extend([bos_id] + tokenizer.encode(content_str) + [eos_id])
                    if len(buffer) >= BUFFER_LIMIT: flush_buffer()
                    
        elif category == "NuminaMath":
            df = pd.read_parquet(file_path, engine='pyarrow', columns=['problem', 'solution'])
            df = df.dropna(subset=['problem', 'solution'])
            for _, row in df.iterrows():
                if global_counters[category].value >= target_for_cat: break
                text_str = f"{str(row['problem']).strip()}\n\n{str(row['solution']).strip()}"
                if text_str:
                    buffer.extend([bos_id] + tokenizer.encode(text_str) + [eos_id])
                    if len(buffer) >= BUFFER_LIMIT: flush_buffer()

        flush_buffer()

        if total_tokens_in_file == 0:
            return 0, None
        return total_tokens_in_file, temp_filename
        
    except Exception as e:
        # tqdm.write(f"[PID {pid}] ❌ 处理 {os.path.basename(file_path)} 时出错: {e}")
        return 0, None

# ==============================================================================
# 新增：守护线程监控进度条
# ==============================================================================
def progress_monitor(counters, targets, total_tokens, stop_event):
    """单独开一个线程用来刷新美观的多重进度条"""
    with tqdm(total=total_tokens, desc="🚀 整体进度 (1B Tokens)", unit="tok", position=0, leave=True) as pbar_total:
        
        pbars = {}
        for i, (cat, target) in enumerate(targets.items()):
            pbars[cat] = tqdm(total=target, desc=f"  ├─ {cat:10}", unit="tok", position=i+1, leave=True)
            
        while not stop_event.is_set():
            time.sleep(0.5)
            total_now = 0
            for cat in targets.keys():
                val = counters[cat].value
                pbars[cat].n = val
                pbars[cat].refresh()
                total_now += val
                
            pbar_total.n = total_now
            pbar_total.refresh()
            
            if total_now >= total_tokens:
                break
                
        # 最后刷新一次保证满条
        total_now = 0
        for cat in targets.keys():
            val = counters[cat].value
            pbars[cat].n = val
            pbars[cat].refresh()
            total_now += val
        pbar_total.n = total_now
        pbar_total.refresh()
        
        for p in pbars.values(): p.close()

# ==============================================================================
# 4. 主调度函数
# ==============================================================================
def process(args):
    DATA_ROOT      = args.data_root
    OUTPUT_DIR     = args.output_dir
    OUTPUT_PATH    = os.path.join(OUTPUT_DIR, "all.bin")
    TOKENIZER_PATH = args.tokenizer_path
    MERGES_PATH    = args.merges_path
    NUM_PROCESSES  = args.num_processes

    # ↓ 加在这里，此时 args 已经存在了
    TARGET_TOTAL_TOKENS = args.total_tokens

    try:
        with open("config_zh.yaml", "r", encoding="utf-8") as _f:
            _dp_cfg = yaml.safe_load(_f).get("dataprocess", {})
            _ratios = _dp_cfg.get("category_ratios", {})
    except FileNotFoundError:
        _ratios = {}

    _default_ratios = {"SkyPile": 0.60, "FineWeb": 0.25, "StarCoder": 0.10, "NuminaMath": 0.05}
    _ratios = {**_default_ratios, **_ratios}

    CATEGORY_TARGETS = {
        cat: int(TARGET_TOTAL_TOKENS * ratio)
        for cat, ratio in _ratios.items()
    }

    if not os.path.exists(DATA_ROOT):
        print(f"找不到数据根目录 {DATA_ROOT}")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("加载全局词表与合并规则...")
    vocab, merges_list, special_tokens_dict = load_global_tokenizer_data(TOKENIZER_PATH, MERGES_PATH)
    bos_id = vocab.get("<bos>", 1)
    eos_id = vocab.get("<eos>", 2)

    all_files = []
    for root, dirs, files in os.walk(DATA_ROOT):
        for file in files:
            if file.endswith('.parquet') or file.endswith('.jsonl'):
                all_files.append(os.path.join(root, file))

    random.seed(42)
    random.shuffle(all_files) # 打乱文件，让各个数据集均匀分配给 128 个核心
    
    total_files = len(all_files)
    print(f"扫描完毕，共找到 {total_files} 个数据文件。")
    print(f"服务器 CPU 核心数: {multiprocessing.cpu_count()}，当前开启进程数: {NUM_PROCESSES}")

    temp_dir = tempfile.mkdtemp(prefix="pretrain_bins_")
    
    # 初始化跨进程的安全计数器和锁
    counters = {cat: Value(ctypes.c_longlong, 0) for cat in CATEGORY_TARGETS.keys()}
    locks = {cat: Lock() for cat in CATEGORY_TARGETS.keys()}

    # 启动进度条监控线程
    print("\n" + "="*50)
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
    target=progress_monitor, args=(counters, CATEGORY_TARGETS, TARGET_TOTAL_TOKENS, stop_event))

    monitor_thread.start()

    worker = partial(
        process_single_file, 
        merges_list=merges_list, 
        special_tokens_dict=special_tokens_dict, 
        bos_id=bos_id, 
        eos_id=eos_id,
        temp_dir=temp_dir,
        category_targets=CATEGORY_TARGETS,
    )
    
    valid_temp_files = []
    
# === 请替换 process() 函数中的这段多进程调度代码 ===
    
    with multiprocessing.Pool(processes=NUM_PROCESSES, initializer=init_worker, initargs=(counters, locks)) as pool:
        for tokens_count, temp_file in pool.imap_unordered(worker, all_files, chunksize=1):
            if temp_file is not None:
                valid_temp_files.append(temp_file)
            
            # 删除暴力的 pool.terminate()！
            # 因为我们在子进程内部写了拦截机制，一旦达标，子进程会以毫秒级的速度跳过后续文件。
            # 让 imap_unordered 自然结束，能确保每一个生成的碎片文件都被加入 valid_temp_files！

    # 通知进度条线程结束
    stop_event.set()
    monitor_thread.join()
    
    # 统计最终结果
    final_total_tokens = sum(counters[cat].value for cat in CATEGORY_TARGETS.keys())

    print("\n📦 所有目标分配结束！正在将碎片数据无缝合并为最终的 all.bin...")
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH) 
        
    with open(OUTPUT_PATH, 'wb') as outfile:
        # 为了避免 IO 阻塞，可以使用较大缓冲区拷贝
        for tmp_file in valid_temp_files:
            if os.path.exists(tmp_file):
                with open(tmp_file, 'rb') as infile:
                    shutil.copyfileobj(infile, outfile, length=1024*1024*16) # 16MB Chunk

    shutil.rmtree(temp_dir)

    print(f"\n" + "="*50)
    print(f"🎉 预处理大功告成 (1B Target) 🎉")
    print(f"总处理 Token 数量: {final_total_tokens:,}")
    for cat in CATEGORY_TARGETS.keys():
        print(f"  ├─ {cat:10}: {counters[cat].value:,} Tokens")
        
    print(f"最终大文件保存至: {OUTPUT_PATH}")
    print(f"最终大文件大小: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024 / 1024:.2f} GB")

    # ==============================================================================
    # 5. 最后一步的定心丸：Sanity Check
    # ==============================================================================
    print("\n🔍 正在进行最终的 Sanity Check (抽取前 100 个 Token 进行解码测试)...")
    try:
        with open(OUTPUT_PATH, 'rb') as f:
            sample_tokens = np.fromfile(f, dtype=np.uint32, count=100).tolist()
            
        print(f"前 100 个 Token IDs: {sample_tokens}")
        decoded_text = decode_tokens(sample_tokens, vocab)
        print(f"解码结果:\n{'-'*60}\n{decoded_text}\n{'-'*60}")
        print("💡 提示：如果你在这里看到了正常人类语言（夹杂一点 <bos> 和 <eos>），说明你的大模型炼丹材料完美无瑕！")
    except Exception as e:
        print(f"Sanity Check 失败，错误信息: {e}")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Pretokenize corpus for LLM pretraining")
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Root dir containing SkyPile / FineWeb / StarCoder / NuminaMath subfolders"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data",
        help="Output directory for all.bin (default: ./data)"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default="./data/tokenizer/tokenizer.json",
        help="Path to tokenizer vocab JSON"
    )
    parser.add_argument(
        "--merges_path", type=str, default="./data/tokenizer/merges.txt",
        help="Path to BPE merges file"
    )
    parser.add_argument(
        "--num_processes", type=int, default=128,
        help="Number of parallel worker processes (default: 128)"
    )
    parser.add_argument(
        "--total_tokens", type=int, default=1_000_000_000,
        help="Target total token count (default: 1B)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process(args)