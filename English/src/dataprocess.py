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

# Import our hand-crafted, invincible Rust engine
import custom_bpe

# ==============================================================================
# 🔥 NEW: 1B level corpus ratio and target control
# Target total: 1 billion Tokens.
# SkyPile(60%), FineWeb(25%), StarCoder(10%), NuminaMath(5%)
# ==============================================================================

# Global shared variables (inherited during multiprocessing initialization)
global_counters = {}
global_locks = {}

def init_worker(counters, locks):
    """Child process initialization function, inherits shared counters and locks"""
    global global_counters, global_locks
    global_counters = counters
    global_locks = locks

# ==============================================================================
# 1. Engine initialization: Runs only once globally to avoid IO storms and redundant computations
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
# 2. Sanity Check decoding function
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
# 3. Core processing logic (runs independently in a single child process)
# ==============================================================================
def process_single_file(file_path, merges_list, special_tokens_dict, bos_id, eos_id, temp_dir, category_targets):
    pid = os.getpid()
    
    # Identify which dataset category the current file belongs to
    category = None
    if "SkyPile" in file_path: category = "SkyPile"
    elif "FineWeb" in file_path: category = "FineWeb"
    elif "starcoderdata" in file_path: category = "StarCoder"
    elif "NuminaMath" in file_path: category = "NuminaMath"
    
    if not category:
        return 0, None
        
    target_for_cat = category_targets[category]
    
    # If the token target for this category is met, skip processing this file
    if global_counters[category].value >= target_for_cat:
        return 0, None
    
    try:
        tokenizer = custom_bpe.Tokenizer(merges_list, special_tokens_dict)
        temp_filename = os.path.join(temp_dir, f"{os.path.basename(file_path)}_{pid}.bin")
        total_tokens_in_file = 0
        
        # Reduce buffer size to increase flush frequency and prevent excessive token generation
        BUFFER_LIMIT = 50000 
        buffer = []

        def flush_buffer():
            """Core interceptor: Before saving the buffer to disk, precisely calculate and deduct the quota, truncating and discarding any excess"""
            nonlocal total_tokens_in_file, buffer
            if not buffer: return
            
            with global_locks[category]:
                current_val = global_counters[category].value
                # If another process has already fulfilled the quota, clear and exit
                if current_val >= target_for_cat:
                    buffer.clear()
                    return
                
                # Calculate how many more tokens can fit
                allowed = target_for_cat - current_val
                if len(buffer) > allowed:
                    buffer = buffer[:allowed] # Extreme truncation to ensure not a single token exceeds the limit!
                
                # Deduct global quota
                global_counters[category].value += len(buffer)
                
            with open(temp_filename, 'ab') as f:
                np.array(buffer, dtype=np.uint32).tofile(f)
            total_tokens_in_file += len(buffer)
            buffer.clear()

        # --- Extraction logic for different datasets (with global quota interruption check) ---
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
        # tqdm.write(f"[PID {pid}] ❌ Error processing {os.path.basename(file_path)}: {e}")
        return 0, None

# ==============================================================================
# NEW: Daemon thread to monitor progress bars
# ==============================================================================
def progress_monitor(counters, targets, total_tokens, stop_event):
    """Start a separate thread to refresh beautiful multi-level progress bars"""
    with tqdm(total=total_tokens, desc="🚀 Overall Progress (1B Tokens)", unit="tok", position=0, leave=True) as pbar_total:
        
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
                
        # Final refresh to ensure the bar is full
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
# 4. Main scheduling function
# ==============================================================================
def process(args):
    DATA_ROOT      = args.data_root
    OUTPUT_DIR     = args.output_dir
    OUTPUT_PATH    = os.path.join(OUTPUT_DIR, "all.bin")
    TOKENIZER_PATH = args.tokenizer_path
    MERGES_PATH    = args.merges_path
    NUM_PROCESSES  = args.num_processes

    # ↓ Added here, args already exists at this point
    TARGET_TOTAL_TOKENS = args.total_tokens

    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    _config_path = os.path.join(_BASE_DIR, "../config_en.yaml")
    try:
        with open(_config_path, "r", encoding="utf-8") as _f:
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
        print(f"Cannot find data root directory {DATA_ROOT}")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading global vocabulary and merge rules...")
    vocab, merges_list, special_tokens_dict = load_global_tokenizer_data(TOKENIZER_PATH, MERGES_PATH)
    bos_id = vocab.get("<bos>", 1)
    eos_id = vocab.get("<eos>", 2)

    all_files = []
    for root, dirs, files in os.walk(DATA_ROOT):
        for file in files:
            if file.endswith('.parquet') or file.endswith('.jsonl'):
                all_files.append(os.path.join(root, file))

    random.seed(42)
    random.shuffle(all_files) # Shuffle files to distribute datasets evenly across 128 cores
    
    total_files = len(all_files)
    print(f"Scan complete, found {total_files} data files.")
    print(f"Server CPU cores: {multiprocessing.cpu_count()}, current running processes: {NUM_PROCESSES}")

    temp_dir = tempfile.mkdtemp(prefix="pretrain_bins_")
    
    # Initialize cross-process safe counters and locks
    counters = {cat: Value(ctypes.c_longlong, 0) for cat in CATEGORY_TARGETS.keys()}
    locks = {cat: Lock() for cat in CATEGORY_TARGETS.keys()}

    # Start progress bar monitoring thread
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
    
# === Please replace this multiprocessing scheduling code in the process() function ===
    
    with multiprocessing.Pool(processes=NUM_PROCESSES, initializer=init_worker, initargs=(counters, locks)) as pool:
        for tokens_count, temp_file in pool.imap_unordered(worker, all_files, chunksize=1):
            if temp_file is not None:
                valid_temp_files.append(temp_file)
            
            # Remove the brutal pool.terminate()!
            # Because we wrote an interception mechanism inside the child process, once the target is met, child processes will skip subsequent files at millisecond speed.
            # Let imap_unordered finish naturally to ensure every generated chunk file is added to valid_temp_files!

    # Notify progress bar thread to terminate
    stop_event.set()
    monitor_thread.join()
    
    # Tally final results
    final_total_tokens = sum(counters[cat].value for cat in CATEGORY_TARGETS.keys())

    print("\n📦 All targets allocated! Seamlessly merging chunk data into the final all.bin...")
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH) 
        
    with open(OUTPUT_PATH, 'wb') as outfile:
        # To avoid IO blocking, a larger buffer can be used for copying
        for tmp_file in valid_temp_files:
            if os.path.exists(tmp_file):
                with open(tmp_file, 'rb') as infile:
                    shutil.copyfileobj(infile, outfile, length=1024*1024*16) # 16MB Chunk

    shutil.rmtree(temp_dir)

    print(f"\n" + "="*50)
    print(f"🎉 Preprocessing successfully completed (1B Target) 🎉")
    print(f"Total processed tokens: {final_total_tokens:,}")
    for cat in CATEGORY_TARGETS.keys():
        print(f"  ├─ {cat:10}: {counters[cat].value:,} Tokens")
        
    print(f"Final large file saved to: {OUTPUT_PATH}")
    print(f"Final large file size: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024 / 1024:.2f} GB")

    # ==============================================================================
    # 5. The final reassurance: Sanity Check
    # ==============================================================================
    print("\n🔍 Performing final Sanity Check (Extracting first 100 Tokens for decoding test)...")
    try:
        with open(OUTPUT_PATH, 'rb') as f:
            sample_tokens = np.fromfile(f, dtype=np.uint32, count=100).tolist()
            
        print(f"First 100 Token IDs: {sample_tokens}")
        decoded_text = decode_tokens(sample_tokens, vocab)
        print(f"Decoded result:\n{'-'*60}\n{decoded_text}\n{'-'*60}")
        print("💡 Hint: If you see normal human language here (mixed with some <bos> and <eos>), it means your LLM training materials are flawless!")
    except Exception as e:
        print(f"Sanity Check failed, error info: {e}")

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