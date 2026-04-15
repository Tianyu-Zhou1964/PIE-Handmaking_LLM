# Python side: responsible for reading corpus from multiple data sources (JSONL + Parquet),
# sampling according to mixing ratios, concatenating everything into one large list of strings,
# and passing it all at once to Rust.
# Rust side: receives the strings, converts them into underlying integer arrays (Byte IDs),
# and in a pure-Rust environment runs the full loop of "count pairs -> find most frequent -> merge -> update array"
# until the target vocabulary size is reached.
# Rust side: passes the final "Vocab" and "Merges" back to Python.
# Python side: saves the results as JSON and wraps encode/decode methods.
import json
import time
import random
import os
import glob
import pandas as pd  # Added: for reading Parquet format files
import custom_bpe    # Our compiled Rust artifact
import yaml          # Load configuration from config_zh.yaml

# ==============================================================================
# Configuration: loaded from config_zh.yaml
# ==============================================================================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_config_path = os.path.join(_BASE_DIR, "../config_en.yaml")
with open(_config_path, "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)
    _tok_cfg = _cfg["tokenizer"]

SAVE_PATH    = _tok_cfg["save_path"]
MERGES_PATH  = _tok_cfg["merges_path"]
VOCAB_SIZE   = _tok_cfg["vocab_size"]
TARGET_CHARS = _tok_cfg["target_chars"]


# ==============================================================================
# Multi-source dataset configuration.
# Each source specifies: path, format, field name, and target character count.
# Mixing rationale:
#   Chinese (SkyPile)   ~20M chars — Chinese is the primary language, takes the largest share
#   English (FineWeb)   ~15M chars — English is secondary
#   Code (StarCoder)    ~10M chars — python/rust/markdown split evenly
#   Math (Numina)       ~ 5M chars — small quantity but domain-critical, sample fully
# ==============================================================================

def get_bytes_to_unicode_mapping():
    """
    GPT-2 standard Byte-to-Unicode mapping rule.
    Returns a dict that losslessly maps every integer byte in 0-255 to a visible Unicode character (1:1).
    This completely eliminates "control character poisoning" and "text bloat from UTF-8 decode failures".
    """
    # Collect ASCII characters that are already visible and safe.
    # "Safe" here means visible — invisible characters cause many problems during training.
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    # 33~126:  visible ASCII,  !"#$...XYZ...xyz
    # 161~172: visible Latin characters
    # 174~255: visible Latin characters
    # bs collects the byte numbers in 0-255 that are already visible characters.
    # Those excluded (i.e. unsafe):
    # 0-32:    control characters (newline \n, backspace, space, etc.)
    # 127:     DEL character
    # 128-160: extended control characters; putting them in JSON causes corruption
    # 173:     soft hyphen, invisible

    cs = bs[:]
    # cs is the mapping target, initially identical to bs —
    # meaning safe bytes map to themselves.
    # For example 65 → 65, i.e. byte 65 maps to chr(65) = "A", unchanged.

    n = 0
    # Apply an offset mapping to the remaining bytes (control chars, incomplete UTF-8 bytes).
    for b in range(256):
        if b not in bs:      # Found an unsafe byte
            bs.append(b)     # 0-187 are safe characters; from 188 onward unsafe characters up to 255
            cs.append(256 + n)  # Register the unsafe byte, but map it to a safe Unicode range above 256.
            # We use 68 Unicode characters from the Latin Extended block —
            # letters with hats, tails, etc., such as Ā ā Ĉ ĉ Ġ ġ.
            # For example, byte 0 (null character, invisible) maps to chr(256) = Ā.
            # Byte 1 maps to chr(257) = ā, and so on.
            n += 1
    # Combine into a dict: { 0: 'Ā', ..., 65: 'A', ..., 228: 'ä' }
    return {b: chr(c) for b, c in zip(bs, cs)}
    # Reading right-to-left: zip(bs, cs) pairs the two lists one-to-one,
    # and the dict comprehension b: chr(c) for produces the final dict:
    # { 33: '!',  34: '"', ... 126: '~',   # safe visible ASCII first
    #   161: '¡', ... 172: '¬',            # safe Latin characters
    #   174: '®', ... 255: 'ÿ',            # safe Latin characters
    #   0: 'Ā',   1: 'ā',  2: 'Ă', ...    # dangerous bytes at the end, mapped to offset chars
    #   10: 'Ċ',  32: 'Ġ', ...             # newline, space are also here
    # }
    # That is: byte numbers 0-255 each correspond to some character.
    # The dict key is the byte number (0-255); the value is the corresponding visible character.
    # But the order is not 0-255 — safe bytes come first, dangerous bytes come after.
    # For example, space (32) and newline (10) appear near the end,
    # mapped to offset characters like Ġ and Ċ.

DATASET_CONFIGS = _tok_cfg["datasets"]
# Each entry in the yaml already contains: name, path, format, field, target_chars.

# ==============================================================================
# Utility: stream-sample text from a single JSONL file.
# Retains the original reservoir-sampling logic; only parameterizes the field name.
# ==============================================================================
def sample_from_jsonl(filepath, field, target_chars):
    """
    Reads the specified field from a single JSONL file using reservoir sampling,
    until target_chars characters have been collected.
    """
    random.seed(42)
    k = int(target_chars / 300)
    # Estimate how many records to sample: JSONL texts average ~300 characters
    # (shorter than Wikipedia), so divide by 300.

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
# Utility: sample text from a single Parquet file.
# Unlike JSONL, Parquet doesn't support streaming reads — the whole file is loaded into memory at once.
# So we read it with pandas, then randomly shuffle and sample (file sizes are acceptable, no reservoir needed).
# ==============================================================================
def sample_from_parquet(filepath, field, target_chars):
    """
    Randomly samples text from a single Parquet file until target_chars characters are collected.
    If field is 'problem+solution', the two fields are concatenated as the text.
    """
    df = pd.read_parquet(filepath)
    # Read the entire Parquet file into a DataFrame; each row is one data record.

    if field == "problem+solution":
        # Special handling for NuminaMath: concatenate problem and solution with a newline.
        # This way the vocabulary sees both the natural-language problem description
        # and the symbolic notation used in the solution.
        df["_text"] = df["problem"].fillna("") + "\n" + df["solution"].fillna("")
        texts_raw = df["_text"].tolist()
    else:
        texts_raw = df[field].dropna().tolist()
        # dropna() drops null values; tolist() converts to a Python list.

    random.seed(42)
    random.shuffle(texts_raw)
    # Shuffle to avoid sampling only from the beginning of the file.

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
# Main sampling function: iterates over all files under one data source,
# accumulating text until the target character count is reached.
# ==============================================================================
def load_source(config):
    """
    Given a data source config, scans all files of the matching format in the directory
    and samples file by file until that source's target character count is reached.
    """
    name         = config["name"]
    path         = config["path"]
    fmt          = config["format"]
    field        = config["field"]
    target_chars = config["target_chars"]

    print(f"\n[{name}] Starting sampling, target character count: {target_chars:,}")
    start = time.time()

    # Scan all files of the matching format in the directory;
    # sort to ensure a consistent order across runs.
    if fmt == "jsonl":
        files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    elif fmt == "parquet":
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    if not files:
        print(f"[{name}] ⚠️  No files found in directory, skipping")
        return []

    all_texts    = []
    total_chars  = 0
    remaining    = target_chars  # How many characters are still needed

    for filepath in files:
        if total_chars >= target_chars:
            break  # Already have enough, no need to read more files.

        filename = os.path.basename(filepath)
        per_file_target = min(remaining, target_chars // len(files) + 1)
        # Per-file sampling target: roughly evenly distributed, but not exceeding remaining need.
        # This ensures multiple files are sampled from, rather than exhausting everything from the first file.

        if fmt == "jsonl":
            texts, chars = sample_from_jsonl(filepath, field, per_file_target)
        else:
            texts, chars = sample_from_parquet(filepath, field, per_file_target)

        all_texts   += texts
        total_chars += chars
        remaining   -= chars
        print(f"  ✓ {filename}: sampled {chars:,} chars, cumulative {total_chars:,} / {target_chars:,}")

    print(f"[{name}] Done. Sampled {len(all_texts)} records, {total_chars:,} chars in {time.time()-start:.1f}s")
    return all_texts


# ==============================================================================
# Main pipeline
# ==============================================================================
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # 1. Multi-source sampling: read each data source in turn, merge into one large list.
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Multi-source corpus sampling")
    print("=" * 60)

    all_texts = []
    for config in DATASET_CONFIGS:
        texts = load_source(config)
        all_texts += texts

    # Global shuffle to avoid block-structured corpus distribution skewing BPE statistics.
    # Without shuffling, the first 20M characters would all be Chinese,
    # and the pair frequencies Rust counts early on would be heavily biased toward Chinese.
    random.seed(42)
    random.shuffle(all_texts)

    total_chars = sum(len(t) for t in all_texts)
    print(f"\n✅ Sampling complete. {len(all_texts)} texts, {total_chars:,} characters total")

    # -------------------------------------------------------------------------
    # 2. Call the Rust engine to train BPE (Rust is untouched; we just pass in the list).
    # -------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Step 2: Hand off to Rust engine for BPE training")
    print("=" * 60)
    print(f"🚀 Passing {len(all_texts)} texts to Rust. Target vocab size: {VOCAB_SIZE}")
    start_time = time.time()

    _, merges = custom_bpe.train_bpe(all_texts, VOCAB_SIZE)
    # Returns a tuple; _ holds an empty hash table, merges holds the merge rules.
    # Recall the format: vec!(((u32, u32), u32)...)

    print(f"🎉 Rust training complete! Total time: {time.time()-start_time:.2f}s.")
    print(f"Obtained {len(merges)} merge rules.")

    # ==============================================================================
    # 3. Post-processing: build the true Vocab and Merges using the safe mapping.
    # ==============================================================================
    print("\nPython: Building final Vocab mapping table...")

    # Obtain the 1:1 byte-to-visible-character safe mapping.
    byte_to_unicode = get_bytes_to_unicode_mapping()
    # The mapping dict we just created:
    # { 33: '!',  34: '"', ... 126: '~',   # safe visible ASCII first
    #   161: '¡', ... 172: '¬',            # safe Latin characters
    #   174: '®', ... 255: 'ÿ',            # safe Latin characters
    #   0: 'Ā',   1: 'ā',  2: 'Ă', ...    # dangerous bytes at the end, mapped to offset chars
    #   10: 'Ċ',  32: 'Ġ', ...             # newline, space are also here
    # }

    # Use the mapped single characters directly as the base vocabulary; discard raw bytes.
    vocab = {byte_to_unicode[i]: i for i in range(256)}
    # A dict of single-character string -> integer.

    vocab_inverse = {v: k for k, v in vocab.items()}
    # Inverse dict: integer -> single-character string.

    # Reconstruct merge rules.
    for (p0, p1), new_id in merges:
        s0 = vocab_inverse[p0]
        s1 = vocab_inverse[p1]

        # Simple string concatenation. Because each byte has been mapped to exactly one character,
        # merging only ever makes strings longer — there will never be strange bloat like <0xE4>!
        combined = s0 + s1

        vocab[combined] = new_id
        vocab_inverse[new_id] = combined

    # At this point vocab is already in a pure-text safe form;
    # no more try...except decoding is needed!
    readable_vocab = vocab

    # -------------------------------------------------------------------------
    # Register special tokens (original logic preserved)
    # -------------------------------------------------------------------------
    current_max_id = max(vocab.values())
    special_tokens = list(_tok_cfg["special_tokens"])  # Read from yaml
    num_reserved = _tok_cfg["num_reserved_tokens"]
    for i in range(num_reserved):
        special_tokens.append(f"<|reserved_{i}|>")

    for i, token in enumerate(special_tokens):
        new_special_id = current_max_id + 1 + i
        readable_vocab[token] = new_special_id

    print(f"Python: Added {len(special_tokens)} special tokens")

    # Save vocab JSON
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(readable_vocab, f, ensure_ascii=False, indent=2)

    print(f"✅ Vocab saved to: {SAVE_PATH}")
    print(f"   Final vocab size: {len(readable_vocab)}")

    # Save merges.txt
    with open(MERGES_PATH, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for (p0, p1), _ in merges:
            token0 = vocab_inverse[p0]
            token1 = vocab_inverse[p1]
            f.write(f"{token0} {token1}\n")

    print(f"✅ Merge rules saved to: {MERGES_PATH}")
