<p align="center">
  <h1 align="center">🥧 PIE — Handmaking your own Large Language Model from scratch</h1>
  <p align="center">
    <b>Why did I start this project? Because I was sick and tired of university textbooks that nobody can understand — walls of formulas, broken logic!</b>
  </p>
  <p align="center">
    On top of that, I noticed AI is evolving way too fast for static textbooks that get published once every few years.
    So I wanted to build a live-updating, discussion-driven textbook platform. That was the idea.
  </p>
  <p align="center">
    The biggest problem? A platform without content is dead on arrival. I thought about reaching out to other authors for permission to use their materials, but that was too much hassle. Besides, who am I to ask? So I decided to write my own textbook, anchored on the thing I'd always been interested in but never truly understood — the Transformer.
  </p>
    <p align="center">
    Starting January 30th, I've been posting videos for about two and a half months — roughly 50 episodes, ~8 minutes each — covering essentially all the math and the hands-on code for building a base model from scratch. To date: 10M+ views, 11K+ followers. Now I'm keeping my promise and open-sourcing the code.
  </p>
  <p align="center">
    <a href="README_zh.md">🇨🇳中文讲解</a>
    <a href="#what-is-this-project">Overview</a> ·
    <a href="#quick-start">Quick Start</a> ·
    <a href="#project-structure">Project Structure</a> ·
    <a href="#technical-highlights">Technical Highlights</a> ·
    <a href="#companion-tutorials">Tutorials</a> ·
    <a href="https://huggingface.co/Tianyu-Zhou/PIE1.0-0.2B-dense-base/tree/main">🤗 Model Download</a> ·
    <a href="https://modelscope.cn/models/Zaoshangzhou/PIE1.0-0.2B-dense-base/files">Model Download (ModelScope)</a> ·
  </p>
</p>

---

## What Is This Project?

PIE is a **fully from-scratch** ~0.2B-parameter(default, you can change it into whatever you like) language model pretraining framework — and my first textbook. It covers the entire pretraining pipeline end-to-end:

```
Corpus Sampling → BPE Tokenizer Training → Data Preprocessing → Model Definition → Multi-GPU Distributed Training → Streaming Inference
```



This is not a "just-import-it" project — no `from transformers import AutoModel`, no off-the-shelf tokenizer library, no black boxes of any kind. The mathematical principles behind every module are spelled out in the code comments, because this project is itself a product of the **Feynman Learning Method**: if you can't explain something clearly, you don't truly understand it.

**Model Specifications (PIE1.0-0.2B-dense-base) — these are just defaults, feel free to hack everything!:**

| Parameter | Value |
|---|---|
| Parameter Count | ~0.2B |
| Embedding Dimension (d_model) | 1024 |
| Transformer Layers | 16 |
| Attention Heads (Q Heads) | 16 |
| KV Heads (GQA) | 4 |
| Vocabulary Size | 32128 |
| Max Sequence Length | 1024 |
| Activation Function | SwiGLU |
| Positional Encoding | RoPE |
| Normalization | RMSNorm |
| Mixed Precision Training | BF16 |

**Comments are not decoration — they are the core asset of this project.**

Here's a snippet from the Rust BPE training loop:
```rust
// Core optimization: pop the highest-frequency valid pair from the heap.
// Frequencies stored in the heap may be "stale" — a previous merge round
// already changed a pair's real frequency, but the old entry hasn't been
// removed from the heap yet (lazy deletion).
// So every pop must be cross-checked against pair_freqs in real time;
// mismatches are simply discarded.
while let Some((freq, pair)) = heap.pop() {
    if let Some(&current_freq) = pair_freqs.get(&pair) {
        if current_freq == freq && freq > 1 {
            best_pair = Some((pair, freq));
            break;
        }
    }
}
```

And here's an explanation from the model script:
```python
  output = output.contiguous()
  # Step 2: Fix the memory layout.
  # Because transpose only changes the stride (the read-order logic),
  # the underlying data is still scattered. For example:
  # Before transpose: it tells the CPU "read one number, move 1 slot forward for the next."
  # After transpose: it tells the CPU "read one number, skip 3 slots to get the next."
  # contiguous() physically rearranges the data in memory so it's sequential again.
```

---

这是英文版：

---

## Quick Start

> **Goal:** Download the weights and run inference. For the full training pipeline, see the *Full Reproduction* section below.

### Step 1 — Create Environment and Clone Repository

```bash
conda create -n pie python=3.11 -y
conda activate pie

git clone https://github.com/Tianyu-Zhou1964/PIE-Handmaking_LLM.git
cd PIE-Handmaking_LLM
```

### Step 2 — Install Rust Toolchain

The BPE engine is implemented in Rust + PyO3 and requires local compilation. **Install Rust before installing Python dependencies.**

**Linux / macOS:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

**Windows (PowerShell):**
```powershell
winget install Rustlang.Rustup
# Or download rustup-init.exe from https://rustup.rs and run it
# Reopen a terminal after installation for environment variables to take effect
```

Verify:
```bash
rustc --version   # Success if a version number is printed
```

### Step 3 — Install Python Dependencies

**Important! Always install PyTorch manually — the default pip version is CPU-only.**

PyTorch must be installed with an explicit CUDA version. All other dependencies are handled by `requirements.txt`:

```bash
# Choose the line that matches your CUDA version (check: top-right of nvidia-smi output)
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8
# CPU-only inference: pip install torch

# Install remaining dependencies (maturin included — no separate install needed)
pip install -r requirements.txt
```

### Step 4 — Build the BPE Engine

```bash
cd Chinese/src/custom_bpe/
maturin develop --release
```

Verify the build:
```bash
cd Chinese/src/
python -c "import custom_bpe; print('BPE engine loaded successfully ✓')"
```

### Step 5 — Download Model Weights

**International users (HuggingFace):**
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Tianyu-Zhou/PIE1.0-0.2B-dense-base',
    filename='PIE-0.2B-dense.pth',
    local_dir='./Checkpoint'
)
"
```

**Users in China (ModelScope):**
```bash
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download(
    'Zaoshangzhou/PIE1.0-0.2B-dense-base',
    local_dir='./Checkpoint'
)
"
```

Confirm the weights are in place:
```bash
ls Checkpoint/
# You should see PIE-0.2B-dense.pth
```

### Step 6 — Run Inference

```bash
cd Chinese/src/
python inference.py
```

Supports CUDA / Apple MPS / CPU. `device="auto"` detects the backend automatically — no manual configuration needed.
Sampling parameters (temperature, top_k, top_p, repetition_penalty, etc.) can be tuned in `Chinese/config_zh.yaml`.

---

## Full Reproduction

### 0. Requirements

| Dependency | Minimum Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.11 recommended |
| PyTorch | 2.0+ | Requires `F.scaled_dot_product_attention` (FlashAttention backend) |
| CUDA | 11.8+ | Required for training; inference supports CPU or Apple MPS |
| Rust toolchain | stable | For compiling the Rust BPE engine — [one-line install](https://rustup.rs/) |
| maturin | 1.0+ | PyO3 build tool — compiles Rust into an importable Python module |

#### Dependency Notes

PIE's dependency list is intentionally minimal — just 7 packages, all "could implement ourselves but this is more convenient" utilities. No black-box libraries.

| Package | Version | Purpose |
|---|---|---|
| **torch** | ≥2.0 | Neural network framework: tensor ops, autograd, DDP distributed training |
| **numpy** | latest | Low-level numerical computation, data foundation for torch ops |
| **pandas** | latest | Tabular data processing (multi-source corpus indexing, quota management, etc.) |
| **pyarrow** | latest | Parquet file I/O — efficient read/write for large binary corpora |
| **pyyaml** | latest | Config file parsing (`config_zh.yaml`, `config_en.yaml`) |
| **tqdm** | latest | Progress bar visualization |
| **maturin** | ≥1.0 | Rust → Python build tool for the PyO3 BPE engine |

### 1. Install Python Dependencies

```bash
# Example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies (6 packages — not a 300-line conda environment dump)
pip install -r requirements.txt
# Includes: numpy, pandas, pyarrow, pyyaml, tqdm, maturin
```

### 2. Build the Rust BPE Engine

`custom_bpe` is the project's core tokenizer, implemented in Rust + PyO3. It is **50× faster** than a pure Python implementation. It is not on PyPI and must be compiled from source:

```bash
# Install Rust if you haven't already
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install maturin (PyO3 build tool)
pip install maturin

# Navigate to the Rust project directory for the target language and compile
# Chinese version:
cd Chinese/src/custom_bpe/
maturin develop --release
cd ../../..

# English version (same steps):
cd English/src/custom_bpe/
maturin develop --release
cd ../../..
```

Once compiled, `import custom_bpe` will be available in Python from any working directory.

### 3. Edit Configuration

All hyperparameters, file paths, and data mixture ratios are centralized in `config_zh.yaml` (or `config_en.yaml`) — **one file controls everything**. Before training, you **must** update the dataset paths to match your local machine:

```yaml
# Dataset path for tokenizer training
tokenizer:
  datasets:
    - name: "SkyPile"
      path: "/your/path/to/skypile"   # ← update this

# Path to the preprocessed binary corpus
training:
  data_path: "./Dataset/all.bin"      # ← update this

# Checkpoint and vocabulary paths for inference
inference:
  checkpoint_path: "../../Checkpoint/PIE-0.2B-dense.pth"
  vocab_path: "../../Tokenizer/tokenizer_32128/tokenizer_32128.json"
  merges_path: "../../Tokenizer/tokenizer_32128/merges.txt"
```

### 4. Full Pipeline

All commands below are run from the `Chinese/src/` directory:

```bash
cd Chinese/src/

# ── Step 1: Train the BPE Tokenizer ─────────────────────────────────────────
# Python side: multi-source corpus sampling (Chinese 40%, English 30%, Code 20%, Math 10%)
# Rust side: core BPE training loop (count pair frequencies → merge → update) — 50× faster than Python
# Output: Tokenizer/tokenizer_32128/tokenizer_32128.json and merges.txt
python tokenizer.py

# ── Step 2: Large-Scale Data Preprocessing ──────────────────────────────────
# 128 parallel processes for tokenization; cross-process shared counter for
# token-level quota control
# Output: Dataset/all.bin (~4 GB, memory-mapped directly during training)
python dataprocess.py \
  --data_root /path/to/your/datasets \
  --tokenizer_path ../../Tokenizer/tokenizer_32128/tokenizer_32128.json \
  --merges_path ../../Tokenizer/tokenizer_32128/merges.txt \
  --total_tokens 1000000000

# ── Step 3: Multi-GPU Distributed Training ──────────────────────────────────
# DDP (DistributedDataParallel) + NCCL backend; adjust nproc_per_node to your GPU count
# Supports BF16 mixed precision, torch.compile, warmup + cosine decay LR schedule,
# and checkpoint resumption
torchrun --nproc_per_node=2 train.py  # adjust GPU count as needed

# ── Step 4: Interactive Inference ───────────────────────────────────────────
# Streaming decode with Top-K + Top-P + temperature sampling + repetition penalty
# All parameters are configurable in config_zh.yaml
python inference.py
```

Pretrained weights are available on HuggingFace: [🤗 Tianyu-Zhou/PIE1.0-0.2B-dense-base](https://huggingface.co/Tianyu-Zhou/PIE1.0-0.2B-dense-base/tree/main). Download and place the file in the `Checkpoint/` directory.

---

## Project Structure

```
PIE0.1/
├── Chinese/                     # Chinese pretraining version
│   ├── config_zh.yaml           # Unified config file (all hyperparameters, paths, data ratios)
│   └── src/
│       ├── custom_bpe/          # Rust BPE engine (PyO3 compiled artifact)
│       │   └── src/
│       │       ├── engine.rs    # Unified entry point: encoder + decoder combined, single file I/O
│       │       ├── encode.rs    # High-performance inference encoder (Aho-Corasick + doubly-linked-list BPE)
│       │       ├── decode.rs    # Inference decoder (streaming UTF-8 + GPT-2 byte mapping restoration)
│       │       └── lib.rs       # PyO3 module registration
│       ├── tokenizer.py         # Tokenizer training: multi-source sampling → Rust BPE → vocabulary serialization
│       ├── dataprocess.py       # Data preprocessing: 128-process parallel tokenization → all.bin
│       ├── model.py             # Model definition: RoPE / GQA / SwiGLU / RMSNorm / MoE
│       ├── train.py             # Training script: DDP / BF16 / Cosine LR / checkpoint resumption
│       └── inference.py         # Inference script: streaming decoding / Top-K Top-P / repetition penalty
├── English/                     # English pretraining version (same structure as Chinese)
├── Checkpoint/
│   └── PIE-0.2B-dense.pth      # Pretrained weights (dense version, MoE disabled)
└── Tokenizer/
    └── tokenizer_32128/
        ├── tokenizer_32128.json # Vocabulary (32128 tokens, including 64 reserved special token slots)
        └── merges.txt           # BPE merge rules
```

---

## Technical Highlights

### Model Architecture — `model.py`

| Component | Implementation Details |
|---|---|
| **RoPE (Rotary Position Embedding)** | Rotation via complex multiplication; `precompute_rope_operators` pre-computes the frequency matrix for all positions — just slice during training, zero redundant computation |
| **GQA (Grouped Query Attention)** | KV heads = 1/4 of Q heads (Q=16, KV=4); reduces KV Cache VRAM by 4× during inference |
| **Flash Attention** | Uses PyTorch 2.0's native `F.scaled_dot_product_attention` — no third-party library dependencies |
| **SwiGLU Activation** | Same as LLaMA; FFN hidden dimension aligned to `multiple_of=256` to match optimal hardware GEMM tile sizes |
| **RMSNorm** | Pre-Norm architecture (normalize before entering the layer); more stable training than Post-Norm, eliminates mean-centering computation |
| **MoE (Optional)** | 4 experts with Top-2 routing; Sort & Batched sparse computation eliminates CPU-GPU sync bottlenecks; auxiliary loss (weight 0.01) prevents expert monopolization; disabled in the current release (`use_moe=False`) — dense performs better at the 0.2B scale |
| **Weight Tying** | Embedding and unembedding layers share weights (`output.weight = tok_embeddings.weight`), saving ~30% of total parameters |
| **KV Cache** | `setup_cache` pre-allocates zero tensors before inference begins; in-place writes during the generation loop completely avoid VRAM fragmentation from incremental tensor concatenation |

### BPE Tokenizer — `tokenizer.py` + Rust Engine

**Training Side (`tokenizer.py`):**

The Python side handles multi-source corpus sampling and vocabulary serialization; all core BPE training is performed in Rust. Uses GPT-2's standard Byte-to-Unicode safe mapping, losslessly mapping all 256 raw bytes to visible Unicode characters, completely eliminating control character contamination (`\x00`, `\n`, etc.) in JSON vocabulary files.

Multi-source training corpus ratios:

| Dataset | Language / Domain | Sampling Target (Characters) |
|---|---|---|
| SkyPile | Chinese general | 20 million |
| FineWeb | English general | 15 million |
| StarCoder (Python/Rust/Markdown) | Code (I only selected Python and Rust) | 10 million combined |
| NuminaMath | Math (problems + solutions) | 5 million |

The vocabulary includes 64 reserved special token slots covering dialogue turns (`<|user|>` / `<|assistant|>`), tool calls (`<|tool_call|>` / `<|tool_result|>`), chain-of-thought (`<|think|>` / `<|think_end|>`), and multimodal placeholders (`<|image|>` / `<|audio|>`), preserving interfaces for future SFT and multimodal extensions.

**Inference Side (Rust Engine):**

`engine.rs` is the unified engine combining encoder and decoder — only one file I/O operation at initialization. The entire `vocab.json` and `merges.txt` parsing is done entirely within Rust. Key optimizations:

- **FxHashMap**: Non-cryptographic ultra-fast hash map; `(u32, u32)` key lookups are 30%+ faster than the standard `HashMap`
- **Aho-Corasick Automaton**: O(N) linear-time scanning of all special tokens, completely replacing regex backtracking
- **Dual-Path BPE Merging**: Short words (< 50 bytes) use ranks array scanning, refreshing only the 1–2 affected positions; long words (consecutive spaces, extremely long URLs, and other degenerate text) use array-simulated doubly-linked list + BinaryHeap, reducing time complexity from O(N²) to O(N log N)
- **Zero-Allocation Buffers**: `word_buffer` and `list_buffer` are pre-allocated outside the loop; the hot path only calls `clear()` — zero heap allocations
- **GIL Release**: Calls `py.allow_threads()` during encoding/decoding, never blocking the Python main thread
- **Streaming UTF-8 Decoding**: Built-in byte buffer that automatically handles the half-character garbling that occurs when BPE splits a single CJK character (3-byte UTF-8) across multiple tokens

### Data Preprocessing — `dataprocess.py`

- **128-Process Parallelism**: Driven by Python `multiprocessing`, fully utilizing multi-core CPUs
- **Precise Quota Control**: Cross-process shared counters + mutex locks ensure each data source's token quota (SkyPile 60%, FineWeb 25%, StarCoder 10%, NuminaMath 5%) is exact down to every single token — no more, no less
- **Millisecond-Level Interruption**: Per-process buffer interception mechanism — the instant a data source's quota is exhausted, processing stops immediately, not one extra sample processed
- **Real-Time Progress Monitoring**: `tqdm` daemon thread provides multiple progress bars, giving you live visibility into each data source's processing progress

### Training — `train.py`

- **DDP Multi-GPU Training**: Launched via `torchrun` with NCCL backend; one independent process per GPU, `DistributedSampler` automatically shards the dataset, AllReduce synchronizes gradients
- **Hand-Written Warmup + Cosine Decay LR Schedule**: Deliberately not using `CosineAnnealingLR` — because under DDP multi-GPU, the number of batches per GPU changes, and hardcoding `T_max` would misalign the cosine curve with actual training progress. The hand-written `get_lr(step)` dynamically aligns to `T_MAX_STEPS` at runtime — no bugs when you change the config
- **BF16 Mixed Precision**: `torch.amp.autocast` — compute-intensive ops like matrix multiplication run in BF16, precision-sensitive ops like Softmax / Norm stay in FP32
- **`torch.compile` Acceleration**: `fullgraph=False` to accommodate MoE routing's dynamic branching — no forced static graph
- **Checkpoint Resumption**: Saves a checkpoint every `save_interval` steps (model weights + optimizer momentum state + epoch/step counters); automatically loads `model_latest.pth` on next launch to resume, rolling retention of the 3 most recent historical checkpoints
- **Ctrl+C Safe Save**: `KeyboardInterrupt` is caught — on interruption, immediately writes `model_interrupted.pth`, no progress lost
- **Process Group Safe Cleanup**: `dist.destroy_process_group()` is called in the `finally` block — whether or not an exception occurred, the NCCL communication matrix is properly released, preventing `Address already in use` errors

### Inference — `inference.py`

- **Autoregressive Generation**: Each forward pass takes the logits at the last position — Top-K → Top-P → temperature scaling → multinomial sampling — until `<eos>` is generated or `max_new_tokens` is reached
- **Repetition Penalty**: Looks back at the most recent `repetition_window` (default 100) tokens; subtracts a penalty from the logits of already-generated tokens, preventing the model from falling into "parrot" loops
- **Streaming Output**: Calls `engine.decode_stream(token_id)` for token-by-token decoding; the byte buffer automatically handles multi-byte UTF-8 characters being split across tokens, achieving true "generate-and-print" streaming
- **Multiple Checkpoint Format Support**: Automatically detects `model` / `model_state_dict` / raw weights — three different save formats
- **Apple MPS Support**: In `device="auto"` mode, detection order is CUDA → MPS → CPU — runs on M-series chips too

---

## Training Configuration Reference

Below are the training hyperparameters for the current release (PIE-0.2B-dense):

```yaml
training:
  batch_size: 4           # Per-GPU batch size (3× RTX 4090, effective batch = 12)
  seq_len: 256            # Training sequence length
  learning_rate: 3.0e-4  # Peak learning rate
  lr_min: 3.0e-5          # Cosine Decay minimum learning rate
  warmup_steps: 400       # Linear warmup steps
  epochs: 15              # Training epochs
  use_bf16: true          # BF16 mixed precision
  grad_clip_max_norm: 1.0 # Gradient clipping threshold (L2 norm)

dataprocess:
  total_tokens: 1_000_000_000   # Preprocessed corpus: default 1B tokens
  category_ratios:
    SkyPile:    0.60
    FineWeb:    0.25
    StarCoder:  0.10
    NuminaMath: 0.05
```

Training environment: NVIDIA RTX 4090 (24GB), ~7B tokens of corpus data, DDP + BF16.

---

## Companion Tutorials

This project is the code implementation of my Douyin (Chinese TikTok) video series. Best used together:
[Visit the author's Douyin page](https://v.douyin.com/XZWeuSl-IIY/)

| Series | Content Focus | Progress |
|---|---|---|
| **"Handtearing LLMs"** (手撕大模型) | Theory: from Attention to RoPE to MoE — each episode breaks down one core mechanism | 18 episodes, theory complete |
| **"Handmaking LLMs"** (手搓大模型) | Practice: from the Rust BPE tokenizer to full multi-GPU training of a base model — every line of code explained | 5 major episodes, practice complete |

Code comments extensively reference specific episodes in the format `Ep1`, `Ep9`, `Ep15`, etc. — while reading the code, you can jump directly to the corresponding theory explanation, creating a bidirectional "code ↔ theory" index.

> The personal website [passionie.uk](https://passionie.uk) hosts an interactive e-book version of the tutorials with hot-reloading and AI-assisted Q&A.

---

## FAQ

**Q: Why not use HuggingFace Transformers?**

Because importing packages is not learning. The goal of this project is to understand the mathematical principles behind every single component and be able to implement them yourself — if you can't explain something from start to finish, you don't truly understand it (Feynman said that).

**Q: The MoE code is still in there, but why is `use_moe=False` in the current version?**

At the 0.2B parameter scale, MoE's routing overhead (CPU-GPU synchronization, expert load balancing, etc.) cancels out the gains from sparse activation. Experimental results show the dense version achieves lower loss and more stable training. The complete MoE implementation is preserved in the code — it can be enabled directly when scaling up to larger models.

**Q: Why write the tokenizer in Rust instead of C++?**

Rust's memory safety guarantees (the borrow checker) completely eliminate segfaults and race conditions without sacrificing performance, and PyO3 provides a mature solution for FFI integration with Python. The cognitive overhead of writing C++ extensions is significantly higher. Also, Rust is just fun.

**Q: Why not use `tiktoken` or `sentencepiece`?**

Because I wanted to thoroughly understand the implementation details of BPE — Byte-to-Unicode mapping, reservoir sampling, merge rule serialization formats, and so on. Using an off-the-shelf library means you don't have to understand any of it. Only after writing it myself did I understand why `tiktoken` is fast and why `sentencepiece` is designed the way it is.

---


Contact: **tyro1964@gmail.com**
Or visit [Douyin DM](https://v.douyin.com/XZWeuSl-IIY/)

---

## License

This project is open-sourced under the [Apache-2.0 License](./LICENSE).

---

## Acknowledgments

Thanks to everyone who asked questions, caught errors, and offered encouragement in the Douyin comments — your questions are my best teachers.

---

<p align="center">
  <i>"You don't have to be great to start, but you have to start to become great."</i>
</p>
