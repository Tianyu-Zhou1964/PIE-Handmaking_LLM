<p align="center">
  <h1 align="center">🥧 PIE — 手搓大语言模型</h1>
  <p align="center">
    <b>为啥做这个项目？因为我当时恨透了大学不讲人话的教材，密密麻麻的公式，断裂的逻辑！</b>
  </p>
  <p align="center">
    此外，我观察到 AI 发展的太快了，静态、几年出版一次的老教材完全跟不上这个速度，
    就想做一个热更新的实时讨论的教材网站，这就是我的想法。
  </p>
  <p align="center">
    最大的问题是，一个平台没有内容根本活不了，我本来想联系别人让别人允许我用他们的教材，但是太麻烦了。而且，我算哪根葱？于是我就决定自己做一套教材，锚定了我一直感兴趣但其实没学明白的 Transformer，
  </p>
    <p align="center">
    然后从 1 月 30 日开始更新视频，至今过去了两个半月更新了 50 左右期视频，每期 8 分钟左右，基本上讲完了数学原理和手搓 base 模型代码实战的部分。迄今获得 100w 播放量，1.1w 粉丝。现在履行承诺，开源代码仓库。
  </p>
  <p align="center">
    <a href="#这个项目是什么">项目简介</a> ·
    <a href="#快速开始">快速开始</a> ·
    <a href="#项目结构">项目结构</a> ·
    <a href="#技术亮点">技术亮点</a> ·
    <a href="#配套教程">配套教程</a> ·
    <a href="https://huggingface.co/Tianyu-Zhou/PIE1.0-0.2B-dense-base/tree/main">🤗 模型下载</a> ·
    <a href="https://modelscope.cn/models/Zaoshangzhou/PIE1.0-0.2B-dense-base/files">模型下载（魔塔）</a>
  </p>
</p>

---

## 这个项目是什么？

PIE 是一个**完全从零实现**的 ~0.2B 参数中文语言模型预训练框架更是我的第一本教材，覆盖预训练全链路：

```
语料采样 → BPE 分词器训练 → 数据预处理 → 模型定义 → 多卡分布式训练 → 流式推理
```



它不是一个"调包侠"项目——没有 `from transformers import AutoModel`，没有现成的 Tokenizer 库，没有任何黑箱。每个模块的数学原理都在代码注释里写得清清楚楚，因为这个项目本身就是我践行**费曼学习法**的产物：如果你不能把一件事讲清楚，说明你还没真正理解它。

**模型规格（PIE1.0-0.2B-dense-base）这只是默认，你全都可以随便魔改！：**

| 参数 | 值 |
|---|---|
| 参数量 | ~0.2B |
| 词嵌入维度 (d_model) | 1024 |
| Transformer 层数 | 16 |
| 注意力头数 (Q 头) | 16 |
| KV 头数 (GQA) | 4 |
| 词表大小 | 32128 |
| 最大序列长度 | 1024 |
| 激活函数 | SwiGLU |
| 位置编码 | RoPE |
| 归一化 | RMSNorm |
| 混合精度训练 | BF16 |

**注释不是点缀，是这个项目的核心资产。**

这是 Rust BPE 训练循环里的一段：
```rust
// 核心优化：从堆中获取最高频且有效的 Pair
// 堆里存的频率可能是"过时"的——上一轮合并已经改变了某个 Pair 的真实频率，
// 但旧数据还没从堆里清掉（懒惰删除）。
// 所以每次弹出都要和 pair_freqs 实时校验，不一致就直接丢弃。
while let Some((freq, pair)) = heap.pop() {
    if let Some(&current_freq) = pair_freqs.get(&pair) {
        if current_freq == freq && freq > 1 {
            best_pair = Some((pair, freq));
            break;
        }
    }
}
```

这是 model 脚本中的一段解释
```python
  output = output.contiguous()
  # 第二步：整理内存布局
  # 因为 transpose 只是改变了读取逻辑的步长，数据是乱序的，例如：
  # 转置前：它告诉 CPU，"读完一个数，往后走 1 位读下一个"。
  # 转置后：它告诉 CPU，"读完一个数，先别管隔壁，跳过 3 位去读下一个"。
  # contiguous() 会把数据在内存里重新按顺序排列
```

---

## 快速开始

### 0. 环境要求

| 依赖 | 最低版本 | 说明 |
|---|---|---|
| Python | 3.10+ | 推荐 3.11 |
| PyTorch | 2.0+ | 需要 `F.scaled_dot_product_attention`（FlashAttention 后端） |
| CUDA | 11.8+ | 训练必须；推理可用 CPU 或 Apple MPS |
| Rust 工具链 | stable | 用于编译 Rust BPE 引擎，[一键安装](https://rustup.rs/) |
| maturin | 1.0+ | PyO3 的构建工具，将 Rust 编译为可 `import` 的 Python 模块 |
#### 依赖说明

PIE 的依赖表非常精简——只有 7 个包，都是"能自己实现但用它更方便"的工具，不包含任何黑箱库。

| 包名 | 版本 | 用途 |
|---|---|---|
| **torch** | ≥2.0 | 神经网络框架：张量计算、自动微分、DDP 分布式训练 |
| **numpy** | latest | 底层数值计算，torch 操作的数据基础 |
| **pandas** | latest | 数据表处理（多源语料索引、配额管理等） |
| **pyarrow** | latest | Parquet 文件 I/O，大规模二进制语料的高效读写 |
| **pyyaml** | latest | 配置文件解析（`config_zh.yaml`、`config_en.yaml`） |
| **tqdm** | latest | 进度条可视化 |
| **maturin** | ≥1.0 | Rust → Python 编译工具，用于 PyO3 BPE 引擎 |
### 1. 安装 Python 依赖

```bash
# 例如 CUDA 12.1：
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 安装其余依赖（只有 6 个包，不是从 conda 环境 dump 的几百行垃圾）
pip install -r requirements.txt
# 包含：torch, numpy, pandas, pyarrow, pyyaml, tqdm
```

### 2. 编译 Rust BPE 引擎

`custom_bpe` 是本项目的核心分词器，用 Rust + PyO3 实现，性能比纯 Python 实现快 50 倍以上。它不在 PyPI 上，需要从源码编译：

```bash
# 安装 Rust（如果还没有的话）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 安装 maturin（PyO3 的构建工具）
pip install maturin

# 进入对应语言的 Rust 项目目录，编译并安装到当前 Python 环境
# 中文版：
cd Chinese/src/custom_bpe/
maturin develop --release
cd ../../..

# 英文版同理：
cd English/src/custom_bpe/
maturin develop --release
cd ../../..
```

编译成功后，在对应目录下执行 `import custom_bpe` 即可在 Python 中调用 Rust 引擎。

### 3. 修改配置

所有超参数、路径、数据配比都集中在 `config_zh.yaml`（或 `config_en.yaml`）中，**改一处即改全局**。训练前**必须**把配置文件中的数据集路径改成你自己机器上的实际路径：

```yaml
# 分词器训练的数据集路径
tokenizer:
  datasets:
    - name: "SkyPile"
      path: "/your/path/to/skypile"   # ← 改这里

# 预处理后的二进制语料路径
training:
  data_path: "./Dataset/all.bin"      # ← 改这里

# 推理时的权重和词表路径
inference:
  checkpoint_path: "../../Checkpoint/PIE-0.2B-dense.pth"
  vocab_path: "../../Tokenizer/tokenizer_32128/tokenizer_32128.json"
  merges_path: "../../Tokenizer/tokenizer_32128/merges.txt"
```

### 4. 全流程复现

以下命令均在 `Chinese/src/` 目录下执行：

```bash
cd Chinese/src/

# ── 第一步：训练 BPE 分词器 ──────────────────────────────────────────────────
# Python 端多源语料采样（中文 40%、英文 30%、代码 20%、数学 10%）
# Rust 端执行核心 BPE 训练循环（统计 pair 频率 → 合并 → 更新），比纯 Python 快 50 倍
# 产物：Tokenizer/tokenizer_32128/tokenizer_32128.json 和 merges.txt
python tokenizer.py

# ── 第二步：大规模数据预处理 ──────────────────────────────────────────────────
# 128 进程并行 tokenize，跨进程共享计数器实现精确配额控制（精确到 token 级别）
# 产物：Dataset/all.bin（约 4GB，供训练直接 mmap 读取）
python dataprocess.py \
  --data_root /path/to/your/datasets \
  --tokenizer_path ../../Tokenizer/tokenizer_32128/tokenizer_32128.json \
  --merges_path ../../Tokenizer/tokenizer_32128/merges.txt \
  --total_tokens 1000000000

# ── 第三步：多卡分布式训练 ──────────────────────────────────────────────────
# DDP（DistributedDataParallel）+ NCCL 后端，按 GPU 数量调整 nproc_per_node
# 支持 BF16 混合精度、torch.compile 加速、Warmup + Cosine Decay 调度、断点续训
torchrun --nproc_per_node=2 train.py # 启用多少张卡可以自己修改

# ── 第四步：交互式推理 ──────────────────────────────────────────────────────
# 流式解码，Top-K + Top-P + 温度采样 + 重复惩罚，所有参数可在 config_zh.yaml 中调节
python inference.py

```
#### PS：如果你还没下载，请先去下载我的模型文件 
预训练权重已上传至 HuggingFace：[🤗 Tianyu-Zhou/PIE1.0-0.2B-dense-base](https://huggingface.co/Tianyu-Zhou/PIE1.0-0.2B-dense-base/tree/main)，下载后放到 `Checkpoint/` 目录即可。

---

## 项目结构

```
PIE0.1/
├── Chinese/                     # 中文预训练版本
│   ├── config_zh.yaml           # 统一配置文件（所有超参数、路径、数据配比）
│   └── src/
│       ├── custom_bpe/          # Rust BPE 引擎（PyO3 编译产物）
│       │   └── src/
│       │       ├── engine.rs    # 统一入口：编码 + 解码合体，共享一次文件 I/O
│       │       ├── encode.rs    # 推理端高性能编码器（Aho-Corasick + 双向链表 BPE）
│       │       ├── decode.rs    # 推理端解码器（流式 UTF-8 + GPT-2 字节映射还原）
│       │       └── lib.rs       # PyO3 模块注册
│       ├── tokenizer.py         # 分词器训练：多源采样 → Rust BPE → 词表序列化
│       ├── dataprocess.py       # 数据预处理：128 进程并行 tokenize → all.bin
│       ├── model.py             # 模型定义：RoPE / GQA / SwiGLU / RMSNorm / MoE
│       ├── train.py             # 训练脚本：DDP / BF16 / Cosine LR / 断点续训
│       └── inference.py         # 推理脚本：流式解码 / Top-K Top-P / 重复惩罚
├── English/                     # 英文预训练版本（结构同 Chinese）
├── Checkpoint/
│   └── PIE-0.2B-dense.pth      # 预训练权重（dense 版本，MoE 已禁用）
└── Tokenizer/
    └── tokenizer_32128/
        ├── tokenizer_32128.json # 词表（32128 个 token，含 64 个特殊 token 预留位）
        └── merges.txt           # BPE 合并规则
```

---

## 技术亮点

### 模型架构 — `model.py`

| 组件 | 实现细节 |
|---|---|
| **RoPE 旋转位置编码** | 复数乘法实现旋转，`precompute_rope_operators` 预计算全部位置的频率矩阵，训练时直接切片，零重复计算 |
| **GQA（分组查询注意力）** | KV 头数为 Q 头数的 1/4（Q=16 头，KV=4 头），推理时 KV Cache 显存需求缩减 4 倍 |
| **Flash Attention** | 使用 PyTorch 2.0 原生 `F.scaled_dot_product_attention`，无第三方库依赖 |
| **SwiGLU 激活函数** | LLaMA 同款，FFN 隐藏层维度对齐到 `multiple_of=256`，适配硬件 GEMM 最优 Tile 大小 |
| **RMSNorm** | Pre-Norm 结构（先归一化再进层），相比 Post-Norm 训练更稳定，省去了均值中心化计算 |
| **MoE（可选）** | 4 专家 Top-2 路由，Sort & Batched 稀疏计算消除 CPU-GPU 同步瓶颈；辅助损失（权重 0.01）防止专家垄断；当前发布版本已禁用（`use_moe=False`），在 0.2B 规模下 dense 效果更优 |
| **权重共享** | Embedding 层与 Unembedding 层权重绑定（`output.weight = tok_embeddings.weight`），节省约 30% 参数量 |
| **KV Cache** | `setup_cache` 在推理开始前预分配全零张量，生成循环中原地写入，彻底避免逐步拼接张量的显存碎片化 |

### BPE 分词器 — `tokenizer.py` + Rust 引擎

**训练端（`tokenizer.py`）：**

Python 端负责多源语料采样与词表序列化，核心 BPE 训练全部由 Rust 完成。采用 GPT-2 标准的 Byte-to-Unicode 安全映射，将 256 个原始字节无损地映射到可见 Unicode 字符，彻底杜绝控制字符（`\x00`、`\n` 等）在 JSON 词表中造成污染。

训练语料多源配比：

| 数据集 | 语言/领域 | 采样字符目标 |
|---|---|---|
| SkyPile | 中文通用 | 2000 万 |
| FineWeb | 英文通用 | 1500 万 |
| StarCoder (Python/Rust/Markdown) | 代码（我只选了 Python 和 Rust） | 合计 1000 万 |
| NuminaMath | 数学（题目+解答） | 500 万 |

词表包含 64 个特殊 token 预留位，覆盖对话轮次（`<|user|>` / `<|assistant|>`）、工具调用（`<|tool_call|>` / `<|tool_result|>`）、思维链（`<|think|>` / `<|think_end|>`）、多模态占位符（`<|image|>` / `<|audio|>`），为后续 SFT 和多模态扩展保留接口。

**推理端（Rust 引擎）：**

`engine.rs` 是编码与解码合体的统一引擎，初始化时只需一次文件 I/O，整个 `vocab.json` 和 `merges.txt` 解析全部在 Rust 内完成。核心优化：

- **FxHashMap**：非加密极速哈希表，`(u32, u32)` 键的查找比标准 `HashMap` 快 30%+
- **Aho-Corasick 自动机**：O(N) 线性时间扫描所有特殊 token，彻底替代正则回溯
- **双路径 BPE 合并**：短词（< 50 字节）走 ranks 数组扫描，只刷新受影响的 1~2 个位置；长词（连续空格、超长 URL 等畸形文本）走数组模拟双向链表 + BinaryHeap，时间复杂度从 O(N²) 压制到 O(N log N)
- **零分配 Buffer**：`word_buffer` 和 `list_buffer` 在循环外预分配，热路径中只 `clear()`，零堆内存分配
- **GIL 释放**：编码/解码期间调用 `py.allow_threads()`，不阻塞 Python 主线程
- **流式 UTF-8 解码**：内置字节缓冲区，自动处理 BPE 将一个汉字（3 字节 UTF-8）切分到多个 token 时的半字乱码问题

### 数据预处理 — `dataprocess.py`

- **128 进程并行**：多进程 `multiprocessing` 驱动，充分利用多核 CPU
- **精确配额控制**：跨进程共享计数器 + 互斥锁，各数据源（SkyPile 60%、FineWeb 25%、StarCoder 10%、NuminaMath 5%）的 token 配额精确到每一个 token，不多不少
- **毫秒级中断**：单进程内 Buffer 拦截机制，某数据源配额耗尽时立即停止，不多处理一条数据
- **实时进度监控**：`tqdm` 守护线程提供多进度条，随时掌握每个数据源的处理进度

### 训练 — `train.py`

- **DDP 多卡训练**：`torchrun` 启动，NCCL 后端，每张卡一个独立进程，`DistributedSampler` 自动切分数据集，AllReduce 同步梯度
- **手写 Warmup + Cosine Decay 学习率调度**：没有用 `CosineAnnealingLR`，原因是 DDP 多卡时每张卡分到的 batch 数量会变，`T_max` 如果写死就会导致余弦曲线对不齐实际训练进度；手写 `get_lr(step)` 在运行时动态对齐 `T_MAX_STEPS`，配置改了不会出 bug
- **BF16 混合精度**：`torch.amp.autocast`，矩阵乘法等计算密集算子走 BF16，Softmax / Norm 等精度敏感算子保持 FP32
- **`torch.compile` 加速**：`fullgraph=False`，兼容 MoE 路由的动态分支，不强制静态图
- **断点续训**：每隔 `save_interval` 步保存 checkpoint（模型权重 + 优化器动量状态 + epoch/step 计数），下次启动自动加载 `model_latest.pth` 恢复，滚动保留最近 3 个历史 checkpoint
- **Ctrl+C 安全存档**：`KeyboardInterrupt` 捕获，中断时立即写入 `model_interrupted.pth`，不丢失进度
- **进程组安全释放**：`finally` 块中调用 `dist.destroy_process_group()`，无论是否异常退出，NCCL 通信矩阵都会被正确释放，防止 `Address already in use`

### 推理 — `inference.py`

- **自回归生成**：每步前向传播取最后一个位置的 logits，Top-K → Top-P → 温度缩放 → Multinomial 采样，直到生成 `<eos>` 或达到 `max_new_tokens`
- **重复惩罚**：回看最近 `repetition_window`（默认 100）个 token，对已出现的 token logits 减去惩罚值，防止模型陷入"复读机"循环
- **流式输出**：调用 `engine.decode_stream(token_id)` 逐 token 解码，字节缓冲区自动处理多字节 UTF-8 字符的跨 token 截断，做到真正的"边生成边打印"
- **兼容多种 checkpoint 格式**：自动检测 `model` / `model_state_dict` / 纯权重三种保存格式
- **Apple MPS 支持**：`device="auto"` 模式下，依次检测 CUDA → MPS → CPU，M 系列芯片也能跑

---

## 训练配置参考

以下为当前发布版本（PIE-0.2B-dense）的训练超参数：

```yaml
training:
  batch_size: 4           # 单卡 batch size（3× RTX 4090，有效 batch = 12）
  seq_len: 256            # 训练序列长度
  learning_rate: 3.0e-4  # 峰值学习率
  lr_min: 3.0e-5          # Cosine Decay 最低学习率
  warmup_steps: 400       # 线性 Warmup 步数
  epochs: 15              # 训练轮数
  use_bf16: true          # BF16 混合精度
  grad_clip_max_norm: 1.0 # 梯度裁剪阈值（L2 范数）

dataprocess:
  total_tokens: 1_000_000_000   # 预处理语料：默认 1B tokens
  category_ratios:
    SkyPile:    0.60
    FineWeb:    0.25
    StarCoder:  0.10
    NuminaMath: 0.05
```

训练环境：NVIDIA RTX 4090（24GB），约 7B tokens 语料，DDP + BF16。

---

## 配套教程

这个项目是我抖音系列课程的代码实现，两个系列配合使用效果最佳：
[点击前往作者抖音主页](https://v.douyin.com/XZWeuSl-IIY/)

| 系列 | 内容定位 | 进度 |
|---|---|---|
| **「手撕大模型」** | 理论篇：从 Attention 到 RoPE 到 MoE，每集讲透一个核心机制 | 18 集，理论完结 |
| **「手搓大模型」** | 实战篇：从 Rust BPE 分词器到完整多卡训练出 base 模型，代码一行一行讲清楚| 5 大集，实战完结 |

代码中大量注释以 `Ep1`、`Ep9`、`Ep15` 等形式**直接引用对应集数**，看代码时可以快速跳转到对应的理论讲解，形成"代码 ↔ 原理"的双向索引。

> 个人网站 [passionie.uk](https://passionie.uk) 上有交互式电子书版本的教程，支持热重载和 AI 辅助答疑。

---

## 常见问题

**Q: 为什么没有用 HuggingFace Transformers？**

因为调包不是学习。这个项目的目标是把每一个组件的数学原理都搞清楚、都能自己实现——如果你不能把一件事从头到尾讲清楚，说明你还没真正理解它（费曼说的）。

**Q: MoE 代码保留了，但为什么当前版本 `use_moe=False`？**

在 0.2B 这个参数规模下，MoE 的路由开销（CPU-GPU 同步、专家负载均衡等）会抵消掉稀疏激活带来的收益，实验结果是 dense 版本 loss 更低、训练更稳定。MoE 的完整实现保留在代码里，Scale Up 到更大模型时可以直接启用。

**Q: 为什么 Rust 写分词器，而不是 C++？**

Rust 的内存安全保证（借用检查器）可以在不牺牲性能的前提下彻底排除段错误和竞态条件，和 Python 的 FFI 对接也有 PyO3 这个成熟方案。写 C++ 扩展的心智负担要高得多。而且，Rust 很好玩。

**Q: 为什么不用 `tiktoken` 或 `sentencepiece`？**

因为想彻底搞懂 BPE 的实现细节——Byte-to-Unicode 映射、水塘抽样、合并规则的序列化格式等。用现成的库就什么都不用理解了。自己写一遍之后才明白 `tiktoken` 为什么快、`sentencepiece` 为什么设计那样。

---


联系方式 **tyro1964@gmail.com**
或者前往 [抖音私信](https://v.douyin.com/XZWeuSl-IIY/)

---

## 开源协议

本项目基于 [Apache-2.0 License](./LICENSE) 开源。

---

## 致谢

感谢所有在抖音评论区提问、纠错、鼓励的朋友们——你们的问题是我最好的老师。

---

<p align="center">
  <i>「你不需要很厉害才能开始，但你需要开始才能变得很厉害。」</i>
</p>
