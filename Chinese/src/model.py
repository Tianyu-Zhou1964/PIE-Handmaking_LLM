# ==============================================================================
# 基础依赖库：对应 Ep1 (深度学习框架) & Ep2 (张量运算)
# ==============================================================================
import torch                      # 核心张量运算库 (Ep2: 算力的基石，矩阵乘法的舞台)
import torch.nn as nn             # 神经网络组件库 (Ep6: 线性层、嵌入层等积木块)
import torch.nn.functional as F    # 函数式接口 (Ep7/9: Softmax、激活函数等无参数运算)

# ==============================================================================
# 工具类与类型注解：提升代码健壮性 (生产环境规范)
# ==============================================================================
from dataclasses import dataclass # 自动生成配置类的样板代码 (Ep4: 优雅的超参数封装)
from typing import Optional, Tuple # 静态类型检查 (Ep13: 明确输入输出，减少低级 Bug)
import yaml  # 用于从 config_zh.yaml 加载配置

# ==============================================================================
# 配置类：对应 Ep4 & Ep8 (定义维度、头数等超参数)
# ==============================================================================
@dataclass # 一个为了简化变量初始化值的便捷工具
class ModelArgs:
    dim: int = 1024              # 词嵌入维度 (Ep3: Embedding向量的长度 / Ep6: d_model)
    n_layers: int = 16           # Transformer Block 的层数 (Ep9: 层层嵌套)
    n_heads: int = 16            # 多头注意力的头数 (Ep8: 专家分工)
    vocab_size: int = 32128    # 词表大小 (Ep3: 赛博字典的大小)
    multiple_of: int = 256      # FFN 隐藏层维度的倍数 (Ep9: 膨胀层维度的对齐)
    norm_eps: float = 1e-5      # RMSNorm 防止分母为0的极小值 (Ep9/18: epsilon)
    max_seq_len: int = 1024      # 最大序列长度 (Ep2: 上下文窗口大小)
    n_experts: int = 4          # 总专家数量
    n_experts_per_tok: int = 2  # 每个 Token 启发 2 个专家 (Top-2 路由)
    use_moe: bool = False

    @classmethod
    def get_args(cls, preset: str = "tiny", **kwargs):
        """获取预设配置，确保训练和推理的一致性"""
        if preset == "tiny":
            # 这里的关键是：显式地把类变量里的 vocab_size 塞进去
            return cls(dim=1024, n_layers=16, n_heads=16, max_seq_len=1024, **kwargs)
        return cls(**kwargs)
    
    @classmethod
    def from_yaml(cls, config_path: str = "config_zh.yaml", **overrides):
        """从 YAML 配置文件的 model 段构建 ModelArgs，overrides 可覆盖任意字段"""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["model"]
        cfg.update(overrides)
        # 只取 dataclass 中定义过的字段，忽略 yaml 中多余的键 (如 rope_base)
        import dataclasses
        valid_keys = {field.name for field in dataclasses.fields(cls)}
        filtered = {k: v for k, v in cfg.items() if k in valid_keys}
        return cls(**filtered)

# ==============================================================================
# RMSNorm：对应 Ep9 (层归一化) & Ep18 (2.1节 RMS归一化)
# ==============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Ep13: nn.Parameter 代表这是可以被梯度下降更新的参数 (gamma)
        # 对应 Ep9 第一步：gamma * x_new + beta (RMSNorm通常省略beta)
        self.weight = nn.Parameter(torch.ones(dim)) 
        # 比如刚才定义了 dim 是 512，这行代码生成一个 1 行*512 列的行向量，初始化为 1

    def _norm(self, x): 
    # x 是词矩阵，是三维张量，形状(Batch, Seq, Dim)，这个 batch 是批次
    # batch是为了GPU并行计算服务的，比如4090有16384个CUDA核心，batch如果等于一的话相当于16384个大厨炒一粒米
    # 效率极低，所以我们应该选取合适的 batch 提高显存利用率。研究模型算法我们重点关注后两维，Seq 和 Dim
    # Seq 是 token 数量，比如 10 个字 Seq=10，是词矩阵的行数；Dim 是维度数，表示一个词包含多少词义，这里 512

        # 按照教材 Ep18 公式：x / sqrt(mean(x^2) + eps)
        # rsqrt 是 1/sqrt 的倒数平方根函数
        # x.pow(2).mean(-1) 计算 x^2 的均值 (方差的简化版，RMSProp思想)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # 这个-1的意思就是对(Batch, Seq, Dim)的最后一个做文章，也就是对 Dim 取平方和均值
        # keepdim必须有，否则维度数 Dim 就被 mean 操作压扁成一维了。

    def forward(self, x): # 这个方法是归一化+再缩放 RMS_Norm的标准操作，属于从输入走向输出，
        # 所以算是一个前向传播过程，nn.Module强制要求它必须叫做 forward
        # 这里的 x 尺寸还是(Batch, Seq, Dim)
        # Ep13: 前向传播计算图构建
        # 转换 float 保证精度，计算完归一化后再乘上可学习的缩放参数 gamma (self.weight)
        output = self._norm(x.float()).type_as(x)
        #float转成浮点数保障精度，.type_as(x)是保持输入和输出数据类型相同，比如都是float16/float32
        return output * self.weight
        # 归一化后的再缩放操作，这个乘法是逐元素相乘
        # self.weight会从 1*512 向量转换成(Batch, Seq, Dim)尺寸

# ==============================================================================
# RoPE 预计算：对应 Ep15 (旋转位置编码)
# ==============================================================================
def precompute_rope_operators(dim: int, seq_len: int, base: float = 10000.0):
    """预计算一个旋转算子矩阵"""
    # 教材 Ep15 第三部分：计算 theta_i = 10000^(-2i/d)
    # dim 这里传入的是 head_dim (Ep8: 每个头的维度)
    # 所谓头就是矩阵的几列合称，比如我们的总维度数 d 是 512，头 heads 数是 8，这里 head_dim 就是512/8=64
    # torch.arange(0, dim, 2) 对应 Ep15 中的 i (取偶数位)，从 0 到 64(左闭右开)，步长为 2 构建一个
    # 一维张量（也是向量）tensor([0,2,4,...,62])
    # [: (dim // 2)]意思是从 0 取到dim//2也就是32，32 依旧不到(左闭右开)
    theta_i = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 构建旋转频率相当于 theta_i=1/(base^(2i/d)),base一般取 10000
    # 也就是theta_i = 10000^(-2i/d)
    
    # 生成序列位置索引 m (0, 1, 2, ..., seq_len-1)
    # 对应 Ep15 中的 m （表示词矩阵第几行）
    m = torch.arange(seq_len, device=theta_i.device)
    # device=theta_i.device用来保证 m 和 theta_i放在相同设备，如显卡
    # 计算 m * theta_i ，得到所有位置在所有频率下的旋转角度
    # 对应 Ep15：m * theta
    theta_i = torch.outer(m, theta_i).float()  # (seq_len, dim/2)
    # outer(m, theta_i)的意思是 m 的第一项乘到θi 的每一项，然后 m 的第二项乘到θi，直到 seq_len最后一项
    # 将角度转化为复数形式: cos(m*theta) + i*sin(m*theta)
    # 对应 Ep17：欧拉公式 e^(ix) = cosx + isinx
    # torch.polar(模长, 角度) -> 模长为1的复数向量
    theta_i_cis = torch.polar(torch.ones_like(theta_i), theta_i)  # complex64
    # cis 表示：cos + i sin。.polar就是极坐标的意思，用模长torch.ones_like(theta_i)
    # 和角度theta_i定义一个复数，所以theta_i_cis就是一个旋转算子矩阵
    # 所有元素模长均为1所以作用后长度不变只改变角度，尺寸是m*32
    return theta_i_cis

# ==============================================================================
# 应用 RoPE：对应 Ep15 (第四部分：复数乘法实现旋转)
# ==============================================================================
def apply_rope(x: torch.Tensor, theta_i_cis: torch.Tensor):
    # 对应 Ep15 "分组策略（两两配对）"
    # 现在这个x 不再是三维张量，形状(Batch, Seq, 512)，而是(Batch, Seq, 8, 64)，8 是头数
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # x.shape[:-1]表示取出除了最后一维64 以外的所有值，*表示解包，把元组变成数字
    # 比如x尺寸是(2,512,8,64),*x.shape[:-1]就是2,512,8
    # reshape里第四个参数-1表示自动计算524,288 / (2 * 512 * 8 * 2) = 32
    # 所以reshape(2, 512, 8, -1, 2)就是reshape(2, 512, 8, 32, 2)
    # 所以x_complex变成了 2批次，512 行，8 个头，32 个房间，每个房间两个元素的矩阵
    # 最外面.view_as_complex是把最内层两个元素看成一个复数的实部和虚部
    theta_i_cis = theta_i_cis.view(1, x_complex.shape[1], 1, x_complex.shape[-1])
    # 这个操作.view是为了对齐尺寸，theta_i_cis只有两阶，x_complex有四阶，
    # 我们要给 theta_i_cis 编造虚构的两个阶： batch 和 heads，全部取 1 即可
    # theta_i_cis成为 4 阶张量是(1, 512, 1, 32)
    rotated_complex = x_complex * theta_i_cis
    # x_complex 是(2, 512, 8, 32)，theta_i_cis 是(1, 512, 1, 32)
    # 两个复数矩阵直接相乘，利用复数乘法法则完成旋转
    x_real_grouped = torch.view_as_real(rotated_complex)
     # 形状从 (B, S, H, 32) 变成 (B, S, H, 32, 2)
    # 最后一个维度 2 分别代表 [实部, 虚部]
    x_out = x_real_grouped.flatten(3)
    # 形状从 (B, S, H, 32, 2) 变成 (B, S, H, 64)
    # flatten(3) 表示从第四阶（即 32 那一维）开始往后全部压扁
    return x_out.type_as(x)
    # .type_as前面出现过，还是保障精度相同，为float16

# ==============================================================================
# 注意力机制：对应 Ep6 (Q,K,V), Ep7 (Attention), Ep8 (多头)
# ==============================================================================
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        # Ep8: 每个头的维度 = 总维度 / 头数
        self.head_dim = args.dim // args.n_heads

        # GQA: K 和 V 的头数是 Q 的 1/4，推理时 KV Cache 显存占用减少 4×
        # 比如 Q 用 8 头，K/V 只用 2 头，forward 里会把 K/V repeat 到和 Q 一样的头数再做注意力
        self.n_kv_heads = args.n_heads // 4

        # Ep6: 定义线性变换矩阵 Wq, Wk, Wv
        # 对应公式 Q = X * Wq, K = X * Wk, V = X * Wv
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        # GQA: Wk 和 Wv 的输出维度只有 n_kv_heads * head_dim，比 Wq 小 4 倍
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # nn.Linear创建权重参数矩阵 W 尺寸args.dim*args.dim这里是 512*512，没有偏置
        # 并且每个元素值都被初始化为均值为 0，方差为 1 的独立随机变量，之后用于训练
        # 注意nn.Linear创建的实例它内部实现了__call__函数，它可以作为函数被调用
        # 这个特性我们马上就要用到

        # Ep8: 输出变换矩阵 Wo (用于多头融合 Fusion)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)
        # 跟刚才一模一样，也可以被训练
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: torch.Tensor, theta_i_cis: torch.Tensor, mask: Optional[torch.Tensor], start_pos: int = 0):
        bsz, seqlen, _ = x.shape
        # bsz 是 batchsize ，seqlen 是行数，_是维度数这里不用
        
        # Ep6: 计算 Q, K, V (线性投影)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 相当于xq=x*self.wq,x:2*512*512,wq:512*512进行矩阵乘法之后,xq还是2*512*512

        # Ep8: 重塑为多头形式 (batch, seqlen, n_heads, head_dim)
        # 将大的向量切分为 n_heads 份
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        # GQA: K 和 V 只切成 n_kv_heads 份，比 Q 少
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        # 其实就是把原来的dim=512分成两个维度8*64，8个头，每个头分走矩阵512列中的64列

        # Ep15: 应用 RoPE 旋转位置编码
        # 注意：只对 Q 和 K 进行旋转，V 不旋转 (Ep15 第三部分)
        xq = apply_rope(xq, theta_i_cis)
        xk = apply_rope(xk, theta_i_cis)
        # 调用刚才的旋转操作，此时xq和xk已经完全注入位置信息

        if self.cache_k is not None and self.cache_v is not None:
            # 1. 把当前新计算的 K, V 填入缓存的对应位置
            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
            
            # 2. 取出从位置 0 到当前位置的所有 K, V，参与后续的注意力计算
            # 这样一来，不管你传进来几个词，我们都能拿到完整的上下文！
            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            # 如果没有分配 Cache (说明处于训练阶段)，直接用当前的
            keys = xk
            values = xv

        # 调整维度以便进行矩阵乘法 (batch, n_heads, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)     # 👇 修改：使用取出的 keys
        values = values.transpose(1, 2) # 👇 修改：使用取出的 values

        # 这里我们以 xk 为例好了xk不转置是(2, 512, 8, 64)，.transpose(1,2)是把第二阶和第三阶调换顺序
        # 相当于把 xk 转置变成了：(2, 8, 512, 64)，便于接下来做矩阵乘法
        # 这一步结束后的xq，xk，xv其实是对头进行分配，batch=2 并行计算两个，head=8 共 8 个头
        # 每个头都有 512 行词矩阵，负责 64 个词义

        # GQA: 把 K 和 V 的头数 repeat 到和 Q 一样，使得矩阵乘法维度对齐
        # repeat_interleave(4, dim=1) 把每个 kv_head 原地重复 4 次
        # 比如 K 原来是 (2, 2, 512, 64)，repeat 后变成 (2, 8, 512, 64)，和 Q 对齐
        # 这只是逻辑上的广播复制，参数量本身没有增加
        n_rep = self.n_heads // self.n_kv_heads  # = 4
        keys = keys.repeat_interleave(n_rep, dim=1)     # 👇 修改
        values = values.repeat_interleave(n_rep, dim=1) # 👇 修改

        # Ep7: Flash Attention 替换手写的 matmul + softmax + matmul
        # F.scaled_dot_product_attention 是 PyTorch 2.0 内置的 Flash Attention 实现
        # 核心优势：不把完整的 (B, H, S, S) score 矩阵写入显存，而是分块计算，显存省 30~50%，速度快 20~30%
        # is_causal=True 自动处理因果掩码，内部做法和我们的 mask 一样，但不需要显式传入 -inf 矩阵
        # / (self.head_dim ** 0.5) 对应 Ep7 "缩放(Scaling)"：除以 sqrt(d_k)，Flash Attention 内部自动完成
        # 结果形状：(bsz, n_heads, seqlen, head_dim)，和原来 torch.matmul(scores, xv) 完全一致

        ## scores = torch.matmul(xq, xk.transpose(2, 3)) / (self.head_dim ** 0.5)
        # ↑ 已弃用：改用 Flash Attention，不再需要手动计算 score 矩阵
        # 核心公式Score=Q*KT/sqrt(dk)
        # .matmul会把这两个矩阵的前两维度理解为并行计算，
        # 它会同时帮 2 个 Batch、每个 Batch 里的 8 个头，各自独立计算那张 512×512 的表
        # 结果 scores 的形状：(2, 8, 512, 512)

        ## # Ep2: 文字接龙的逻辑，不能看到未来的词
        ## if mask is not None:
        ##     scores = scores + mask  # 加上因果掩码 (Causal Mask)，通常是将未来的位置置为负无穷
        # ↑ 已弃用：is_causal=True 已自动处理，不再需要手动加 mask
        # mask是 512*512 的矩阵，矩阵左下角和对角线都是 0，右上角是负无穷，作用是防止模型偷看后续的词
        # Score 矩阵每个元素表示的是行所在的词会从列所在的词保留多少信息来更新下一次的自己
        # 这种屏蔽操作的作用就是不让词用还没读到的词更新自己，防止作弊，强迫它学会推理
        # 比如输入"床前"我们希望它输出"明月光"，结果它一看正确的答案就在后面，他直接就把后面的字搬过来了，
        # 而且不考虑后面的字具体是什么。
        
        ## # Ep7/18: Softmax 归一化，得到概率分布 (Gating)
        ## scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # ↑ 已弃用：Flash Attention 内部自动完成 softmax，不需要手动做
        # F是torch.nn.functional,.softmax就是对 scores 矩阵的行向量化成概率
        # dim=-1 表示对列压扁，确保行被 softmax 化

        ## output = torch.matmul(scores, xv)  # (bsz, n_heads, seqlen, head_dim)
        # ↑ 已弃用：Flash Attention 已经包含了这一步
        # 结果是 "融入上下文信息" 后的新词义 (Ep7 第五部分)

        is_causal = (seqlen > 1) 
        output = F.scaled_dot_product_attention(xq, keys, values, is_causal=is_causal)
        
        # Ep8: 拼接 (Concatenate)
        # 实现了将多头结果拼接回去
        output = output.transpose(1, 2)
        # 第一步：把"头"换回到后面去
        # 形状从 (2, 8, 512, 64) 变为 (2, 512, 8, 64)
        # 逻辑语义：从"8个专家各自的512个词"变回"512个词，每个词有8个专家的意见"

        output = output.contiguous()
        # 第二步：整理内存布局
        # 因为 transpose 只是改变了读取逻辑的步长，数据是乱序的，例如：
        # 转置前：它告诉 CPU，"读完一个数，往后走 1 位读下一个"。
        # 转置后：它告诉 CPU，"读完一个数，先别管隔壁，跳过 3 位去读下一个"。
        # contiguous() 会把数据在内存里重新按顺序排列

        output = output.view(bsz, seqlen, -1)
        # 第三步：缝合（多头融合）
        # 把最后两维 (8, 64) 拍扁成一维 (512)
        # -1 会自动计算 8 * 64 = 512
        
        # Ep8: 融合 (Fusion) 经过 Wo 输出
        return self.wo(output)

# ==============================================================================
# 前馈神经网络：对应 Ep9 (FFN) & SwiGLU 激活函数
# ==============================================================================
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Ep9 第二步：膨胀 (Expansion)
        # SwiGLU 结构通常将维度设为 4*dim 的 2/3 (LLaMA/Qwen 的做法)
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # 确保维度是 multiple_of 的倍数 (硬件计算效率优化)
        hidden_dim = args.multiple_of * (hidden_dim // args.multiple_of)
        # 512*4*2/3=1365不是 256 的倍数，所以我们去计算一个 256 的最小膨胀倍数
        # ((hidden_dim) // args.multiple_of)算下来就是 5
        # 最终hidden_dim是1280


        # Ep9: 定义三个矩阵，对应 SwiGLU 的三个分支
        # w1: 门控分支 (Gate)
        # w3: 内容分支 (Content) - 注意这里代码通常 w3 是内容，w1 是门控，或者反之，取决于实现习惯
        # w2: 压缩分支 (Compression)
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x):
        # Ep9 第三步：SwiGLU = (Swish(Input * W1) * (Input * W3)) * W2
        # F.silu 就是 Swish 函数 (Ep9: x * sigmoid(x))
        # 1. self.w1(x) -> 门控线性变换
        # 2. F.silu(...) -> 激活函数 (非线性)
        # 3. self.w3(x) -> 内容线性变换
        # 4. * -> 逐元素相乘 (Element-wise product)
        # 5. self.w2(...) -> 压缩回原维度 (Ep9 第四步)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
        # SwiGLU(x) = (\frac{xW_1}{1 + e^{-xW_1}} \cdot xW_3)W_2
        # 数学公式: SwiGLU(x) = (SiLU(xW1) * xW3) * W2
        # 其中 SiLU(x) = x * (1 + exp(-x))

# ==============================================================================
# 混合专家架构：对应 Ep18 (MoE) 
# ==============================================================================
class MoeLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_experts = args.n_experts
        self.top_k = args.n_experts_per_tok
        
        # 1. 定义路由器 (Gate)：输入词向量，输出每个专家的分值
        self.gate = nn.Linear(args.dim, args.n_experts, bias=False)
        
        # 2. 定义专家组：用 ModuleList 装起一堆 FeedForward
        self.experts = nn.ModuleList([FeedForward(args) for _ in range(args.n_experts)])
        # nn.MouduleList是特殊的列表，还会同步管理它们的参数、状态和设备，这里封装了 8 个专家
        # 每个专家都一整套FFN结构
        ## self.register_buffer("aux_loss", torch.tensor(0.0))
        ## 防止几个专家垄断所有情况，设计一个辅助损失函数aux_loss，.register_buffer可以保证aux_loss可以在 CPU 和 GPU 移动
        
        # 👇 修复隐患三：不再将 aux_loss 绑定在模型自身属性上，防止 torch.compile 吞掉
        # self.aux_loss = None

    def forward(self, x):
        # x 形状: (Batch, SeqLen, Dim)
        orig_shape = x.shape # 暂存一下x的形状，之后复原的时候要用
        x = x.view(-1, x.shape[-1]) # 拍扁成 (Batch*SeqLen, Dim) 方便计算
        
        logits = self.gate(x) 
        # 经过一次线性变换，logits 矩阵形状为(TotalTokens, n_experts)

        probs = F.softmax(logits, dim=-1)
        # 1. 对 logits 行归一化，化为 probs 概率分布

        ## 防止几个专家垄断所有情况，设计一个辅助损失函数
        ## self.aux_loss.zero_()
        
        # 👇 修复：不再用 self.aux_loss，而是建立一个局部变量，一会通过 return 直接传出去！
        # self.aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        if self.training:
        # 确保每个专家都得到分工，不会被垄断
            # 1. 重要度 (Importance): 这批 Token 对 8 个专家的平均偏好
            importance = probs.mean(0) 
            # importance 是一个 1*8 的行向量，理想情况下，应该8个0.125是最平均的
            
            # 2. 频率 (Frequency): 专家实际被选为 Top-K 的频率
            # 先拿到 Top-K 的索引
            _, indices = torch.topk(probs, self.top_k, dim=-1)
            # indices 是一个(TotalTokens, 2)矩阵，保存每行的两个选中的专家的索引
            mask_flat = torch.zeros_like(probs).scatter_(1, indices, 1)
            # torch.zeros_like(probs)创建一个形状和probs一样的全 0 矩阵
            # target.scatter_(dim, index, src)表示要给 target 这个矩阵的特定位置填 1
            # dim=1 表示按行，dim=0表示按列，index 就是哪些位置要填，src 表示填什么值，我们这里填 1
            frequency = mask_flat.mean(0)
            # 对列求平均，0 表示把行拍扁
            # frequency 是一个行向量，表示8个专家被选择的频率，我们希望它尽可能接近8个0.125
            
            # 3. 计算辅助损失并存入模型成员
            # 数学原理：当 importance 和 frequency 都均匀时，这个乘积和最小
            # 👇 修复：直接赋值给局部变量
            aux_loss = self.n_experts * torch.sum(importance * frequency)
            ## self.aux_loss = self.n_experts * torch.sum(importance * frequency)
            # 垄断水平指数，越大越不均匀importance * frequency是向量点积，
            # 如果模型很看好专家 A（importance 高），而且确实选了专家 A（frequency 高），那么这一项的值就会爆炸式增长。
            # 在完美均匀的情况下（每个专家都被平分），这个 sum 的结果刚好是 1/n_experts。也就是 0.125，极不均匀时为 1
        
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        # 2. 选出最合适的 Top-K 个专家
        # topk_weights 是一个(TotalTokens, 2)矩阵，保存每行的两个专家的权重，
        # topk_indices 也是一个(TotalTokens, 2)矩阵，保存每行的两个专家的索引

        # 👇 致命Bug修复：只有在选的专家数量大于 1 时，才做归一化！
        # 如果 TopK = 1，归一化会让权重永远变成 1.0，梯度就会在除法这里断裂，路由器就成了瞎子！
        if self.top_k > 1:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        # 归一化权重，确保加起来还是 1


        # ==============================================================================
        # 👇 性能飞跃修复：工业界纯 PyTorch 标配写法（Sort & Batched 稀疏计算）
        # 彻底抛弃 .any() 和 .where() 的“动态挑拣”，消除 CPU-GPU 同步阻塞
        # 核心思想：让 Token 按专家排好队 -> 一刀切分 -> 整块送去算 -> 根据原序号拼回去
        # ==============================================================================
        
        N, D = x.shape  # N 是 TotalTokens (比如 24576), D 是维度 Dim (比如 768)
        
        # 第一步：展平选票。
        # 每个 Token 选了 top_k(2) 个专家，相当于我们要处理 N * 2 个独立的小任务
        flat_indices = topk_indices.view(-1)  # 变成一维长条，形状 (N * 2)
        flat_weights = topk_weights.view(-1)  # 变成一维长条，形状 (N * 2)
        # 为了配合选票，把输入的特征矩阵 x 也原地复制两份 (Token0, Token0, Token1, Token1...)
        flat_x = x.repeat_interleave(self.top_k, dim=0)  # 形状变成 (N * 2, Dim)

        # 第二步：核心魔法！按专家 ID 对所有任务进行大排序
        # torch.argsort 会返回从小到大排好序后的“原位置索引”
        # 比如原本选票是 [专家3, 专家0, 专家1, 专家0]，排序后索引告诉你：第2个和第4个任务是去专家0的
        sort_indices = torch.argsort(flat_indices)
        
        # 顺藤摸瓜，把 专家ID、特征矩阵、权重 全部按照这个新顺序重新排队！
        # 排完队后，要去专家 0 的 Token 全挤在最前面，去专家 1 的紧随其后... 内存完全连续！
        sorted_expert_ids = flat_indices[sort_indices]
        sorted_x = flat_x[sort_indices]
        sorted_weights = flat_weights[sort_indices]

        # 第三步：清点人数
        # torch.bincount 是一个纯 GPU 的超快算子，一瞬间就能数出 0, 1, 2, 3 号专家各分到了几个 Token
        # .tolist() 转成 Python 列表，方便下一步做切分
        counts = torch.bincount(sorted_expert_ids, minlength=self.n_experts)
        ends = counts.cumsum(0)
        starts = torch.cat([torch.zeros(1, device=ends.device, dtype=ends.dtype), ends[:-1]])

        expert_outs = []
        for i, expert in enumerate(self.experts):
            s, e = starts[i].item(), ends[i].item()
            if s < e:
                chunk = torch.narrow(sorted_x, 0, s, e - s)
                w = torch.narrow(sorted_weights, 0, s, e - s)
                out_i = expert(chunk)
                out_i = out_i * w.unsqueeze(-1)
                expert_outs.append(out_i)
            else:
                expert_outs.append(torch.empty((0, D), device=x.device, dtype=x.dtype))

        # 第六步：缝合怪
        # 把所有专家算完的碎片，首尾相连拼成一根长条
        cat_outs = torch.cat(expert_outs, dim=0)

        # 第七步：逆排序，物归原主！
        # 创建一个空的长条，然后根据第二步记下的 sort_indices，把结果精准地“塞回”它们当初原本的位置
        out_flat = torch.empty_like(cat_outs)
        out_flat[sort_indices] = cat_outs

        # 第八步：合并 Top-K
        # 把长条变回 (N, 2, Dim) 的形状，然后把同一个 Token 从 2 个专家那里得到的答案加起来！
        # .sum(dim=1) 表示把中间那个“2”压扁，完美融合完成
        out = out_flat.view(N, self.top_k, D).sum(dim=1)

        # 👇 废弃原本导致 CPU-GPU 世纪大堵车的碎步循环（动态索引 + .where + .any）
        ## for i, expert in enumerate(self.experts):
        ##     expert_mask = (topk_indices == i)
        ##     if expert_mask.any():
        ##         token_idx, _ = torch.where(expert_mask)
        ##         ... 此处省略原逻辑 ...

        # 👇 修复：不仅返回最终计算结果，还要把 aux_loss 当作快递一样顺带寄出去
        return out.view(*orig_shape), aux_loss
    

# ==============================================================================
# Transformer 单元：对应 Ep9 (Transformer Unit)
# ==============================================================================
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id # 记录当前是第几层
        self.attention = Attention(args) # args里面有很多参数，Attention类会自己使用它需要的（我们已经定义好了）
        self.feed_forward = MoeLayer(args) if args.use_moe else FeedForward(args) # 根据 use_moe 开关选择 MoE 或普通 FFN
        # Ep9: 两个 RMSNorm，分别用于 Attention 前和 FFN 前
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)


    def forward(self, x, theta_i_cis, mask, start_pos: int = 0):
        # Ep9 第五步：残差连接 (Residual Connection)
        # Result = x + Attention(Norm(x))
        # 这叫 Pre-Norm 结构 (先归一化再进层)，比 Ep9 讲的 Post-Norm 训练更稳定
        h = x + self.attention(self.attention_norm(x), theta_i_cis, mask, start_pos) 
        # 第一次残差连接，保证自己语义信息不丢失
        # x 除了一开始是刚刚经过词嵌入的词矩阵，之后都是隐藏状态的输出，形状永远都是(Batch, Seqlen, Dim)

        # Ep9: FFN 部分的残差连接
        # Result = h + FFN(Norm(h))
        
        ffn_out = self.feed_forward(self.ffn_norm(h))
        if isinstance(ffn_out, tuple):
            moe_out, aux_loss = ffn_out
        else:
            moe_out, aux_loss = ffn_out, torch.tensor(0.0, device=h.device)
        
        return h + moe_out, aux_loss

# ==============================================================================
# Transformer 主模型：对应 Ep1 (大模型整体架构)
# ==============================================================================
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size # 词表尺寸
        self.n_layers = args.n_layers # Tranformer Unit的层数

        # Ep3: 词嵌入层 (Embedding)，将 Token ID 映射为向量
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        
        # Ep9: 堆叠 n_layers 个 TransformerBlock
        self.layers = nn.ModuleList([TransformerBlock(i, args) for i in range(args.n_layers)])
        
        # Ep9 第六步：最后的归一化层
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # Ep2: 输出层 (Unembedding)，将向量映射回 Logits (预测下一个词的概率)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Ep15: 预先计算好所有长度的 RoPE 旋转频率，避免训练时重复计算
        self.register_buffer("theta_i_cis", precompute_rope_operators(args.dim // args.n_heads, args.max_seq_len))
        # .register_buffer让 theta_i_cis 可以在 CPU 和 GPU 移动，是模型的成员，但是不是参数，不会更新

        # 设定掩码逻辑
        mask = torch.full((args.max_seq_len, args.max_seq_len), float("-inf")) 
        # 设定一个掩码方阵，维度数为最大 token 量，全部元素初始化为负无穷
        mask = torch.triu(mask, diagonal=1)
        # .triu表示裁切右上角，diagonal=1表示对角线右移一个
        # [[ 0, -inf, -inf, -inf],  # 第1个词只能看第1个词（它自己）
        #  [ 0,    0, -inf, -inf],  # 第2个词能看第1, 2个词
        #  [ 0,    0,    0, -inf],  # 第3个词能看第1, 2, 3个词
        #  [ 0,    0,    0,    0]]  # 第4个词能看前4个词
        self.register_buffer("mask", mask)
        # .register_buffer让 mask 可以在 CPU 和 GPU 移动，是模型的成员，但是不是参数，不会更新

        self.apply(self._init_weights)
        self.output.weight = self.tok_embeddings.weight


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def setup_cache(self, batch_size: int, max_seq_len: int):
        """在开始 generate 推理前调用此方法分配显存"""
        for layer in self.layers:
            # 预分配全零张量，形状：(Batch, MaxSeq, KV_Heads, Head_Dim)
            layer.attention.cache_k = torch.zeros(
                (batch_size, max_seq_len, layer.attention.n_kv_heads, layer.attention.head_dim),
                device=self.output.weight.device,
                dtype=self.output.weight.dtype
            )
            layer.attention.cache_v = torch.zeros(
                (batch_size, max_seq_len, layer.attention.n_kv_heads, layer.attention.head_dim),
                device=self.output.weight.device,
                dtype=self.output.weight.dtype
            )

    def clear_cache(self):
        """推理结束后调用，清理缓存，释放显存"""
        for layer in self.layers:
            layer.attention.cache_k = None
            layer.attention.cache_v = None


    def forward(self, tokens, inputs_embeds=None, start_pos: int = 0):
            # 1. 第一步：获取初始向量 (h: Batch, SeqLen, Dim)
            if tokens is not None:
                h = self.tok_embeddings(tokens)
            else:
                h = inputs_embeds

            # 2. 第二步，添加掩码逻辑
            _bsz, seqlen, _ = h.shape
            current_mask = self.mask[:seqlen, :seqlen] if seqlen > 1 else None
            # 把我们准备好的因果掩码，切成和当前收到的矩阵一样大的方阵

            # 将预计算好的频率矩阵切片到当前长度，并确保它和 h 在同一设备 (CPU/GPU)
            current_theta_cis = self.theta_i_cis[start_pos : start_pos + seqlen]

            # 新建一个小篮子，用来装所有层产生的 aux_loss
            total_aux_loss = 0.0

            # 3. 第三步：灵魂所在！通过 N 层 Block 循环加工，进行递归调用
            # 这里的 self.layers 就是刚才在 __init__ 里定义的 nn.ModuleList
            for layer in self.layers:
                # 每一层算完的结果，直接当成下一层的输入 (接力赛)
                # 每次交棒，顺手把 aux_loss 扔进小篮子里汇总
                h, layer_aux_loss = layer(h, current_theta_cis, current_mask, start_pos)
                total_aux_loss = total_aux_loss + layer_aux_loss
                ## h = layer(h, current_theta_cis, current_mask)

            # 3. 第四步：最后一层归一化 (Final Norm)
            # 别忘了，Pre-Norm 结构下，最后一层输出后还要拉回标准分布
            h = self.norm(h)

            # 4. 第五步：映射回词表空间 (Logits)
            logits = self.output(h)
            
            return logits.float(), total_aux_loss
            ## return logits.float() # 转回 float 确保计算损失时精度够用