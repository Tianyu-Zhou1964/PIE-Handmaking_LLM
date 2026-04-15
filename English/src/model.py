# ==============================================================================
# Core dependencies: corresponding to Ep1 (deep learning framework) & Ep2 (tensor operations)
# ==============================================================================
import torch                      # Core tensor computation library (Ep2: the foundation of compute power, the stage for matrix multiplication)
import torch.nn as nn             # Neural network component library (Ep6: building blocks like linear layers, embedding layers, etc.)
import torch.nn.functional as F    # Functional interface (Ep7/9: parameter-free operations like Softmax, activation functions, etc.)

# ==============================================================================
# Utilities and type annotations: improving code robustness (production standards)
# ==============================================================================
from dataclasses import dataclass # Auto-generates boilerplate for config classes (Ep4: elegant hyperparameter encapsulation)
from typing import Optional, Tuple # Static type checking (Ep13: clarify inputs/outputs, reduce low-level bugs)
import yaml  # Used to load configuration from config_zh.yaml

# ==============================================================================
# Config class: corresponding to Ep4 & Ep8 (defines dimensions, number of heads, and other hyperparameters)
# ==============================================================================
@dataclass # A convenient tool for simplifying variable initialization
class ModelArgs:
    dim: int = 1024              # Embedding dimension (Ep3: length of the Embedding vector / Ep6: d_model)
    n_layers: int = 16           # Number of Transformer Block layers (Ep9: layer upon layer)
    n_heads: int = 16            # Number of heads in multi-head attention (Ep8: division of labor among experts)
    vocab_size: int = 32128    # Vocabulary size (Ep3: size of the cyber dictionary)
    multiple_of: int = 256      # Multiplier for FFN hidden layer dimension (Ep9: alignment of the expansion layer dimension)
    norm_eps: float = 1e-5      # Tiny value in RMSNorm to prevent division by zero (Ep9/18: epsilon)
    max_seq_len: int = 1024      # Maximum sequence length (Ep2: context window size)
    n_experts: int = 4          # Total number of experts
    n_experts_per_tok: int = 2  # Each Token activates 2 experts (Top-2 routing)
    use_moe: bool = False

    @classmethod
    def get_args(cls, preset: str = "tiny", **kwargs):
        """Retrieve a preset configuration to ensure consistency between training and inference"""
        if preset == "tiny":
            # The key here is: explicitly shove the class-level vocab_size in
            return cls(dim=1024, n_layers=16, n_heads=16, max_seq_len=1024, **kwargs)
        return cls(**kwargs)
    
    @classmethod
    def from_yaml(cls, config_path: str = "config_zh.yaml", **overrides):
        """Build ModelArgs from the 'model' section of a YAML config file; overrides can override any field"""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["model"]
        cfg.update(overrides)
        # Only keep fields defined in the dataclass, ignoring extra keys in yaml (e.g. rope_base)
        import dataclasses
        valid_keys = {field.name for field in dataclasses.fields(cls)}
        filtered = {k: v for k, v in cfg.items() if k in valid_keys}
        return cls(**filtered)

# ==============================================================================
# RMSNorm: corresponding to Ep9 (layer normalization) & Ep18 (Section 2.1 RMS normalization)
# ==============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Ep13: nn.Parameter indicates this is a parameter that can be updated by gradient descent (gamma)
        # Corresponds to Ep9 step one: gamma * x_new + beta (RMSNorm usually omits beta)
        self.weight = nn.Parameter(torch.ones(dim)) 
        # For example, if dim is defined as 512, this line generates a row vector of 1 row * 512 columns, initialized to 1

    def _norm(self, x): 
    # x is the word matrix, a 3D tensor with shape (Batch, Seq, Dim), where batch is the batch size
    # batch serves GPU parallel computation; for example, a 4090 has 16384 CUDA cores — if batch equals one, it's like 16384 chefs cooking a single grain of rice
    # extremely inefficient, so we should choose an appropriate batch to improve memory utilization. When studying the model algorithm we focus on the last two dims: Seq and Dim
    # Seq is the number of tokens, e.g. 10 characters means Seq=10, and is the number of rows in the word matrix; Dim is the number of dimensions, representing how many semantic features a word carries, here 512

        # Following the textbook Ep18 formula: x / sqrt(mean(x^2) + eps)
        # rsqrt is the reciprocal square root function, i.e. 1/sqrt
        # x.pow(2).mean(-1) computes the mean of x^2 (a simplified version of variance, RMSProp idea)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # The -1 here means operating on the last dimension of (Batch, Seq, Dim), i.e. taking the mean of squares along Dim
        # keepdim must be present, otherwise the Dim dimension would be squashed into a scalar by the mean operation.

    def forward(self, x): # This method performs normalization + rescaling, the standard operation of RMS_Norm; it goes from input to output,
        # so it counts as a forward pass, and nn.Module requires it to be named forward
        # Here x still has shape (Batch, Seq, Dim)
        # Ep13: building the computation graph for the forward pass
        # Cast to float to ensure precision; after computing the normalization, multiply by the learnable scaling parameter gamma (self.weight)
        output = self._norm(x.float()).type_as(x)
        # .float() casts to floating point to ensure precision; .type_as(x) keeps the input and output data types the same, e.g. both float16/float32
        return output * self.weight
        # The rescaling operation after normalization; this multiplication is element-wise
        # self.weight will be broadcast from a 1*512 vector to shape (Batch, Seq, Dim)

# ==============================================================================
# RoPE precomputation: corresponding to Ep15 (Rotary Position Encoding)
# ==============================================================================
def precompute_rope_operators(dim: int, seq_len: int, base: float = 10000.0):
    """Precompute a matrix of rotation operators"""
    # Textbook Ep15 Part 3: compute theta_i = 10000^(-2i/d)
    # dim here is head_dim (Ep8: the dimension of each head)
    # A "head" refers to a group of several columns in a matrix; e.g. if total dimension d is 512 and heads is 8, then head_dim is 512/8=64
    # torch.arange(0, dim, 2) corresponds to i in Ep15 (taking even positions), from 0 to 64 (left-closed right-open), step size 2, building a
    # 1D tensor (also a vector) tensor([0,2,4,...,62])
    # [: (dim // 2)] means take from 0 up to dim//2 which is 32, 32 still excluded (left-closed right-open)
    theta_i = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Constructing the rotation frequency, equivalent to theta_i=1/(base^(2i/d)), base is usually 10000
    # That is, theta_i = 10000^(-2i/d)
    
    # Generate sequence position indices m (0, 1, 2, ..., seq_len-1)
    # Corresponds to m in Ep15 (represents which row of the word matrix)
    m = torch.arange(seq_len, device=theta_i.device)
    # device=theta_i.device ensures m and theta_i are on the same device, e.g. the GPU
    # Compute m * theta_i to get the rotation angle for every position at every frequency
    # Corresponds to m * theta in Ep15
    theta_i = torch.outer(m, theta_i).float()  # (seq_len, dim/2)
    # outer(m, theta_i) means the first element of m multiplies every element of θi, then the second element of m multiplies θi, until the last element of seq_len
    # Convert angles to complex number form: cos(m*theta) + i*sin(m*theta)
    # Corresponds to Ep17: Euler's formula e^(ix) = cosx + isinx
    # torch.polar(magnitude, angle) -> complex vector with magnitude 1
    theta_i_cis = torch.polar(torch.ones_like(theta_i), theta_i)  # complex64
    # cis stands for: cos + i sin. .polar is polar coordinates, defining a complex number using magnitude torch.ones_like(theta_i)
    # and angle theta_i, so theta_i_cis is a matrix of rotation operators
    # All elements have magnitude 1, so applying it only changes angles without changing lengths; size is m*32
    return theta_i_cis

# ==============================================================================
# Apply RoPE: corresponding to Ep15 (Part 4: rotation via complex multiplication)
# ==============================================================================
def apply_rope(x: torch.Tensor, theta_i_cis: torch.Tensor):
    # Corresponds to Ep15 "grouping strategy (pairing up two by two)"
    # Now this x is no longer a 3D tensor of shape (Batch, Seq, 512), but (Batch, Seq, 8, 64), where 8 is the number of heads
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # x.shape[:-1] takes all values except the last dimension 64, and * unpacks the tuple into individual numbers
    # For example if x has shape (2,512,8,64), *x.shape[:-1] gives 2,512,8
    # The fourth argument -1 in reshape auto-computes 524,288 / (2 * 512 * 8 * 2) = 32
    # So reshape(2, 512, 8, -1, 2) becomes reshape(2, 512, 8, 32, 2)
    # So x_complex becomes a matrix of 2 batches, 512 rows, 8 heads, 32 slots, with 2 elements per slot
    # The outermost .view_as_complex treats the innermost two elements as the real and imaginary parts of a complex number
    theta_i_cis = theta_i_cis.view(1, x_complex.shape[1], 1, x_complex.shape[-1])
    # This .view operation is for aligning dimensions; theta_i_cis only has two orders, while x_complex has four orders,
    # so we fabricate two virtual orders for theta_i_cis: batch and heads, both set to 1
    # theta_i_cis becomes a 4D tensor of shape (1, 512, 1, 32)
    rotated_complex = x_complex * theta_i_cis
    # x_complex is (2, 512, 8, 32), theta_i_cis is (1, 512, 1, 32)
    # Two complex matrices multiplied directly, applying complex multiplication rules to complete the rotation
    x_real_grouped = torch.view_as_real(rotated_complex)
     # Shape changes from (B, S, H, 32) to (B, S, H, 32, 2)
    # The last dimension 2 represents [real part, imaginary part] respectively
    x_out = x_real_grouped.flatten(3)
    # Shape changes from (B, S, H, 32, 2) to (B, S, H, 64)
    # flatten(3) means flatten everything starting from the fourth dimension (i.e. the 32 dimension) onwards
    return x_out.type_as(x)
    # .type_as appeared earlier; still ensures input and output have the same dtype, e.g. float16

# ==============================================================================
# Attention mechanism: corresponding to Ep6 (Q,K,V), Ep7 (Attention), Ep8 (multi-head)
# ==============================================================================
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        # Ep8: dimension per head = total dimension / number of heads
        self.head_dim = args.dim // args.n_heads

        # GQA: K and V have 1/4 as many heads as Q, reducing KV Cache memory usage by 4× during inference
        # For example Q uses 8 heads, K/V only use 2 heads; in forward, K/V will be repeated to match Q's head count before computing attention
        self.n_kv_heads = args.n_heads // 4

        # Ep6: define linear transformation matrices Wq, Wk, Wv
        # Corresponds to the formula Q = X * Wq, K = X * Wk, V = X * Wv
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        # GQA: output dimension of Wk and Wv is only n_kv_heads * head_dim, 4x smaller than Wq
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # nn.Linear creates a weight parameter matrix W of size args.dim*args.dim (here 512*512), with no bias
        # Each element is initialized as an independent random variable with mean 0 and variance 1, then used for training
        # Note that instances created by nn.Linear implement the __call__ function internally, so they can be called as functions
        # This feature is about to be used right away

        # Ep8: output transformation matrix Wo (for multi-head Fusion)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)
        # Exactly the same as before, also trainable
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: torch.Tensor, theta_i_cis: torch.Tensor, mask: Optional[torch.Tensor], start_pos: int = 0):
        bsz, seqlen, _ = x.shape
        # bsz is batchsize, seqlen is the number of rows, _ is the dimension count which we don't use here
        
        # Ep6: compute Q, K, V (linear projection)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # Equivalent to xq=x*self.wq, x: 2*512*512, wq: 512*512, after matrix multiplication xq is still 2*512*512

        # Ep8: reshape into multi-head form (batch, seqlen, n_heads, head_dim)
        # Split the large vector into n_heads parts
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        # GQA: K and V are only split into n_kv_heads parts, fewer than Q
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        # This essentially splits the original dim=512 into two dimensions: 8*64, where 8 heads each take 64 of the 512 columns

        # Ep15: apply RoPE rotary position encoding
        # Note: only Q and K are rotated, V is not (Ep15 Part 3)
        xq = apply_rope(xq, theta_i_cis)
        xk = apply_rope(xk, theta_i_cis)
        # Call the rotation operation defined earlier; at this point xq and xk have been fully injected with positional information

        if self.cache_k is not None and self.cache_v is not None:
            # 1. Write the newly computed K, V into the corresponding positions in the cache
            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
            
            # 2. Retrieve all K, V from position 0 up to the current position for subsequent attention computation
            # This way, no matter how many words you pass in, we can always get the full context!
            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            # If no cache has been allocated (meaning we're in the training phase), just use the current ones
            keys = xk
            values = xv

        # Adjust dimensions for matrix multiplication (batch, n_heads, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)     # 👇 Modified: use the retrieved keys
        values = values.transpose(1, 2) # 👇 Modified: use the retrieved values

        # Taking xk as an example: xk without transpose is (2, 512, 8, 64), .transpose(1,2) swaps the 2nd and 3rd dimensions
        # Effectively transposes xk to: (2, 8, 512, 64), making it ready for subsequent matrix multiplication
        # After this step, xq, xk, xv are distributed across heads; batch=2 computes two in parallel, head=8 has 8 heads
        # Each head has a 512-row word matrix and is responsible for 64 semantic features

        # GQA: repeat K and V heads to match Q's head count, so dimensions align for matrix multiplication
        # repeat_interleave(4, dim=1) repeats each kv_head in-place 4 times
        # For example K was originally (2, 2, 512, 64), after repeat it becomes (2, 8, 512, 64), aligned with Q
        # This is only a logical broadcast copy; the actual parameter count does not increase
        n_rep = self.n_heads // self.n_kv_heads  # = 4
        keys = keys.repeat_interleave(n_rep, dim=1)     # 👇 Modified
        values = values.repeat_interleave(n_rep, dim=1) # 👇 Modified

        # Ep7: Flash Attention replaces the hand-written matmul + softmax + matmul
        # F.scaled_dot_product_attention is PyTorch 2.0's built-in Flash Attention implementation
        # Core advantage: instead of writing the full (B, H, S, S) score matrix to memory, it computes in tiles, saving 30~50% memory and running 20~30% faster
        # is_causal=True automatically handles the causal mask; internally it works the same as our mask, but doesn't require explicitly passing in a -inf matrix
        # / (self.head_dim ** 0.5) corresponds to Ep7 "Scaling": divide by sqrt(d_k); Flash Attention handles this internally
        # Output shape: (bsz, n_heads, seqlen, head_dim), identical to the original torch.matmul(scores, xv)

        ## scores = torch.matmul(xq, xk.transpose(2, 3)) / (self.head_dim ** 0.5)
        # ↑ Deprecated: replaced by Flash Attention; no longer need to manually compute the score matrix
        # Core formula: Score = Q * K^T / sqrt(dk)
        # .matmul treats the first two dimensions of these matrices as parallel computation,
        # it simultaneously helps 2 Batches, each with 8 heads, each independently compute that 512×512 table
        # Resulting scores shape: (2, 8, 512, 512)

        ## # Ep2: word prediction logic, cannot look at future words
        ## if mask is not None:
        ##     scores = scores + mask  # Add the causal mask; typically sets future positions to negative infinity
        # ↑ Deprecated: is_causal=True already handles this; no longer need to manually add mask
        # mask is a 512*512 matrix; the lower-left triangle and diagonal are 0, the upper-right triangle is negative infinity, preventing the model from peeking at subsequent words
        # Each element in the Score matrix represents how much information the word at that row will retain from the word at that column to update itself
        # This masking operation prevents words from updating themselves with words not yet read, preventing cheating and forcing the model to learn to reason
        # For example, if the input is "床前" (bedside) we want it to output "明月光" (moonlight), but if it can just look at the correct answer right after itself, it'll copy it directly
        # without considering what the subsequent characters actually are.
        
        ## # Ep7/18: Softmax normalization to get a probability distribution (Gating)
        ## scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # ↑ Deprecated: Flash Attention handles softmax internally; no longer needed
        # F is torch.nn.functional; .softmax converts each row vector of the scores matrix into probabilities
        # dim=-1 means collapse along the column axis, ensuring rows are softmax-normalized

        ## output = torch.matmul(scores, xv)  # (bsz, n_heads, seqlen, head_dim)
        # ↑ Deprecated: Flash Attention already includes this step
        # The result is the new word semantics "fused with contextual information" (Ep7 Part 5)

        is_causal = (seqlen > 1) 
        output = F.scaled_dot_product_attention(xq, keys, values, is_causal=is_causal)
        
        # Ep8: Concatenation
        # Implements concatenating multi-head results back together
        output = output.transpose(1, 2)
        # Step 1: move the "heads" dimension back to the end
        # Shape changes from (2, 8, 512, 64) to (2, 512, 8, 64)
        # Logical semantics: from "512 words each seen by 8 experts separately" back to "512 words, each with opinions from 8 experts"

        output = output.contiguous()
        # Step 2: reorganize the memory layout
        # Because transpose only changed the stride for reading logic, the data is out of order, for example:
        # Before transpose: it tells the CPU, "after reading one number, move forward 1 position to read the next."
        # After transpose: it tells the CPU, "after reading one number, skip 3 positions to read the next."
        # contiguous() re-arranges the data sequentially in memory

        output = output.view(bsz, seqlen, -1)
        # Step 3: merge (multi-head fusion)
        # Flatten the last two dimensions (8, 64) into one dimension (512)
        # -1 automatically computes 8 * 64 = 512
        
        # Ep8: Fusion — pass through Wo for output
        return self.wo(output)

# ==============================================================================
# Feed-forward neural network: corresponding to Ep9 (FFN) & SwiGLU activation function
# ==============================================================================
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Ep9 Step 2: Expansion
        # SwiGLU structure typically sets the dimension to 2/3 of 4*dim (the LLaMA/Qwen approach)
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # Ensure the dimension is a multiple of multiple_of (hardware compute efficiency optimization)
        hidden_dim = args.multiple_of * (hidden_dim // args.multiple_of)
        # 512*4*2/3=1365 is not a multiple of 256, so we compute the smallest expansion multiple of 256
        # ((hidden_dim) // args.multiple_of) works out to 5
        # Final hidden_dim is 1280


        # Ep9: define three matrices corresponding to the three branches of SwiGLU
        # w1: gate branch (Gate)
        # w3: content branch (Content) - note that in code conventions w3 is usually content and w1 is gate, or vice versa, depending on implementation style
        # w2: compression branch (Compression)
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x):
        # Ep9 Step 3: SwiGLU = (Swish(Input * W1) * (Input * W3)) * W2
        # F.silu is the Swish function (Ep9: x * sigmoid(x))
        # 1. self.w1(x) -> gate linear transform
        # 2. F.silu(...) -> activation function (non-linearity)
        # 3. self.w3(x) -> content linear transform
        # 4. * -> element-wise multiplication
        # 5. self.w2(...) -> compress back to the original dimension (Ep9 Step 4)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
        # SwiGLU(x) = (\frac{xW_1}{1 + e^{-xW_1}} \cdot xW_3)W_2
        # Math formula: SwiGLU(x) = (SiLU(xW1) * xW3) * W2
        # where SiLU(x) = x * (1 + exp(-x))

# ==============================================================================
# Mixture of Experts architecture: corresponding to Ep18 (MoE) 
# ==============================================================================
class MoeLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_experts = args.n_experts
        self.top_k = args.n_experts_per_tok
        
        # 1. Define the router (Gate): takes a word vector as input, outputs a score for each expert
        self.gate = nn.Linear(args.dim, args.n_experts, bias=False)
        
        # 2. Define the expert pool: pack a bunch of FeedForward modules into a ModuleList
        self.experts = nn.ModuleList([FeedForward(args) for _ in range(args.n_experts)])
        # nn.ModuleList is a special list that also synchronously manages the parameters, states, and devices of its children; here it wraps 8 experts
        # Each expert has a full FFN structure
        ## self.register_buffer("aux_loss", torch.tensor(0.0))
        ## To prevent a few experts from monopolizing all cases, we design an auxiliary loss function aux_loss; .register_buffer ensures aux_loss can be moved between CPU and GPU
        
        # 👇 Fix for hidden issue three: no longer binding aux_loss as a model attribute, preventing torch.compile from swallowing it
        # self.aux_loss = None

    def forward(self, x):
        # x shape: (Batch, SeqLen, Dim)
        orig_shape = x.shape # Save x's shape temporarily; needed for restoring later
        x = x.view(-1, x.shape[-1]) # Flatten to (Batch*SeqLen, Dim) for easier computation
        
        logits = self.gate(x) 
        # After one linear transformation, logits matrix has shape (TotalTokens, n_experts)

        probs = F.softmax(logits, dim=-1)
        # 1. Row-normalize logits into a probability distribution probs

        ## To prevent a few experts from monopolizing all cases, design an auxiliary loss function
        ## self.aux_loss.zero_()
        
        # 👇 Fix: no longer using self.aux_loss; instead create a local variable and pass it out directly through return!
        # self.aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        if self.training:
        # Ensure every expert gets assigned work and is not monopolized
            # 1. Importance: the average preference of this batch of Tokens for each of the 8 experts
            importance = probs.mean(0) 
            # importance is a 1*8 row vector; ideally all 8 values should be 0.125 for maximum balance
            
            # 2. Frequency: the rate at which each expert is actually selected as Top-K
            # First get the Top-K indices
            _, indices = torch.topk(probs, self.top_k, dim=-1)
            # indices is a (TotalTokens, 2) matrix storing the indices of the two selected experts per row
            mask_flat = torch.zeros_like(probs).scatter_(1, indices, 1)
            # torch.zeros_like(probs) creates an all-zero matrix with the same shape as probs
            # target.scatter_(dim, index, src) fills specific positions in target with the value from src
            # dim=1 means by row, dim=0 means by column, index is which positions to fill, src is the fill value, here 1
            frequency = mask_flat.mean(0)
            # Take the mean along the row axis; 0 means collapse the rows
            # frequency is a row vector representing how often each of the 8 experts is selected; we want it as close to 8 values of 0.125 as possible
            
            # 3. Compute the auxiliary loss and store it
            # Mathematical principle: when both importance and frequency are uniform, this product sum is minimized
            # 👇 Fix: directly assign to the local variable
            aux_loss = self.n_experts * torch.sum(importance * frequency)
            ## self.aux_loss = self.n_experts * torch.sum(importance * frequency)
            # Monopolization index: larger means more uneven. importance * frequency is a vector dot product;
            # if the model heavily favors expert A (high importance) and actually selects expert A (high frequency), this term grows explosively.
            # In the perfectly uniform case (every expert gets an equal share), the sum result is exactly 1/n_experts, i.e. 0.125; at maximum imbalance it approaches 1
        
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        # 2. Select the most suitable Top-K experts
        # topk_weights is a (TotalTokens, 2) matrix storing the weights of the two selected experts per row
        # topk_indices is also a (TotalTokens, 2) matrix storing the indices of the two selected experts per row

        # 👇 Critical bug fix: only normalize when the number of selected experts is greater than 1!
        # If TopK = 1, normalization would make the weight always 1.0; gradients would break at the division, and the router would go blind!
        if self.top_k > 1:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        # Normalize weights to ensure they still sum to 1


        # ==============================================================================
        # 👇 Performance leap fix: industry-standard pure PyTorch approach (Sort & Batched sparse computation)
        # Completely abandons the "dynamic picking" of .any() and .where(), eliminating CPU-GPU synchronization stalls
        # Core idea: have Tokens queue up sorted by expert -> slice with a single cut -> send each chunk to compute -> reassemble according to original indices
        # ==============================================================================
        
        N, D = x.shape  # N is TotalTokens (e.g. 24576), D is dimension Dim (e.g. 768)
        
        # Step 1: Flatten the ballots.
        # Each Token selects top_k(2) experts, meaning we have N * 2 independent mini-tasks to handle
        flat_indices = topk_indices.view(-1)  # Flatten into a 1D strip, shape (N * 2)
        flat_weights = topk_weights.view(-1)  # Flatten into a 1D strip, shape (N * 2)
        # To match the ballots, duplicate the input feature matrix x in-place (Token0, Token0, Token1, Token1...)
        flat_x = x.repeat_interleave(self.top_k, dim=0)  # Shape becomes (N * 2, Dim)

        # Step 2: The core magic! Sort all tasks by expert ID
        # torch.argsort returns the "original position indices" sorted from smallest to largest
        # For example if the original ballots are [Expert3, Expert0, Expert1, Expert0], the sorted indices tell you: the 2nd and 4th tasks go to Expert0
        sort_indices = torch.argsort(flat_indices)
        
        # Follow the thread: re-queue expert IDs, feature matrix, and weights all according to this new order!
        # After queuing, Tokens going to Expert 0 all crowd to the front, those going to Expert 1 follow next... memory is fully contiguous!
        sorted_expert_ids = flat_indices[sort_indices]
        sorted_x = flat_x[sort_indices]
        sorted_weights = flat_weights[sort_indices]

        # Step 3: Count up
        # torch.bincount is a blazing-fast pure GPU operator that instantly counts how many Tokens experts 0, 1, 2, 3 each received
        # .tolist() converts to a Python list for convenient slicing in the next step
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

        # Step 6: Frankenstein assembly
        # Concatenate the fragments computed by all experts end-to-end into a single long strip
        cat_outs = torch.cat(expert_outs, dim=0)

        # Step 7: Reverse sort — return everything to its rightful owner!
        # Create an empty strip, then use the sort_indices recorded in Step 2 to precisely "stuff" each result back into its original position
        out_flat = torch.empty_like(cat_outs)
        out_flat[sort_indices] = cat_outs

        # Step 8: Merge Top-K
        # Reshape the strip back to (N, 2, Dim), then sum up the answers each Token received from its 2 experts!
        # .sum(dim=1) collapses the middle "2" axis, completing the perfect fusion
        out = out_flat.view(N, self.top_k, D).sum(dim=1)

        # 👇 Deprecated: the old loop that caused a century-long CPU-GPU traffic jam (dynamic indexing + .where + .any)
        ## for i, expert in enumerate(self.experts):
        ##     expert_mask = (topk_indices == i)
        ##     if expert_mask.any():
        ##         token_idx, _ = torch.where(expert_mask)
        ##         ... original logic omitted here ...

        # 👇 Fix: not only return the final computation result, but also ship the aux_loss out like a courier package
        return out.view(*orig_shape), aux_loss
    

# ==============================================================================
# Transformer unit: corresponding to Ep9 (Transformer Unit)
# ==============================================================================
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id # Record which layer this is
        self.attention = Attention(args) # args contains many parameters; the Attention class will use what it needs (we've already defined them)
        self.feed_forward = MoeLayer(args) if args.use_moe else FeedForward(args) # Select MoE or standard FFN based on the use_moe switch
        # Ep9: two RMSNorm layers, used before Attention and before FFN respectively
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)


    def forward(self, x, theta_i_cis, mask, start_pos: int = 0):
        # Ep9 Step 5: Residual Connection
        # Result = x + Attention(Norm(x))
        # This is called Pre-Norm structure (normalize before entering the layer), which is more stable to train than the Post-Norm discussed in Ep9
        h = x + self.attention(self.attention_norm(x), theta_i_cis, mask, start_pos) 
        # First residual connection, ensuring the original semantic information is not lost
        # x starts as the word matrix freshly passed through word embedding; afterward it's always the hidden state output, shape always (Batch, Seqlen, Dim)

        # Ep9: residual connection for the FFN part
        # Result = h + FFN(Norm(h))
        
        ffn_out = self.feed_forward(self.ffn_norm(h))
        if isinstance(ffn_out, tuple):
            moe_out, aux_loss = ffn_out
        else:
            moe_out, aux_loss = ffn_out, torch.tensor(0.0, device=h.device)
        
        return h + moe_out, aux_loss

# ==============================================================================
# Main Transformer model: corresponding to Ep1 (overall large model architecture)
# ==============================================================================
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size # Vocabulary size
        self.n_layers = args.n_layers # Number of Transformer Unit layers

        # Ep3: Embedding layer, maps Token IDs to vectors
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        
        # Ep9: stack n_layers TransformerBlocks
        self.layers = nn.ModuleList([TransformerBlock(i, args) for i in range(args.n_layers)])
        
        # Ep9 Step 6: the final normalization layer
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # Ep2: output layer (Unembedding), maps vectors back to Logits (predicted probability of the next word)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Ep15: precompute RoPE rotation frequencies for all lengths to avoid repeated computation during training
        self.register_buffer("theta_i_cis", precompute_rope_operators(args.dim // args.n_heads, args.max_seq_len))
        # .register_buffer allows theta_i_cis to move between CPU and GPU; it is a member of the model but not a parameter, so it won't be updated

        # Set up the masking logic
        mask = torch.full((args.max_seq_len, args.max_seq_len), float("-inf")) 
        # Create a square mask matrix with dimensions equal to the maximum token count; all elements initialized to negative infinity
        mask = torch.triu(mask, diagonal=1)
        # .triu trims the upper-right triangle; diagonal=1 shifts the diagonal one position to the right
        # [[ 0, -inf, -inf, -inf],  # Word 1 can only see word 1 (itself)
        #  [ 0,    0, -inf, -inf],  # Word 2 can see words 1 and 2
        #  [ 0,    0,    0, -inf],  # Word 3 can see words 1, 2, and 3
        #  [ 0,    0,    0,    0]]  # Word 4 can see all 4 previous words
        self.register_buffer("mask", mask)
        # .register_buffer allows mask to move between CPU and GPU; it is a member of the model but not a parameter, so it won't be updated

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
        """Call this method before starting generate inference to allocate GPU memory"""
        for layer in self.layers:
            # Pre-allocate all-zero tensors, shape: (Batch, MaxSeq, KV_Heads, Head_Dim)
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
        """Call after inference ends to clear the cache and free GPU memory"""
        for layer in self.layers:
            layer.attention.cache_k = None
            layer.attention.cache_v = None


    def forward(self, tokens, inputs_embeds=None, start_pos: int = 0):
            # 1. Step one: obtain the initial vectors (h: Batch, SeqLen, Dim)
            if tokens is not None:
                h = self.tok_embeddings(tokens)
            else:
                h = inputs_embeds

            # 2. Step two: add masking logic
            _bsz, seqlen, _ = h.shape
            current_mask = self.mask[:seqlen, :seqlen] if seqlen > 1 else None
            # Slice the prepared causal mask into a square matrix matching the size of the currently received matrix

            # Slice the precomputed frequency matrix to the current length, and ensure it is on the same device as h (CPU/GPU)
            current_theta_cis = self.theta_i_cis[start_pos : start_pos + seqlen]

            # Create a small basket to collect the aux_loss produced by all layers
            total_aux_loss = 0.0

            # 3. Step three: the soul of it all! Iteratively process through N Block layers, performing recursive calls
            # self.layers here is the nn.ModuleList defined in __init__
            for layer in self.layers:
                # The output of each layer is directly used as the input of the next (relay race)
                # With each handoff, toss the aux_loss into the basket for accumulation
                h, layer_aux_loss = layer(h, current_theta_cis, current_mask, start_pos)
                total_aux_loss = total_aux_loss + layer_aux_loss
                ## h = layer(h, current_theta_cis, current_mask)

            # 3. Step four: final layer normalization (Final Norm)
            # Don't forget: under the Pre-Norm structure, after the last layer's output we still need to pull it back to a standard distribution
            h = self.norm(h)

            # 4. Step five: map back to vocabulary space (Logits)
            logits = self.output(h)
            
            return logits.float(), total_aux_loss
            ## return logits.float() # Cast back to float to ensure sufficient precision when computing the loss
