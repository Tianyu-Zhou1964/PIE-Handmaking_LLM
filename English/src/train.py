# ==============================================================================
# Core dependencies: the command center of model training
# ==============================================================================
import torch                      # Core deep learning framework
import numpy as np                # Numerical computation library
from torch.utils.data import Dataset, DataLoader # Data pipeline components
from torch.utils.data.distributed import DistributedSampler # DDP: splits the dataset into N shards, each GPU sees only its own shard
from torch.nn.parallel import DistributedDataParallel       # DDP: model wrapper that automatically synchronizes gradients across GPUs
import torch.distributed as dist                            # DDP: process group communication, the low-level interface for AllReduce gradient sync
from model import Transformer, ModelArgs # Import our hand-built model architecture (Ep1-Ep18)
import os # For file system operations (create/delete/read/write)
import math # For the math behind hand-written warmup + cosine decay
import yaml # Import the YAML parsing library

# ==============================================================================
# Training configuration: defines the "hyperparameters" of training
# ==============================================================================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_BASE_DIR, "../config_en.yaml"), "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)
    _train_cfg = _cfg["training"]
    _model_cfg = _cfg["model"]

DATA_PATH     = _train_cfg["data_path"] # Corpus path
BATCH_SIZE    = _train_cfg["batch_size"] # Batch size
SEQ_LEN       = _train_cfg["seq_len"] # Sequence length
LR_MAX        = _train_cfg["learning_rate"] # Peak learning rate
EPOCHS        = _train_cfg["epochs"] # Number of training epochs
VOCAB_SIZE    = _model_cfg["vocab_size"] # Vocabulary size
WARMUP_STEPS  = _train_cfg["warmup_steps"] # Number of warmup steps
LR_MIN        = _train_cfg["lr_min"] # Minimum learning rate

# ==============================================================================
# Learning rate schedule: hand-written warmup + cosine decay, replacing CosineAnnealingLR
# ==============================================================================
# ---- Core idea ----
# The learning rate (LR) is like the speed of riding a bicycle:
#   1. Warmup phase: accelerate slowly from a standstill to avoid stomping the pedals and snapping the chain (gradient explosion)
#   2. Cosine Decay phase: after reaching cruising speed, smoothly decelerate along a cosine curve,
#      letting the model "fine-tune" near the loss valley instead of overshooting
# Why not use PyTorch's built-in CosineAnnealingLR?
#   Because it requires you to hard-code T_max (total steps) upfront, but in DDP multi-GPU training
#   the number of batches each GPU receives will vary.
#   If T_max is set incorrectly, the cosine curve will be misaligned with the actual training progress:
#   the LR either bottoms out too early or hasn't fully decayed when training ends.
#   Hand-writing it with a single math.cos is actually more flexible and less bug-prone.

def get_lr(step: int) -> float:
    """
    Hand-written learning rate scheduler: given the current step, returns the LR to use at this moment.
    The full curve looks like this:
      LR
      ▲
      │      ╭──╮                ← Peak LR_MAX (3e-4)
      │     ╱    ╲
      │    ╱      ╲              ← Smooth cosine decay
      │   ╱        ╲
      │  ╱          ╲────────    ← Floor LR_MIN (3e-5)
      │ ╱  warmup
      └──────────────────────→ step
        0   400            T_MAX
    """
    # Warmup phase: linearly increase from 0 to LR_MAX
    if step < WARMUP_STEPS:
        return LR_MAX * step / WARMUP_STEPS
    # Cosine decay phase: smoothly decay from LR_MAX down to LR_MIN
    # progress ∈ [0, 1], representing "how far along you are after warmup ends"
    # cos(0) = 1 → LR is at its peak at the start of decay
    # cos(π) = -1 → LR is at its lowest at the end of decay
    # Wrapping with 0.5*(1+cos) compresses the range to [0, 1],
    # then multiply by (peak - floor) + floor to perfectly map the interval
    progress = (step - WARMUP_STEPS) / (T_MAX_STEPS - WARMUP_STEPS)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))

# ==============================================================================
# Dataset class: defines how to read and slice data
# ==============================================================================
class PretokenizedDataset(Dataset):
    # PretokenizedDataset inherits from Dataset, which requires you to implement __len__ and __getitem__
    def __init__(self, data_path, seq_len):
        super().__init__()
        # Use np.fromfile to instantly load the entire binary corpus into memory,
        # directly mapping the binary data on disk into RAM
        self.data = np.fromfile(data_path, dtype=np.uint32)
        # uint32 tells NumPy how to "interpret" this binary 0/1 sequence;
        # 32 means each number occupies 32 bits, i.e. 4 bytes
        # This single line loads all binary data from data_path into self.data,
        # cutting every 4 bytes into one element (contiguous memory)
        self.seq_len = seq_len
        # Store the sequence length

    def __getitem__(self, idx):
        # Note: any method named __xxx__ is called a "magic method" and is invoked automatically when triggered.
        # Here __getitem__ is triggered by obj[i] (assuming the instance is named obj)
        # The [i] means "which chunk"

        # The argument idx indicates which seq chunk this is;
        # idx=0 means the 1st chunk of 1024 tokens, idx=1 means the 2nd chunk of 1024 tokens

        start_idx = idx * self.seq_len
        # Compute the starting index
        end_idx = start_idx + self.seq_len
        # Compute the ending index
        
        chunk = np.array(self.data[start_idx : end_idx + 1])
        # Create the chunk, storing the 1025 tokens retrieved
        # (one extra token is needed because y is shifted one position to the right)
        
        # First 1024 tokens as input, last 1024 tokens as labels
        # .astype(np.int64): PyTorch's CrossEntropyLoss requires labels to be int64 (Long)
        # Input is also cast to int64 to align with the index type of nn.Embedding
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        # Example: with seq_len=4, chunk = [床, 前, 明, 月, 光]
        #   x = [床, 前, 明, 月]   ← chunk[:-1], the input the model sees
        #   y = [前, 明, 月, 光]   ← chunk[1:], the answers the model is asked to predict

        return (x, y)
        # Returns a tuple containing two lists: input list x and label list y
        # The tokens are still vocabulary indices at this point, not yet embedded into vectors
    
    def __len__(self):
        return (len(self.data) - 1) // self.seq_len
        # Counts how many total chunks there are
        # Subtract 1 because each chunk needs one extra token for y, so total token count must be reduced by 1

# ==============================================================================
# Main training loop: the "evolution" process of a large language model
# ==============================================================================
def train():
    # ==============================================================================
    # DDP initialization: establish the multi-GPU process group, one process per GPU
    # ==============================================================================
    # ---- What is DDP? ----
    # DistributedDataParallel: PyTorch's multi-GPU parallel training solution
    # Core idea: run one completely independent Python process per GPU, each holding a full copy of the model
    # During training:
    #   1. Data is split into N shards by DistributedSampler; each GPU sees only its own shard (data parallelism)
    #   2. Each GPU independently does forward + backward, computing its own gradients
    #   3. NCCL (NVIDIA's high-speed inter-GPU communication library) performs AllReduce,
    #      i.e. computes a weighted average of all GPUs' gradients
    #   4. Each GPU updates its parameters using the same averaged gradient → ensures all GPU model weights stay in sync
    # Advantage: unlike DataParallel, there is no "master GPU bottleneck"; communication and computation can overlap,
    #            making it far more efficient

    use_ddp = "RANK" in os.environ
    if use_ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        # backend='nccl': specifies the NCCL protocol for communication, the de facto standard for NVIDIA GPU multi-GPU training
        # This line blocks until all processes (all GPUs) have called it, then they are all released together
        local_rank = int(os.environ["LOCAL_RANK"])
        # Which GPU this process corresponds to (e.g. 0/1/2/3)
        # LOCAL_RANK is automatically injected as an environment variable by torchrun; no manual setup needed
        world_size = dist.get_world_size()
        # Total number of GPUs participating in training
        # Bind each process to its own GPU, preventing all processes from piling onto GPU 0
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        local_rank = 0
        world_size = 1
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps" # Apple M-chip goes here!
        else:
            device = "cpu" # If you end up here, respect.
            
    is_master = (local_rank == 0)
    # Only rank 0 (the master process) prints and saves, to avoid 3 processes printing simultaneously and mangling the logs


    # ==============================================================================
    # 1. Prepare the data pipeline: the complete path from disk to GPU
    # ==============================================================================
    # Data flow: binary file on disk → Dataset (chunking) → Sampler (sharding) → DataLoader (batching) → GPU
    dataset = PretokenizedDataset(DATA_PATH, SEQ_LEN)
    # dataset is an instance of the PretokenizedDataset class
    # To access the i-th chunk, just use dataset[i], which returns a tuple (x, y)
    # x is the input, y is the label (i.e. the desired output)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True) if use_ddp else None
    # DistributedSampler: one of the core components of DDP
    # It evenly distributes the dataset indices [0, 1, 2, ..., N-1] across world_size GPUs
    # Example: 3 GPUs, 9 samples: GPU0 gets [0,3,6], GPU1 gets [1,4,7], GPU2 gets [2,5,8]
    # shuffle=True ensures the distribution order differs every epoch, preventing the model from "memorizing" data order
    # The sampler is ultimately an integer list (a bunch of idx values), telling each GPU's loader
    # which chunks to fetch from the dataset
    # It ferries the specified data to the current GPU for computation

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                        shuffle=(not use_ddp),
                        num_workers=_train_cfg["num_workers"],
                        pin_memory=(device != "mps"),
                        persistent_workers=(_train_cfg["num_workers"] > 0))
    # DataLoader: the "conveyor belt" for data
    # batch_size=14: each time, fetches 14 samples from the sampler and packs them into one batch
    # dataset[0] → x shape=(1024,)
    # dataset[3] → x shape=(1024,)
    # dataset[6] → x shape=(1024,)
    # ...collect 14 samples...
    # → stacked into a large X batch with shape=(14, 1024); Y is the same size

    # num_workers=8: spawn 8 subprocesses to pre-load the next batch from disk in parallel, so the GPU doesn't wait on the CPU
    # pin_memory=True: pre-pin data in "page-locked" CPU memory to accelerate CPU→GPU transfer (DMA direct path)
    # "Page-locked" means this memory is not allowed to be swapped back to disk — keep it locked in RAM!

    # persistent_workers=True: worker processes are not destroyed and recreated between epochs, saving the overhead of repeatedly forking
    # forking is the process of creating a new subprocess; killing old workers and creating new ones each epoch wastes time

    # Note: shuffle=True is not passed here because DistributedSampler has already taken over the shuffling logic
    
    # Fix: dynamically compute the true maximum number of steps so the cosine decay curve perfectly tracks actual training progress
    # Why compute dynamically? Because len(loader) depends on total_data / (batch_size × world_size)
    # If you change the dataset or add/remove GPUs, the step count changes — hard-coding T_MAX_STEPS will eventually break things
    global T_MAX_STEPS
    # The global declaration makes T_MAX_STEPS not just local to the train function, but a global variable

    steps_per_epoch = len(loader)
    # len(loader) is a PyTorch-provided method: len(loader) = ⌊samples_per_gpu / batch_size⌋
    # This directly computes how many steps are in each training epoch;
    # each step performs: {forward pass, loss computation, backward pass, optimizer state update, parameter update}
    # on batch_size * seq_len tokens

    T_MAX_STEPS = steps_per_epoch * EPOCHS
    # Maximum steps = steps per epoch × number of epochs

    if is_master:
        # Only the first GPU prints the step count once
        print(f"Dynamic computation complete: steps per epoch {steps_per_epoch}, total steps T_MAX_STEPS = {T_MAX_STEPS}")

    # ==============================================================================
    # 2. Initialize the model: build the model using the "tiny" preset
    # ==============================================================================
    # ModelArgs.get_args("tiny", ...) is the configuration factory we defined when hand-building the model in Ep1-Ep18
    # The "tiny" preset corresponds to a set of smaller hyperparameters (layers, heads, hidden_dim),
    # suitable for experimenting within limited GPU memory
    args = ModelArgs.from_yaml(os.path.join(_BASE_DIR, "../config_zh.yaml"))

    # .to(device): move all model parameters to the GPU bound to the current process
    model = Transformer(args).to(device)

    # ---- Resume / checkpoint mechanism ----
    # Training large models can take days or weeks; power cuts, OOM, and accidental Ctrl+C are all common
    # So we need a "save/load" system: periodically save model weights + optimizer state + training progress
    # On next launch, if a checkpoint file is found, automatically resume from the breakpoint instead of starting over
    start_epoch = 0      # Default: start from epoch 0
    global_step = 0      # Global step counter, accumulated across epochs, used for LR scheduling and logging
    RESUME_FILE = "model_latest.pth"  # Checkpoint filename; every save overwrites this file

    # Prepare an empty basket to temporarily hold the optimizer state
    temp_opt_state = None 

    if os.path.exists(RESUME_FILE):
        if is_master:
            print(f"Attempting to resume training from {RESUME_FILE}...")
        # map_location=device: load directly onto the current GPU, preventing all processes from loading onto GPU 0 first and then migrating
        checkpoint = torch.load(RESUME_FILE, map_location=device)
        
        # Support two checkpoint formats:
        #   New format (dict): contains model_state_dict, optimizer_state_dict, epoch, global_step
        #   Old format (raw weights): only has model.state_dict(), without optimizer state or progress info
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # isinstance checks whether checkpoint is a dict
            model.load_state_dict(checkpoint['model_state_dict'])
            # Load the model state dict from the checkpoint dictionary
            temp_opt_state = checkpoint['optimizer_state_dict']
            # Load the optimizer state dict from the checkpoint dictionary
            start_epoch = checkpoint['epoch']
            # Load the epoch number from the checkpoint dictionary
            global_step = checkpoint['global_step']
            # Load the global step count from the checkpoint dictionary (epoch-agnostic global total)
        else:
            model.load_state_dict(checkpoint)
            # If not a dict, load the weights directly; this is for backward compatibility with my old checkpoint format and can be removed
        
        if is_master:
            print(f"✅ Resume successful! Continuing from Epoch {start_epoch+1}, Step {global_step}")

    # ---- torch.compile: compilation acceleration in PyTorch 2.0 ----
    # How it works: "compiles" the Python-written model into optimized CUDA kernels,
    # reducing the overhead of the Python interpreter
    # Analogy: you wrote an article in Chinese; compile is like translating it into machine code for direct execution,
    # bypassing the word-by-word translation process
    model = torch.compile(model, fullgraph=False)

    # ---- DDP (DistributedDataParallel) wrapping: gives the model multi-GPU sync capability ----
    # This step does three things:
    #   1. Registers a "gradient hook" on every parameter of the model
    #   2. Whenever a parameter's gradient is computed, the hook automatically triggers AllReduce,
    #      exchanging gradients with the other GPUs
    #      AllReduce essentially divides each gradient value by the total number of GPUs
    #   3. Communication and computation overlap, so you barely notice the communication overhead
    # device_ids=[local_rank]: tells DDP which GPU this process's model lives on
    model = DistributedDataParallel(model, device_ids=[local_rank]) if use_ddp else model
    
    # ==============================================================================
    # 3. Define the optimizer and loss function
    # ==============================================================================
    # AdamW: the standard optimizer for large model training today (Ep13)
    # Compared to SGD: has built-in momentum and adaptive learning rate, converges faster and more stably
    # Compared to Adam: W = Weight Decay, which decouples L2 regularization from the gradient,
    # preventing the adaptive learning rate from interfering with the regularization effect
    # lr=LR_MAX: initial learning rate, will be dynamically overwritten by get_lr()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX)

    # Pour the previously stashed optimizer state back in
    # Why does the optimizer state matter? Because Adam maintains first-order momentum (mean) and
    # second-order momentum (variance) for every parameter.
    # Without restoring it, it's like telling the optimizer "I know nothing about these parameters";
    # it will wander for hundreds of steps before stabilizing, as if training from scratch
    if temp_opt_state is not None:
        optimizer.load_state_dict(temp_opt_state)
        if is_master:
            print("✅ Optimizer momentum state has also been perfectly restored!")

    # Count total trainable parameters to verify the model scale matches expectations
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # .numel() returns the number of elements in a tensor; e.g. a p with shape (1024, 512) has p.numel() = 1024 * 512 = 524288
    # Only parameters that require gradients are counted, as only they will be updated

    if is_master:
        print(f"My model now has {total_params} parameters waiting to be trained!")

    # CrossEntropyLoss: the standard loss function for classification tasks (Ep13)
    # Essence: measures the "distance" between the model's predicted probability distribution and the ground truth
    # For language models: each position is a vocab_size-class classification problem
    # ("which token in the vocabulary comes next?")
    criterion = torch.nn.CrossEntropyLoss()

    # ==============================================================================
    # 4. Start the training loop
    # ==============================================================================
    if is_master:
        print(f"Starting training, device: {device}, total GPUs: {world_size}...")
        print(f"Note: torch.compile requires 1~2 minutes to compile on the first step, please be patient...")
    model.train()

    SAVE_INTERVAL = 1000  # Save weights every 1000 steps
    LOG_INTERVAL = 100    # Print loss every 100 steps
    saved_checkpoints = []  # Tracks saved checkpoint filenames for rolling cleanup (keep only the latest 3)

    # ---- Mixed Precision Training ----
    # Core idea: use half precision (bf16/fp16) for forward pass and part of backward pass to save VRAM and speed up
    # But keep gradient updates and loss computation in float32 for numerical stability
    # bf16 vs fp16: (bf16 is now almost universally adopted; fp16 is essentially a legacy concern)
    #   bf16 (bfloat16): exponent bits are as wide as float32 (8 bits), less prone to overflow;
    #                    natively supported on Google TPU and NVIDIA A100+
    #   fp16 (float16): higher precision but narrow exponent (5 bits); large values easily become inf,
    #                   requiring GradScaler for dynamic scaling
    
    # Use a try-except-finally structure to ensure that even if training crashes or is interrupted by Ctrl+C,
    # the NCCL process group is properly released, preventing zombie processes that hold GPU memory
    try:
        for epoch in range(start_epoch, EPOCHS):
            # ---- Important: must call sampler.set_epoch(epoch) at the start of every epoch ----
            # Without this, DistributedSampler uses the same random seed to shuffle every epoch
            # Result: every GPU sees its data in the exact same order every round → the model learns "order" instead of "content"
            sampler.set_epoch(epoch)
            # total_loss accumulates the loss across the entire epoch; we compute the average at the end
            # Note: we're accumulating a GPU tensor here, so there's no GPU→CPU sync happening every step
            total_loss = 0
            for X, Y in loader:
            # This triggers the __iter__ magic method of the loader instance
            # DataLoader's __iter__ internally: each iteration fetches a batch of idx from the sampler,
            # calls dataset[idx] to retrieve data, stacks them into a (batch_size, seq_len) batch,
            # then hands you the (X, Y) tuple.
            
                # ---- Complete flow of one training step ----
                # 1. Update LR → 2. Move data to GPU → 3. Zero gradients → 4. Forward pass
                # → 5. Compute loss → 6. Backward pass → 7. Gradient clipping → 8. Update parameters
                global_step += 1

                # Step 1: retrieve the learning rate from our hand-written scheduler for this step and inject it into the optimizer
                # Why not set it directly in the optimizer? Because AdamW only accepts a fixed lr at initialization time
                # We need to manually overwrite the lr value in param_groups every step
                current_lr = get_lr(global_step)
                for param_group in optimizer.param_groups:
                    # The optimizer supports per-group learning rates; optimizer.param_groups has embedding, Transformer, head, etc.
                    param_group['lr'] = current_lr

                # Step 2: move data from CPU (pin_memory) to GPU
                # non_blocking=True: asynchronous transfer; execution continues without waiting for the transfer to finish
                # Works best with pin_memory: CPU page-locked memory → GPU VRAM via DMA direct path, no CPU involvement
                X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)

                # Step 3: zero out gradients left over from the previous step, because PyTorch accumulates gradients by default
                # set_to_none=True: instead of filling gradients with 0, set .grad to None directly
                # Faster than .zero_grad() because it skips the "write zeros" VRAM operation
                # Trade-off: if your code later checks `if param.grad is not None`, be aware of this
                optimizer.zero_grad(set_to_none=True)
                
                # Step 4: forward pass (inside the mixed precision context)
                # torch.amp.autocast: automatically downcasts certain operators (matmul, linear layers, etc.) to half precision
                # Note: not all operators are downcast; PyTorch maintains an internal "whitelist/blacklist"
                # e.g. matmul is downcast to bf16, but softmax and layer_norm stay in fp32
                autocast_device = "cuda" if "cuda" in device else "cpu"
                use_bf16 = _train_cfg["use_bf16"]
                with torch.amp.autocast(autocast_device, enabled=("cuda" in device), dtype=torch.bfloat16 if use_bf16 else torch.float16):
                    logits, aux_loss = model(X)
                    # logits: [batch_size, seq_len, vocab_size] — the model's predicted probability distribution over the vocabulary at each position
                    # aux_loss: MoE auxiliary loss (load balancing loss), preventing all tokens from routing to the same expert
                    # DDP: model.module is the original Transformer; the DDP wrapper does not have a get_aux_loss method
                    # aux_loss = model.module.get_aux_loss()
                    
                    ## main_loss = criterion(logits.view(-1, args.vocab_size), Y.view(-1))
                    ## loss = main_loss + 0.01 * aux_loss

                # .float() is shorthand for .to(torch.float32)
                logits = logits.float()

                main_loss = criterion(logits.view(-1, args.vocab_size), Y.view(-1))
                # .view(-1, args.vocab_size): flatten [batch, seq_len, vocab] into [batch*seq_len, vocab]
                # .view(-1): flatten [batch, seq_len] into [batch*seq_len]
                # This allows CrossEntropyLoss to compute loss per token
                # Total loss = main loss + 0.01 × MoE auxiliary loss

                loss = main_loss + _train_cfg["aux_loss_weight"] * aux_loss.float()
                # Weight read from YAML: too large will make the model focus on "balancing expert load" and neglect language modeling
                # Too small renders the MoE load balancing ineffective, starving certain experts
                # float(aux_loss): ensures aux_loss is also cast to fp32 before the computation

                # --- Steps 5-7: backward pass + gradient clipping + parameter update ---
                loss.backward()
                # .backward() propagates gradients backward via the chain rule, computing ∂Loss/∂W for every parameter

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # Gradient Clipping: constrains the global L2 norm of all parameter gradients to at most 1.0
                # L2 norm is simply the sum of squares of every gradient element — a generalized distance measure
                # If the gradient norm exceeds 1.0, all gradients are scaled down proportionally,
                # preventing the model from taking an overly large step and crashing training
                # g_new = g_old / (L2)
                # This is especially important in the early stages of Transformer training,
                # where attention gradients can suddenly spike

                optimizer.step()
                # The optimizer updates the parameters; for each parameter Wt:
                # Wt+1 = Wt - η(mt/(√vt + ε) - λWt)

                total_loss += loss.detach()
                # detach(): severs the connection between loss and the computation graph,
                # preventing the accumulated total_loss from being included in backpropagation
                
                # ---- Logging: print every LOG_INTERVAL steps ----
                if global_step % LOG_INTERVAL == 0 and is_master:
                    current_lr = optimizer.param_groups[0]['lr'] 
                    # ⚠️ Only at the exact moment we're about to print a log entry (e.g. every 100 steps), call .item() to transfer data from GPU to CPU
                    # The cost of syncing once every 100 steps is negligible, but calling .item() every step would noticeably slow down training
                    current_loss_val = loss.item()
                    print(f"Step {global_step} | Epoch {epoch+1} | Loss: {current_loss_val:.4f} | LR: {current_lr:.2e}")

                # ---- Checkpoint saving: save every SAVE_INTERVAL steps ----
                if global_step % SAVE_INTERVAL == 0:
                    
                    # Only rank 0 writes files, to avoid multiple processes writing to the same file simultaneously and corrupting it
                    if is_master:
                        # ---- Unwrap the original model weights from the DDP + compile "Russian nesting doll" ----
                        # model                          → DistributedDataParallel wrapper
                        # model.module                   → OptimizedModule generated by torch.compile
                        # model.module._orig_mod         → our hand-built original Transformer
                        # If compile was not used, model.module is directly the Transformer
                        # Why peel all the way to the innermost layer? Because when saving/loading we only want clean weight key names
                        # Key names with DDP prefix (module.xxx) or compile prefix are hard to match up on restoration
                        raw_model = model.module._orig_mod if hasattr(model.module, "_orig_mod") else model.module
                        checkpoint_data = {
                            'epoch': epoch,               # Current epoch number; resume will continue from the next epoch
                            'global_step': global_step,   # Global step count; the LR scheduler can pick up from here on resume
                            'model_state_dict': raw_model.state_dict(),     # Model weights
                            'optimizer_state_dict': optimizer.state_dict(), # Optimizer state (Adam momentum)
                        }
                        ckpt_name = f"model_step_{global_step}.pth"
                        # Checkpoint name: model_step_<steps>.pth
                        torch.save(checkpoint_data, ckpt_name)
                        # Save as a dictionary
                        saved_checkpoints.append(ckpt_name)

                        # Also overwrite model_latest.pth so the next launch can blindly load the latest checkpoint
                        torch.save(checkpoint_data, "model_latest.pth")
                        print(f"💾 Step {global_step} progress (including optimizer state) saved: {ckpt_name}")

                        # Rolling cleanup: keep only the latest 3 checkpoints to prevent disk from filling up
                        # Large model checkpoints can be several GB each; over 15 epochs you can accumulate dozens
                        if len(saved_checkpoints) > 3:
                            oldest_ckpt = saved_checkpoints.pop(0) 
                            # saved_checkpoints acts as a queue; once it exceeds 3, the leftmost element is popped each time
                            if os.path.exists(oldest_ckpt):
                                os.remove(oldest_ckpt) 
                                print(f"Cleaned up old checkpoint: {oldest_ckpt}")
                    if use_ddp:
                        dist.barrier()
                        # This is a synchronization barrier; all GPUs must stop here and wait for Rank 0 (master) to finish writing to disk.
                        # Otherwise, other GPUs have already rushed ahead to compute the next batch,
                        # which can cause the DataLoader to fall out of sync or even trigger a VRAM OOM.

            # ---- End of epoch: print the average loss for the entire epoch ----
            if is_master:
                # .item() is only called here — once per epoch — so the overhead is negligible
                avg_loss = (total_loss / len(loader)).item()
                # .item() converts the tensor to a plain Python float
                print(f"--- Epoch {epoch+1} complete, average loss: {avg_loss:.4f} ---")

    except KeyboardInterrupt:
        # ---- Emergency save: safety net when Ctrl+C interrupts training ----
        # Noticed the loss looks wrong halfway through and want to stop and tune? No problem —
        # pressing Ctrl+C won't lose your progress
        if is_master:
            print("\nTraining interruption detected, performing emergency save...")
            raw_model = model.module._orig_mod if hasattr(model.module, "_orig_mod") else model.module
            checkpoint_data = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': raw_model.state_dict(), # Save clean weights
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint_data, "model_interrupted.pth")
            torch.save(checkpoint_data, "model_latest.pth") 
            print("✅ State has been safely saved.")
    
    finally:
        if use_ddp:
            dist.destroy_process_group()
            # .destroy_process_group() releases the communication resources of the process group
            # This line executes no matter what, ensuring the NCCL communication matrix between GPUs is freed.
            # Without this, the next run will definitely report "Address already in use".

# ==============================================================================
# Entry point
# ==============================================================================
if __name__ == "__main__":
    # This script cannot be run directly with `python train.py`!
    # It must be launched with torchrun, which automatically:
    #   1. Spawns world_size processes (one per GPU)
    #   2. Sets environment variables like LOCAL_RANK and WORLD_SIZE
    #   3. Establishes inter-process communication
    # Example launch command (3 GPUs):
    #   torchrun --nproc_per_node=3 train.py
    train()
