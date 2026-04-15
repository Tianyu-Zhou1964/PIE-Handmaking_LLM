# ==============================================================================
# Inference Script: Getting the Trained LLM to "Speak"
# ==============================================================================
# During training we did: feed data → compute loss → backprop → update weights (Ep22)
# Inference is completely different: no loss, no gradients, no weight updates
# We do exactly one thing — the "text completion game" from Ep2:
# give the model a prompt and let it predict one token at a time until it stops
# ==============================================================================

import torch
import torch.nn.functional as F
import custom_bpe  # Import the compiled BPE engine hand-crafted in Rust (Ep21)
from model import Transformer, ModelArgs  # Import the hand-crafted model architecture (Ep1–Ep18 theory + Ep19 implementation)
import yaml
import os

# ---- Load inference config ----
# All hyperparameters (temperature, top_k, top_p, etc.) live in a YAML config file
# Benefit: change parameters without touching code — just edit the config file
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_BASE_DIR, "../config_en.yaml")

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)
    _inf_cfg = _cfg["inference"]  # Inference-related config (model path, sampling params, etc.)
    _model_cfg = _cfg["model"]    # Model architecture config (num layers, heads, dimensions, etc.)


# ==============================================================================
# 1. Sampling Algorithm: Top-K + Top-P Joint Filtering
# ==============================================================================

# ---- Background recap ----
# In Ep2 we covered: a language model's output is a probability distribution (after Softmax)
# The simplest strategy is "greedy": always pick the token with the highest probability
# But greedy decoding produces rigid, repetitive text — like a broken record
#
# The idea behind Top-K + Top-P: instead of just taking the top-1, sample randomly
# from a pool of "reasonably plausible" candidates
# This keeps output coherent (low-probability garbage is filtered out) while preserving creativity

def top_k_top_p_filtering(logits, top_k=50, top_p=0.9, filter_value=-float('Inf')):
    """
    Filter logits (raw scores before Softmax) and keep only "plausible" candidate tokens.
    Logits are the unnormalized raw scores produced at the end of the forward pass,
    before being converted into probabilities.

    Args:
    - logits: 1-D tensor of shape (vocab_size,), one raw score per vocabulary token
    - top_k: keep only the top-K highest-scoring tokens (set the rest to -inf)
    - top_p: after top_k, accumulate probabilities from highest to lowest;
             once the cumulative sum exceeds p, discard everything beyond that point
    - filter_value: value assigned to discarded tokens (-inf → probability 0 after Softmax)

    Two-stage filter:
      Stage 1 (Top-K): coarse filter — keep only the top K candidates, eliminate the vast majority
      Stage 2 (Top-P): fine filter — within those K candidates, accumulate probabilities until
                       the threshold P is reached, then discard the tail "fillers"
    """
    assert logits.dim() == 1

    # ---- Stage 1: Top-K ----
    # torch.topk returns the K largest values and their indices
    # [0] gets the values; [-1] gets the K-th largest value (i.e., the threshold)
    # All logits below this threshold are set to -inf
    if top_k > 0:
        threshold = torch.topk(logits, top_k).values[-1]
        # threshold is the cutoff value; torch.topk(logits, top_k) returns a tuple (values, indices)
        # where values is the top-K scores sorted in descending order, e.g. [12.3, 10.1, 8.7, ..., 3.2]
        # We use index -1 to grab the last (smallest) value and set it as the threshold
        logits[logits < threshold] = filter_value
        # logits < threshold produces a boolean mask (True / False for each element)
        # logits[mask] then assigns filter_value to all positions where the mask is True
        # After this, only top_k logit scores remain; all others are set to -inf
        # (technically -11111111111111111111111111111111 in 32-bit float)

    # ---- Stage 2: Top-P (Nucleus Sampling) ----
    # Idea: sort remaining candidates by probability (high to low) and accumulate
    # Once the running total exceeds p (e.g. 0.9), cut everything after that point
    # This ensures we only sample from the "nucleus" that covers 90% of the probability mass
    if top_p > 0.0:
        # Sort by score in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # Compute cumulative probabilities over the sorted distribution
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Concrete example with 4 tokens:
        # Step 1: softmax probabilities (sorted) and their cumulative sum
        # sorted_probs     = [0.50, 0.30, 0.15, 0.05]
        # cumulative_probs = [0.50, 0.80, 0.95, 1.00]

        sorted_indices_to_remove = cumulative_probs > top_p
        # Step 2: mark positions where cumulative probability already exceeds top_p
        # sorted_probs             = [0.50, 0.30, 0.15, 0.05]
        # cumulative_probs         = [0.50, 0.80, 0.95, 1.00]
        # > 0.9?                   = [False, False, True, True]
        # sorted_indices_to_remove = [False, False, True, True]
        # Note: the token at index 2 (prob 0.15) is the one that pushes the cumulative sum
        # from 0.80 to 0.95, crossing the threshold — but we want to *keep* it,
        # because it's the final piece that completes 90% coverage.

        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        # Shift the entire mask one position to the right
        # (assign elements [1:end] the values of [0:end-1])
        sorted_indices_to_remove[0] = False
        # Set the first element to False explicitly since it was duplicated by the shift
        # [False, False, True, True] becomes:
        # [False, False, False, True]

        logits[sorted_indices[sorted_indices_to_remove]] = filter_value
        # sorted_indices           = [4, 2, 0, 1, 3]   # original indices of sorted positions
        # sorted_indices_to_remove = [F, F, T, T, T]   # which positions to remove
        # result indices to remove = [0, 1, 3]
        # logits[[0, 1, 3]] = filter_value

    return logits


# ==============================================================================
# 2. Initialization: Load Model + Rust BPE Engine
# ==============================================================================
# First step of inference: "wake up" the trained model from disk
# During training we saved a checkpoint (Ep22); now we pour the weights back
# into the model skeleton
# ==============================================================================


# ---- Device selection ----
# In "auto" mode: prefer NVIDIA GPU → fall back to Apple Silicon MPS → fall back to CPU
_device_setting = _inf_cfg["device"]
if _device_setting == "auto":
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
else:
    device = _device_setting

# ---- Build the model skeleton ----
# This only creates an "empty shell" — all parameters are randomly initialized
# Like the "blob of random-parameter clay" from the start of Ep22, not usable yet
args = ModelArgs.from_yaml(_CONFIG_PATH)
model = Transformer(args).to(device)

# ---- Load trained weights (restore from checkpoint) ----
# During training we saved a dictionary with torch.save (Ep22, block 5)
# containing model_state_dict (weights), optimizer_state_dict, etc.
# For inference we only need the model weights — no optimizer state needed since
# inference never updates parameters
CHECKPOINT_PATH = os.path.join(_BASE_DIR, _inf_cfg["checkpoint_path"])

try:
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    # map_location=device: remap tensors to the current device at load time
    # If the model was trained on GPU but you only have CPU, this handles the transfer automatically
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    # Support multiple checkpoint formats:
    #   Format 1: {"model": state_dict, ...}
    #   Format 2: {"model_state_dict": state_dict, ...}  (used by the Ep22 training script)
    #   Format 3: the state_dict directly (bare-bones save)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Pour the weights into the empty shell — from "random clay" to "sculpted form"
    model.load_state_dict(state_dict)

    print("✅ Weights loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load weights: {e}")
    exit()

# ---- Inference optimization ----
if _inf_cfg["use_fp16"]:
    model.half()
    # .half() is equivalent to .to(torch.float16) — compresses all parameters from 32-bit to 16-bit floats
    # During training we used bf16 mixed precision (Ep22); for inference we go full fp16
    # Benefits: halves memory usage, roughly doubles inference speed
    # Why can inference be more aggressive with low precision than training?
    # Because inference doesn't need gradients! Gradients are sensitive to precision,
    # but small numerical errors in the forward pass are perfectly tolerable

model.eval()
# .eval() mode: tells the model "this is exam time, not training time"
# It disables Dropout (randomly drops neurons during training to prevent overfitting;
# not needed at inference) and switches BatchNorm/LayerNorm to use global statistics
# instead of batch statistics


# ---- Load the Rust BPE unified engine ----
# ==============================================================================
# All of the following is handled inside the Rust BpeEngine:
#   - vocab.json is parsed by serde_json directly (nested formats handled automatically)
#   - merges.txt is processed line-by-line in Rust: split + vocab ID lookup (10× faster than Python)
#   - Special token detection is done in a single pass inside Rust
#   - Both encoder and decoder share the same parsed vocab, zero redundancy
# ==============================================================================
print("Loading high-performance Rust BPE engine...")
vocab_path = os.path.join(_BASE_DIR, _inf_cfg["vocab_path"])
# Path to the vocabulary file

merges_path = os.path.join(_BASE_DIR, _inf_cfg["merges_path"])
# Path to the merge rules file

engine = custom_bpe.BpeEngine(vocab_path, merges_path)
# Executing this line creates a BpeEngine instance; the terminal will also print three log lines:
# Rust BpeEngine: vocab loaded, XXXXX entries total
# Rust BpeEngine: merges loaded, XXXXX rules, XX special tokens
# Rust BpeEngine initialization complete!

# This object exposes five callable methods:
# engine.encode(text) → List[int]   string → token IDs, e.g. "To be or not" → [1234, 567, 89, ...]
# engine.decode(ids)  → str         token IDs → string, [1234, 567, 89, ...] → "To be or not"
# engine.decode_stream(token_id) → str   # decode one token at a time, designed for streaming generation.
# Internally maintains a byte buffer; for multi-byte characters (e.g. Chinese), returns "" until
# enough bytes have accumulated to form a complete character, then flushes all at once.
# engine.reset_stream() → None   clears the internal byte buffer of decode_stream.
# engine.token_to_id(token_str) → Optional[int]
# Used to look up special token IDs, e.g. engine.token_to_id("<eos>") returns the stop-token ID
# used in the generation loop to know when to halt.

# Fetch the <eos> (End of Sequence) token ID — generation stops as soon as the model emits it
eos_id = engine.token_to_id("<eos>")
if eos_id is None:
    raise ValueError("<eos> not found in vocabulary — check the special token spelling in vocab.json")

print(f"\n=============================================")
print(f"{_inf_cfg['model_name']} is online! (type 'quit' to exit)")
print(f"=============================================")


# ==============================================================================
# 3. Main Inference Loop: Engineering the Text Completion Game
# ==============================================================================
# Recall the pseudocode from Ep2:
#   input_text = "To be or not"
#   while not finished:
#       next_word = LLM.predict(input_text)
#       input_text = input_text + next_word
#
# What follows is the real implementation of that pseudocode.
# The difference: the real version must handle tokenization, sampling strategies,
# repetition penalties, streaming output, and other engineering details.
# ==============================================================================

while True:
    prompt = input("\nEnter a prompt (type 'quit' to exit): ")
    if prompt.lower() in ['quit', 'exit']:
        # .lower() converts uppercase letters in the prompt to lowercase, making the check case-insensitive
        print("👋 Goodbye!")
        break

    # ---- Encode: natural language → token ID sequence ----
    # engine.encode internally performs: regex pre-tokenization → byte encoding → BPE merges → special token matching
    input_ids = engine.encode(prompt)
    # Takes a Python string, e.g. "To be or not to be", passes it to Rust which splits and merges
    # according to the vocabulary and merge rules, then returns a list of integers
    # Output: a Python list of token IDs, e.g. [2847, 1156, 9634, 7789, 4815]

    x = torch.tensor([input_ids], device=device)
    # Wrap the token ID list into a PyTorch tensor and move it to the GPU
    # Shape: (1, seq_len) — batch_size=1 (inference processes one sample at a time)

    print("🥧 PIE continues: ", end="", flush=True)
    # end="" prevents print from adding a newline, so tokens appear on the same line
    # flush=True means: write immediately, don't wait for the buffer to fill

    engine.reset_stream()
    # Reset the Rust decoder's streaming byte buffer
    # Must be called at the start of every new generation to clear any leftover byte fragments
    # from the previous turn — zero allocation, zero overhead on the Rust side
    generated_ids = []
    # Track all generated token IDs

    # ---- Inference core: the text completion generation loop ----
    with torch.no_grad():
    # torch.no_grad(): tells PyTorch not to build a computation graph and not to track gradients
    # Inference never needs backpropagation (Ep13), so disabling the graph saves substantial
    # memory and compute. Without no_grad, PyTorch would faithfully store all intermediate
    # activations, waiting for a .backward() call that will never come —
    # turning it off dramatically reduces memory usage
        for step in range(_inf_cfg["max_new_tokens"]):
            # ---- Step 1: Forward pass ----
            # Feed the current token sequence to the model and get logits
            # logits shape: (1, seq_len, vocab_size)
            # logits are the model's raw scores for "what should the next token be" —
            # every token in the vocabulary gets one logit score
            outputs = model(x)
            if isinstance(outputs, tuple):
                logits = outputs[0]  # If the model returns a (logits, aux_loss) tuple, take the first element
            else:
                logits = outputs     # Otherwise use the output directly

            next_token_logits = logits[0, -1, :].clone()
            # Batch index 0, last sequence position — that's the logits distribution for the next token
            # Example: you input "I love hot pot", 4 tokens, output shape (bsz, 4, vocab_size):
            # logits[0, 0, :] → saw "I",             predicts position 1 ("love")
            # logits[0, 1, :] → saw "I love",        predicts position 2 ("hot")
            # logits[0, 2, :] → saw "I love hot",    predicts position 3 ("pot")
            # logits[0, 3, :] → saw "I love hot pot", predicts position 4 — the new unknown token
            # For instance it might output a period "." because the sentence is complete

            # So logits[0, -1, :] is the output at the last position; it has seen the entire input
            # and predicts the very first new token — exactly what we need during inference.
            # During training, logits at every position are useful (each position pairs with the
            # next token as a training sample). During generation, only the last position matters
            # because it's the only one predicting a truly unknown next token.

            # ---- Step 2: Repetition penalty ----
            # Without this, models easily fall into "broken record" mode:
            #   "I love you I love you I love you I love you..."
            # Penalty approach: look back at the last repetition_window tokens;
            # for any token that already appeared, subtract a penalty from its logit
            # This lowers its probability in the subsequent Softmax so it's less likely to be chosen again
            window = generated_ids[-_inf_cfg["repetition_window"]:]
            for token_id in set(window):
                next_token_logits[token_id] -= _inf_cfg["repetition_penalty"]

            # ---- Step 3: Top-K + Top-P filtering ----
            # Discard unreliable candidate tokens; keep only the core candidates
            next_token_logits = top_k_top_p_filtering(
                next_token_logits, top_k=_inf_cfg["top_k"], top_p=_inf_cfg["top_p"]
            )
            # After this step, we keep only the top-50 highest-probability candidates (Top-K)
            # and within those 50, only the ones whose cumulative probability reaches 0.9 (Top-P)
            # — often just 7–8 tokens in practice

            # ---- Step 4: Temperature scaling + Softmax → probability distribution ----
            probs = F.softmax(next_token_logits / _inf_cfg["temperature"], dim=-1)
            # probs is short for probabilities
            # Remember Softmax from Ep2? It converts raw scores into a valid probability distribution
            # (all values positive, sum to 1)
            # temperature controls "creativity":
            #   temperature < 1: sharper distribution, high-probability tokens dominate → more conservative output
            #   temperature = 1: unmodified distribution, no adjustment
            #   temperature > 1: flatter distribution, low-probability tokens get a chance → more "out-there" output
            # Mathematically: divide logits by temperature before Softmax:
            #   Softmax(logits / T)
            #   Higher T compresses the gaps between logits → flatter, more uniform distribution
            # Example with a 3-token vocabulary, logits = [10, 9, 1]:
            # Raw (T=1):    Softmax([10, 9, 1])    → [0.731, 0.269, 0.000]  # first token nearly wins alone
            # T=0.1:        Softmax([100, 90, 10]) → [≈1.0,  ≈0.0,  ≈0.0]  # even more extreme, winner-takes-all
            # Why does 90 almost vanish even though 100-90=10? Because e^10 ≈ 22026 —
            # the two values differ by over 20,000×, so yes, it really does become negligible.
            # T=10:         Softmax([1, 0.9, 0.1]) → [0.433, 0.391, 0.176]  # gaps shrink, tail tokens visible
            # e^1 ≈ 2.718, e^0.9 ≈ 2.460, e^0.1 ≈ 1.105 — close enough that Softmax spreads
            # probability more evenly — that's the high-temperature effect.
            # High temperature: generously called "creative thinking"; less generously, "going off the rails"
            # Use high temperature for poetry/brainstorming; use low temperature for math/code

            # ---- Step 5: Random sampling ----
            next_id = torch.multinomial(probs, num_samples=1).item()
            # torch.multinomial: draw one token at random according to the probability distribution
            # Note: this is NOT "pick the most probable" (that would be greedy); it's "sample by probability"
            # Higher-probability tokens are more likely to be drawn, but low-probability tokens still have a chance —
            # that's the source of diversity in generated text

            # ---- Step 6: Stopping criterion ----
            # If the model generates <eos> (End of Sequence), it's signaling "I'm done talking"
            if next_id == eos_id:
                break

            # ---- Step 7: Append ----
            x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
            generated_ids.append(next_id)
            # Append the newly generated token to the end of the input sequence
            # to serve as input for the next forward pass
            # This is "Output becomes the next Input" from Ep2
            # x grows by 1 each step: (1, n) → (1, n+1) → (1, n+2) → ...

            # ---- Step 8: Streaming decode and print ----
            text_chunk = engine.decode_stream(next_id)
            if text_chunk:
                print(text_chunk, end="", flush=True)
            # engine.decode_stream(token_id) → str   # decode one token at a time for streaming generation
            # Internally maintains a byte buffer; for multi-byte characters (e.g. Chinese), returns ""
            # until enough bytes accumulate to form a complete character, then flushes all at once.
        # Example trace for prompt "To be":
        # Iteration 1: generates "or",  input becomes "To be or"
        # Iteration 2: generates "not", input becomes "To be or not"
        # Iteration 3: generates "to",  input becomes "To be or not to"
        # Iteration 4: generates <eos>, model decides it's done — loop exits
        # This is the complete picture of the text-completion machine; the field calls it "autoregressive decoding"

    print("\n" + "-"*30)
    # Print a newline followed by a row of dashes to visually separate each conversation turn
