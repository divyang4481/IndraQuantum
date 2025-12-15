# Indra Distillation Agent Training - System Verified

# Date: 2025-12-14

This document outlines the verified configuration and code changes that enabled stable, memory-efficient training of the Indra Agent V1.

## 1. Verified Configuration (`training/config_agent_v1.yaml`)

- **Model:** IndraV5 (206M Params), Tied Embeddings.
- **Context:** `max_seq_len: 256` (Critical for speed).
- **Training:**
  - `batch_size: 1`
  - `accumulate_grad_batches: 32` (Effective Batch: 32)
  - `use_8bit_optimizer: true` (Huge VRAM saver)
  - `gradient_checkpointing: false` (Speed > Memory since we fit)
- **Distillation:** Teacher `Qwen/Qwen2.5-0.5B-Instruct`.

## 2. Key Code Changes

### A. Automatic Mixed Precision (AMP)

Enabled `torch.amp.autocast('cuda')` and `GradScaler` in `scripts/train_distill_agent.py`. This runs the model in FP16/BF16, cutting VRAM usage by ~50% and boosting speed on Tensor Cores.

### B. Complex Attention Fix (`indra/modules/complex_attention.py`)

PyTorch CUDA cannot perform matrix multiplication on `ComplexHalf` (FP16 Complex) tensors. We implemented a manual upcast to `complex64` (FP32) for the two critical matmuls:

1. **Scores:** `Q @ K*` -> Cast to `complex64` -> MatMul -> Result `complex64`.
2. **Weighted Sum:** `Attn @ V` -> Cast to `complex64` -> MatMul -> Cast back to `query.dtype`.

### C. Correct Masking (Detailed)

We implemented a robust masking strategy to ensure the model _only_ learns from valid data, effectively ignoring "Padding" tokens (used to fill batches to a fixed length).

**1. The Problem**
Standard Causal masking only hides _future_ tokens. Without a dedicated Padding Mask, the model treats "PAD" tokens as valid context. This "pollutes" the attention mechanism with meaningless noise and forces the model to try and predict random padding tokens, wasting capacity.

**2. The Fix: Model-Side (Attention)**
Updated `IndraV5.forward` to accept an `attention_mask` (0 for Pad, 1 for Real).

- **Mechanism:** Additive Masking.
- **Logic:** We convert the binary mask to an additive one (`1.0 -> 0.0`, `0.0 -> -inf`).
- **Combination:** `Final_Mask = Causal_Mask + Padding_Mask`.
- **Result:** If a token is _either_ in the Future _or_ a Padding token, its attention score becomes `-inf`. The Softmax then turns this into `0.0`, ensuring NO information flows from that token.

**3. The Fix: Loss-Side (Distillation)**
Updated `DistillationLoss.forward` to accept the same `attention_mask`.

- **Mechanism:** Element-wise masking.
- **Logic:** We calculate the Cross-Entropy and KL-Divergence loss for _every_ token. Then, we multiply this loss vector by the mask.
- **Result:** Loss from padding tokens becomes `0.0`. We then average only over the number of _valid_ tokens. This ensures the model is graded solely on its performance on real language.

### D. Memory Optimization

- Explicit `torch.cuda.empty_cache()` calls after loading heavy components (Teacher) and periodically during training.
- 4-bit quantization for the Teacher model.

## 3. How to Run

```bash
python scripts/train_distill_agent.py --config training/config_agent_v1.yaml
```

## 4. Expected Performance

- **Throughput:** ~12-13 samples/sec (depending on GPU).
- **VRAM:** Fits comfortably on 6GB-8GB cards.
- **Stability:** No NaNs or OOMs.
