# IndraQuantum V5: Holographic & Symplectic Architecture

## Overview

IndraQuantum V5 is a strict Complex-Valued Neural Network designed to prevent "Complex Collapse" (where the model ignores the imaginary component).

## Core Innovations

### 1. Holographic Loss

**Location:** `indra/losses/holographic.py`
We minimize the **Circular Variance** (Mean Resultant Length $R$) of the phase distribution.

- $R \approx 1$: Phases are clustered (Collapsed).
- $R \approx 0$: Phases are uniform (Holographic/Diverse).
- The loss forces the model to utilize the entire phase space.

### 2. Symplectic Coupling (Quantum FFN)

**Location:** `indra/modules/complex_ffn.py`
To strictly prevent the network from treating real/imag parts as independent channels, we introduce a **Forced Layer Rotation**.

- $z_{out} = FFN(z) \cdot e^{i \theta}$
- $\theta$ is a learnable parameter initialized to non-zero.
- The rotation couples the real and imaginary components, making it impossible for the gradient to flow through only real paths effectively without acknowledging the complex nature.

### 3. Fully Complex Transformer

**Location:** `indra/models/indra_v5.py`

- **Embedding**: Two-stream embedding (Real/Imag) combined into Complex.
- **Attention**: $Attention(Q, K, V) = Softmax(Re(QK^* / \sqrt{d}))V$
- **Dropout**: Complex-aware dropout (`indra/modules/complex_dropout.py`) that masks the entire complex number to preserve phase coherence of remaining elements (or valid absence).

## Project Structure

- `indra/`: Core Library
- `training/`: `train.py` and `config_v5.yaml`
- `utils/`: Logging and Infrastructure
- `runs/`: Logs and Checkpoints (Created at runtime)

## V5 Training Analysis (Nano Experiment)

**Date**: Dec 11, 2025
**Model**: IndraV5-Nano (18M Params, 6 Layers)
**Dataset**: TinyStories

### Findings

1.  **Complex Collapse & Repetition**:

    - Initial runs with steady `rho_phase=0.5` or `1.0` caused severe "repetition loops" (e.g., "Alice Alice Alice").
    - **Cause**: Strong phase penalty early in training forces the model to minimize circular variance before it even learns basic grammar (Cross Entropy). It finds a trivial solution (repeating 1 token) to satisfy both.

2.  **The "Born Rule" Measurement**:

    - Switching from `abs(logits)` to `log(|z|^2 + eps)` + Bias stabilized the magnitude dynamics.
    - However, it still requires careful initialization to avoid NaNs at step 0.

3.  **Phase Annealing is Critical**:

    - Disabling `rho_phase=0.0` immediately allowed the model to specific learn English grammar and varied vocabulary.
    - **Conclusion**: Phase structure (Logic) cannot be forced before Grammar (Content).
    - **Design Update**: V6 or updated V5 training **MUST** use a "Phase Schedule":
      - `Step 0-2000`: `rho_phase = 0.0` (Learn English).
      - `Step 2000+`: `rho_phase -> 0.5` (Learn Logic/Structure).

4.  **Hardware Bottlenecks**:
    - Complex Models (4x FLOPs) on Windows/Laptop require strict VRAM management. Use `batch_size=2` and high `accumulation` to avoid Shared Memory thrashing.

### V6 Upgrade (Dec 11, 2025)

**Problem**: V5 "Bag of Words" Failure.

- Despite valid loss reduction, V5 checkpoints generated "word soup" (e.g., "girl she mom said" in random order).
- **Cause**: Additive Complex Positional Embeddings were too weak or not properly coupled in the complex plane, making the Attention mechanism permutation-invariant.

**Solution**: **Complex Sinusoidal Encoding (RoPE-Style)**.

- Replaced learned additive embedding with a **Multiplicative Fixed Phase Rotation**.
- $z_{pos} = z_{content} \cdot e^{i \theta_{pos}}$
- This strictly enforces order because every position rotates the token's phase by a unique, non-learnable angle.
- This aligns perfectly with the "Phase = Geometry" philosophy of IndraQuantum.

**Outcome (Step 9000)**: **FAILURE (Regression)**.

- Model regressed to "was was was" loops.
- **Diagnosis**: Applying RoPE rotation globally to the residual stream (`z`) rotates the `Value` vectors and `FFN` inputs. This destroys the semantic stability required for the FFN to learn content features. RoPE must only be applied to `Query` and `Key` within Attention.

### V7 Plan (Refined V5)

**Correction**: Return to **Additive Positional Embeddings** (V5) but increase their signal strength.

- The "No Phase Loss" V5 run showed emerging grammar, implying Additive was partially working.
- **Fix**: Scale `pos_embedding` by `sqrt(d_model)` or `10.0` to ensure positional information isn't drowned out by token embeddings.
- **Config**: Keep `rho_phase=0.0` (Annealing) and `batch_size=2`.

### V7.1 Final Tuning (Dec 11, 2025)

**Problem**: V7 "Strong Pos" (Scale 10.0) resulted in stuck loss (~11.3).

- **Simulation**: `debug_gradients.py` revealed `Token/Pos Ratio = 0.10`.
- The Position signal was 10x stronger than Token signal, forcing the model to ignore content ("Bag of Positions").

**Solution**: **Balanced Additive Embeddings (Scale 1.0)**.

- Reverting scaling back to 1.0 yielded `Token/Pos Ratio = 1.07` (Perfect Balance).
- **Conclusion**: The optimal architecture is **IndraV5 Backbone + Additive Complex Positions + Phase Annealing**.
- **Action**: Train for 20k steps to allow long-term coherence to emerge naturally.
