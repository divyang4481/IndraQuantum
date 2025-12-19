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

### V8 Performance Analysis (Dec 15, 2025)

**Problem**: Training loss stuck at floor ~6.6.

- Despite optimal hyperparameters (`PagedAdamW`, `CosineScheduler`, `FP32`), the model fails to significantly reduce CE loss below 6.6.
- This corresponds to a perplexity of ~$e^{6.6} \approx 735$, meaning the model is uncertain and cannot express high confidence.

**Root Cause Analysis**: The **Born Rule vs. Normalization Conflict**.

1.  **The Physics (Born Rule)**: The output probability is defined as $P(token) \propto |z|^2$. (In code: `logits = log(|z|^2)`).
2.  **The Conflict**: To produce a high-confidence prediction (e.g., $P > 0.99$ or logit $10.0$), the squared magnitude $|z|^2$ must be massive ($\approx e^{10} \approx 22,000$).
3.  **The Constraint**: The `ComplexRMSNorm` layer immediately preceding the final projection forces $|z| \approx 1.0$.
4.  **The Result**: To predict confident tokens, the final Linear layer (`self.head`) must learn weights with massive gain ($150\times$). Gradient descent struggles to push weights this far because gradients vanish for large output magnitudes in this log-square landscape ($d(\log x^2)/dx = 2/x$).

**Data Evidence**:

- Training Metrics (Dec 15): Consistent "Smooth CE Loss" floor at 6.6.
- Gradient flows show vanishing updates for the final head compared to internal layers.

**Solutions**:

### 1. Logit Scaling (The "System Temperature" Fix) - **Recommended**

We introduce a learnable scalar $\alpha$ (Inverse Temperature) to the output:
$$Logits = \alpha \cdot \log(|z|^2)$$
This implies a generalized probability rule $P \propto |z|^{2\alpha}$.

- **Why it works**: If the model struggles to grow the magnitude $|z|$ due to the $1/z$ gradient penalty, it can simply **increase $\alpha$** to sharpen predictions. This decouples "Confidence" from "Feature Magnitude".
- **Preserves Quantum Nature**: It maintains the Born Rule structure while accounting for system temperature.

### 2. Gaussian Probability (Removing the Log)

We change the output definition to $Logits = |z|^2$, implying $P \propto e^{|z|^2}$.

- **Why it works**: The derivative of $|z|^2$ is $2z$, meaning gradients **grow** with confidence. Training is extremely stable.
- **Trade-off**: This transitions the model from a "Quantum Born Machine" to a "Boltzmann Machine".

**Action Plan (Step 6600)**:
Implement **Solution 1 (Logit Scaling)**.

1.  Add `self.logit_scale = nn.Parameter(torch.tensor(1.0))` to `IndraV5`.
2.  Update forward pass: `logits = (torch.log(mag_sq + 1e-6) + self.final_bias) * self.logit_scale`.
