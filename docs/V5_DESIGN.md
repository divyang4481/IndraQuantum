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
