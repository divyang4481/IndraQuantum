# Indra V5 Agent: Architecture & Quantum-Inspired Design

## 1. Overview

Indra V5 Agent is a specialized Small Language Model (SLM) designed to inherit reasoning capabilities from large open-source models (Teacher: Qwen2.5-0.5B-Instruct) while utilizing a **Complex-Valued Neural Network (CVNN)** architecture.

The core hypothesis is that **Complex Numbers** allows for parameter-efficient encoding of semantic relationships ("Phase") independent of feature prominence ("Magnitude"), acting as a compression mechanism for intelligence.

---

## 2. Quantum-Inspired Theoretical Framework

While not a quantum simulator, Indra V5 leverages key mathematical properties from Quantum Mechanics to improve information processing:

### 2.1 The Born Rule (Measurement)

- **Concept**: In QM, probability is the squared magnitude of the wave function: $P = |\psi|^2$.
- **Implementation**: Logic interpretation in the final layer uses `mag_sq = real^2 + imag^2` to convert complex states into real-valued logits.
- **Benefit**: Decouples the "internal representation" (Reflections/Rotations) from the "observable probability".

### 2.2 Symplectic Coupling (Phase Coherence)

- **Concept**: Unitary evolution preserves information by rotating states without collapsing them.
- **Implementation**: `QuantumFeedForward` applies a learnable **Phase Rotation** ($e^{i\theta}$) at each layer.
- **Benefit**: Prevents the network from collapsing into two independent Real networks ($Re$ and $Im$). It forces the "Phase" channel to carry signal, effectively doubling valid information capacity per parameter.

### 2.3 Interference (Semantic Cancellation)

- **Concept**: Waves can cancel or amplify each other.
- **Implementation**: Complex Matrix Multiplication in `ComplexAttention` allows latent features to destructively interfere (erase ambiguity) or constructively interfere (reinforce certainty).
- **Benefit**: More robust handling of conflicting context compared to simple ReLU subtraction.

---

## 3. Architecture Specification

### 3.1 Model Configuration (Agent V1)

| Component                    | Value   | Notes                                   |
| :--------------------------- | :------ | :-------------------------------------- |
| **Layers**                   | 8       | Compressed Depth                        |
| **Hidden Dim ($d_{model}$)** | 512     | Complex (512 Real + 512 Imag)           |
| **Heads**                    | 8       | Complex Multi-Head Attention            |
| **Vocab Size**               | 151,936 | Aligned with Qwen-2.5                   |
| **Context Window**           | 256     | Optimized for 6GB VRAM (Local Training) |
| **Params**                   | ~200M   | High Density Information                |

### 3.2 Distillation Strategy

We teach the efficient Complex Student to mimic the Large Real Teacher.

- **Teacher**: `Qwen/Qwen2.5-0.5B-Instruct` (4-bit Quantized).
- **Loss Function**:
  - **Cross Entropy (CE)**: Ground Truth Next-Token prediction (Grammar/Facts).
  - **Kullback-Leibler (KD)**: Distribution Matching (Reasoning/Logic).
  - _Note_: Phase Loss is currently disabled as the Teacher has no phase info; the Student _learns_ its own optimal phase organization to satisfy the Magnitude requirements.

### 3.3 The "Causal" Fix

- **Issue**: Early versions lacked causal masking, leading to "future peeking".
- **Solution**: Strict Upper-Triangular (`-inf`) masking in `IndraV5.forward` ensures autoregressive integrity.

---

## 4. Performance Dynamics

- **Phase 1 (Structure)**: Rapid drop in CE Loss (~16.0 $\to$ 7.0). Model learns punctuation, stopwords, and token bonding.
- **Phase 2 (Logic)**: Slow decay of KD Loss. The Student attempts to map the Teacher's FP32 logic manifold onto its Complex manifold.
- **Convergence**: Expected around Step 10k-15k for basic reasoning.

## 5. Future Roadmap

1.  **Phase-Aware Teachers**: Distill from a larger _Complex_ model (if available) to guide phase alignment directly.
2.  **Rotary Embeddings (RoPE)**: Adapt Complex RoPE for better long-context handling (>256).
3.  **Agentic Fine-Tuning**: Post-distillation training on Tool-Use datasets (Function Calling).
