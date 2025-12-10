# IndraQuantum Phase 2: Design Analysis & Optimization Plan

## 1. Architecture Review & Potential Issues

The `IndraQuantumPhase2` model introduces a "Quantum-Inspired" architecture with several novel components. While theoretically grounded in using phase for context/uncertainty and magnitude for salience, there are practical risks.

### A. Complex Embedding & Phase Initialisation

- **Design:** Tokens are embedded into a complex vector space ($r e^{i\theta}$).
- **Risk:** If $\theta$ (phase) is initialized randomly $\in [-\pi, \pi]$, we introduce maximum phase noise immediately.
- **Impact:** The model starts in a state of "maximum uncertainty," potentially making early convergence harder. The optimizer has to "unlearn" this noise.
- **Mitigation:** Initialize phase close to 0 or with a much smaller variance.

### B. Phase-based RoPE (Positional Encoding)

- **Design:** Position is encoded by rotating the phase: $\theta_{pos} = \theta_{content} + \text{RoPE}(pos)$.
- **Risk:** Standard RoPE rotates pairs of _real_ dimensions. Here, we rotate the _complex phase_ directly. This is mathematically cleaner ($e^{i(\theta + \delta)}$) but relies on the network respecting the phase cyclic property.
- **Issue:** If the network learns to rely on "absolute phase values" (e.g. via unwrapping or linear approximations in real space), the cyclic nature might cause discontinuities. The `ComplexLayer` must be strictly rotation-equivariant for this to work perfectly.

### C. Hybrid Output Layer (The `Alpha` Parameter)

- **Design:** $Logits = \text{RealProjection}(H) + \alpha \cdot \text{MagnitudeProjection}(H)$.
- **Risk:** We previously identified $\alpha$ starting too high causing "Magnitude Collapse" (where the model creates "loud" but meaningless embeddings).
- **Fix:** We have frozen $\alpha=0.006$ (via softplus). This forces the model to learn meaning (Real part) first.
- **Remaining Issue:** If $\alpha$ never unfreezes or grows, we might never actually use the "Quantum Salience" channel. It might be dead weight.

### D. Computational Complexity

- **Factor:** Complex multiplication requires 4 real multiplications and 2 additions $((a+ib)(c+id) = (ac-bd) + i(ad+bc))$.
- **Impact:** Training is ~4x slower per parameter than a real-valued transformer.
- **Optimization:** We must ensure `d_model` in Quantum mode is "worth" 2-4x the `d_model` of a Real mode. (e.g., Is Quantum-128 better than Real-256?).

---

## 2. "Nano" Verification Strategy (Faster Training)

To quickly validate the architecture without waiting 9 hours, we will switch to a **"Nano Overfit" strategy**.

### Goal

Prove that the `IndraQuantum` architecture is _capable_ of learning English grammar and logic by forcing it to overfit a small, meaningful dataset.

### Specifications

1.  **Context Window:** Reduce to **128 tokens**. or **256**

    - _Why?_ 512/1024 is overkill for validating if a model _can_ learn. 128 is enough for "The cat sat on the mat" and simple Q&A.
    - _Speedup:_ Attention is $O(N^2)$. 128 vs 1024 is ~64x faster attention.

2.  **Model Layout (Nano):**

    - `d_model`: **128** (Keep 128 to test stability, but it's small).
    - `n_layers`: **2** (Fastest depth that still allows "deep" learning).
    - `n_heads`: **4**.
    - `vocab`: **32000** (Standard) or reduced. We will stick to standard to avoid tokenizer issues, but the data will only use a subset.

3.  **Dataset: Synthetic "Textbook"**

    - Instead of loading WikiText, we will generate ~1000 examples of valid English and Logical patterns in memory.
    - Examples:
      - "Q: What is the opposite of hot? A: The opposite of hot is cold."
      - "The sun rises in the east and sets in the west."
    - _Why?_ We know the _exact_ ground truth. If the model fails, it's the architecture, not the noisy web data.

4.  **Curriculum Acceleration**
    - We will compress the curriculum into **100 Epochs** (which will take minutes on Nano).
    - Epoch 0-20: CE Only (Learn English).
    - Epoch 20-50: Add KD (Learn Teacher Logic).
    - Epoch 50+: Add Quantum (Phase/Mag).

---

## 3. Proposal for Immediate Action

I propose we create a script `scripts/verify_architecture_nano.py` that:

1.  Generates synthetic training data (high quality, repetitive).
2.  Initializes a **Nano IndraQuantum** (L=2, D=128).
3.  Trains for 100 epochs.
4.  Prints generation output _every 10 epochs_.

**Success Criteria:**

- By Epoch 100, the model must perfectly answer the synthetic questions.
- If it works -> Architecture determines "Valid". We scale up.
- If it fails -> Architecture has a bug (likely Phase noise or Gradient flow).
