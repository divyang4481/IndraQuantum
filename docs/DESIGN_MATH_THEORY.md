# IndraQuantum: Mathematical Foundations & Design Strategy (V3)

## 1. Core Philosophy: The Quantum Linguistic State

We hypothesize that language tokens can be modeled as **Quantum States** in a high-dimensional complex Hilbert space $\mathbb{C}^d$. This allows us to separate **semantic meaning** from **contextual dynamics**.

### The "Meaning-Context" Duality

For any hidden state vector $\psi \in \mathbb{C}^d$, we define:
$$ \psi = M \cdot e^{i\Phi} $$

1.  **Magnitude ($M \in \mathbb{R}^+$)**: Represents the **invariant semantic strength** or "salience" of features. (e.g., "King" and "Queen" share high magnitude in "Royalty" dimensions).
2.  **Phase ($\Phi \in [0, 2\pi)$)**: Represents the **syntactic context**, position, and relational structure. Phase interference allows words to bond or repel based on grammar and order.

---

## 2. Mathematical Architecture

### A. State Representation

The embedding layer maps integer token IDs to complex vectors:
$$ E(x) = M_x \cdot e^{i \Phi_x} $$

- **Initialization**: $M_x$ is strictly positive (ReLU/Softplus). $\Phi_x$ is initialized uniformly.

### B. Positional Encoding: Unitary Rotation (Quantum RoPE)

Position is not a vector added to the state; it is a **Transformation** of the state. We model position as a unitary rotation in the complex plane, preserving semantic magnitude while altering contextual phase.

For position $p$ and dimension $k$:
$$ \psi*{p,k}' = \psi*{p,k} \cdot e^{i \cdot p \cdot \theta_k} $$
This is mathematically equivalent to Rotary Positional Embeddings (RoPE), naturally fitting our complex architecture. Unlike standard RoPE which essentially "hacks" real vectors to behave like complex numbers, we **are** using complex numbers.

### C. Complex Linear Layer

To preserve the phase structure, we must use **True Complex Multiplication**.
For weight matrix $W = A + iB$ and input $z = x + iy$:
$$ W \cdot z = (Ax - By) + i(Ay + Bx) $$

- _Optimization_: We can implement this using 4 real matrix multiplications or efficient complex ops. This binds the Real/Imaginary parts so they evolve together, unlike independent linear layers.

### D. Complex Attention: Interference

Attention is the process of computing the "Overlap" (transition amplitude) between states.
$$ \text{Score}\_{q,k} = \text{Re}\left( \langle \psi*q, \psi_k \rangle \right) = \text{Re}\left( \psi_q \cdot \psi_k^* \right) $$
$$ \psi*q \cdot \psi_k^* = (M_q e^{i\Phi_q}) \cdot (M_k e^{-i\Phi_k}) = M_q M_k e^{i(\Phi_q - \Phi_k)} $$

- The term $e^{i(\Phi_q - \Phi_k)}$ captures the **relative phase** (relative position).
- If phases align (constructive interference), the score is high. If they represent opposite contexts (destructive interference), the score is low.

### E. Measurement (The Output Layer)

In Quantum Mechanics, we extract information via Measurement (Born Rule).
We compute the probability of the next token $v$ by projecting the final query state $\psi_{out}$ onto the token embeddings $E_v$.

$$ P(v) \propto |\langle \psi*{out}, E_v \rangle|^2 $$
Or for numerical stability in a Neural Network (Logits):
$$ \text{Logit}\_v = \text{Re}\left( \langle \psi*{out}, E_v \rangle \right) + \text{Bias}\_v $$
This "Cohere-and-Measure" approach ensures the model outputs predictions based on the alignment of the final quantum state with the vocabulary states.

---

## 3. Implementation Strategy (IndraTiny V3)

1.  **Strict Separation**: Re-implement `ComplexLinear` to enforce the algebra defined above.
2.  **Quantum RoPE**: Implement the proper rotation logic $z \cdot e^{i\theta}$.
3.  **Measurement Layer**: Implement the projection logic described above, discarding the ad-hoc "Hybrid Output".
4.  **Training**: Start from scratch on the full dataset with a cleaner scheduler.

This rigorous validation ensures "It's not just a Transformer with complex numbers; it's a Quantum-Inspired Architecture."
