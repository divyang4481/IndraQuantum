# IndraQuantum Phase 2: Quantum-Inspired Semantic Distillation

## 1. Architecture Overview (v2)

The `IndraQuantumPhase2` model introduces a novel "Quantum Projection" mechanism to enhance the representational capacity of small embedding spaces.

### Core Components

- **Embeddings:** Standard Token Embeddings (`vocab_size`=32000, `d_model`=128).
- **Quantum Projector:**
  - **Phase:** A linear layer measuring semantic "direction" or intent.
  - **Magnitude:** A linear layer measuring sematic "confidence" or salience.
  - **Operation:** $Z = M \cdot e^{iP}$ (Complex Number representation).
- **Backbone:** Standard Transformer Layers (Attention + FFN).
- **Output Head:** Recombines quantum states back into logits for token prediction.

## 2. Dataset Preparation

We utilize a "Mixed Strategy" to ensure the model learns both general language and instruction-following behavior simultaneously.

### Datasets Used

1.  **WikiText-103 (Subset):** ~22k samples.
    - _Purpose:_ General knowledge, grammar, vocabulary.
    - _Source:_ Hugging Face Datasets.
2.  **Stanford Alpaca:** ~52k samples.
    - _Purpose:_ Instruction following (Q&A), logic, reasoning structure.
    - _Source:_ `tatsu-lab/stanford_alpaca`.

### Preparation Pipeline

1.  **Download:** `curl` to fetch raw JSON.
2.  **Tokenization:**
    - Tokenizer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (32k vocab).
    - Format: `### Instruction: ... ### Input: ... ### Response: ...`
    - Max Length: 128 tokens.
3.  **Mixing:** Concatenation of WikiText + Alpaca -> `data/mixed_train.pkl` (~74k samples).

**Script:** `scripts/prepare_mixed_data.py`

## 3. Training Strategy (v4 Unified)

We have moved to a **Unified Training Strategy (v4)**. Instead of pre-training and then fine-tuning, we train the model from scratch using a combined objective function on the mixed dataset.

### v4 Unified Objective

- **Data:** Mixed (WikiText + Alpaca).
- **Loss Function (`loss_v2.py`):**
  - **CE Loss:** Prediction accuracy (The "Grammar" signal).
  - **KD Loss:** Teacher (TinyLlama) probability matching (The "Logic" signal).
    - _Weight:_ Gentle curriculum (0.01 -> 0.05).
  - **Phase/Mag Loss:** Quantum state alignment.
- **Benefit:** The model aligns its "Quantum States" (Phase/Mag) to the Teacher's "Entropy/Confidence" _while_ it learns the English language. This prevents "Catastrophic Forgetting" and ensures deeper semantic integration.

### Previous Attempts (Depreciated)

- _Stage 1:_ Pure Pre-training (Too simple).
- _Stage 2 (v2):_ Aggressive Distillation (Caused model collapse).
- _Stage 2 (v3):_ Gentle Distillation (Split training was inefficient).

## 4. Commands

### Data Preparation

```bash
python scripts/prepare_mixed_data.py
```

### v4 Unified Training (The Main Run)

```bash
python scripts/train_phase2_v4_unified.py
```

### Testing

```bash
python scripts/test_phase2.py checkpoints/phase2_v4_unified/checkpoint_v4_epoch_1.pt --prompt "The future of AI is"
```

## 5. Notebook Analysis

The `notebooks/training_monitor.ipynb` visualizes:

- **Total Loss:** The overall optimization progress.
- **CE vs KD:** The balance between "Predicting" (CE) and "Imitating" (KD).
- **Phase/Mag:** The stability of the Quantum Projector.

**To View Analysis:** Run all cells in `notebooks/training_monitor.ipynb`.
