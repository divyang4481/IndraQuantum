# Loss Strategy: "Quantum Native" vs "Hard Staging"

## Current Strategy (Hard Staging)

- **Epoch 0-3**: Ignore Phase/Teacher. Just predict next token.
- **Epoch 3-6**: Listen to Teacher.
- **Epoch 6-10**: Optimize Phase/Mag structure.

**Flaw:** The model learns a "non-quantum" solution in Epoch 0-3 (collapsing phase to noise). Later, it has to _unlearn_ this to satisfy the Phase loss in Epoch 6.

## Proposed Strategy (Continuous Guidance)

We should provide all signals from the start, but scale them so the "Task" (CE) is dominant but the "Structure" (Quantum) is always respected.

```python
# loss_v2.py (Proposed Update)

# KD: Always help, but ramp up slightly?
# Actually, KD is most useful EARLY when student is clueless.
kd_weight = 0.5 * (1.0 - progress) # High KD early, Low KD later (Let student become independent)

# Quantum Aux: Constant pressure to maintain structure.
aux_weight = 0.1 # Always keep the latent space well-formed.
```

## Why this is better

1. **No "Shock"**: The model doesn't hit a wall at Epoch 3 or 6 where the rules change.
2. **Structural Integrity**: Phase alignment ($e^{i\theta}$) is learnt as the _mechanism_ for prediction, not added as a constraint later.
