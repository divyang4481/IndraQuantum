# IndraQuantum V5

**Advanced Quantized-Linguistic Model Architecture**

## Status: V5 (Final) / Stable

IndraQuantum V5 is a Complex-Valued Neural Network (CVNN) designed to capture "Quantum Linguistic States" (Position = Phase, Meaning = Magnitude).

## Key Features

- **Holographic Loss**: Minimizes Circular Variance to prevent phase collapse.
- **Quantum FFN**: Uses Symplectic Coupling (Forced Rotation) to bind real and imaginary components.
- **Cloud Native**: Ready for Azure/AWS/GCP training.

## Directory Structure

```
IndraQuantum/
├── data/                     # Data loading
├── indra/                    # Core Library (Modules, Models, Losses)
├── training/                 # Training scripts & configs
├── utils/                    # Loggers & Cloud IO
├── scripts/                  # Tools (HF Conversion, Testing)
├── runs/                     # Experiment artifacts
└── docs/                     # Documentation
```

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Verify Install

```bash
python scripts/test_model.py
```

### 3. Train (Local)

```bash
python training/train.py --config training/config_v5.yaml
```

### 4. Convert to HuggingFace

```bash
python scripts/convert_to_hf.py --checkpoint runs/<id>/final_model.pt --config training/config_v5.yaml --output_dir hf_model
```

## Documentation

- [Math & Theory](docs/MATH_SOLUTIONS.md)
- [V5 Design Spec](docs/V5_DESIGN.md)

## License

MIT
