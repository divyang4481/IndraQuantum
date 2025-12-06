"""
IndraQuantum: A Quantum-Inspired Language Model
A quantum-inspired LM optimized for low VRAM using complex-valued embeddings and tensor networks.
"""

__version__ = "0.1.0"

from indra.models.quantum_core import IndraQuantum
from indra.graph.phase_shift import PhaseShift

__all__ = ["IndraQuantum", "DenseComplexLinear", "PhaseShift"]
