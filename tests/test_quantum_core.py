
import unittest
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indra.models.quantum_core import IndraQuantum, ComplexLinear
from indra.models.embedding import QuantumEmbedding
from indra.graph.phase_shift import PhaseShift

class TestQuantumCore(unittest.TestCase):
    
    def setUp(self):
        self.vocab_size = 100
        self.d_model = 32
        self.batch_size = 2
        self.seq_len = 10
        
    def test_complex_linear(self):
        print("\nTesting ComplexLinear Layer...")
        layer = ComplexLinear(self.d_model, self.d_model)
        # Input shape: (batch, seq, d_model * 2)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model * 2)
        output = layer(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model * 2))
        print("ComplexLinear shape check passed.")
        
    def test_quantum_embedding(self):
        print("\nTesting QuantumEmbedding...")
        emb = QuantumEmbedding(self.vocab_size, self.d_model)
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        output = emb(input_ids)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model * 2))
        print("QuantumEmbedding shape check passed.")
        
        # Check if initialization is not just zeros or standard normal
        # (Hard to strictly verify polar init without accessing private logic, 
        # but we can check it's non-zero)
        self.assertTrue(torch.any(output != 0))

    def test_indra_quantum_forward(self):
        print("\nTesting IndraQuantum Model Forward Pass...")
        model = IndraQuantum(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=2,
            n_heads=2,
            max_seq_length=20
        )
        
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        logits = model(input_ids)
        
        # Output should be (batch, seq, vocab_size)
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))
        print("IndraQuantum forward pass shape check passed.")

    def test_phase_shift(self):
        print("\nTesting PhaseShift Layer...")
        layer = PhaseShift(self.d_model, num_phases=4, use_graph=False)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model * 2)
        output = layer(x)
        
        self.assertEqual(output.shape, x.shape)
        print("PhaseShift shape check passed.")

if __name__ == '__main__':
    unittest.main()
