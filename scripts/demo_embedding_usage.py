
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indra.models.embedding import QuantumEmbedding

def demo_creation_and_training():
    print("=== 1. Creating and Training Quantum Embeddings ===")
    
    # 1. Creation
    vocab_size = 1000
    d_model = 64 # This means 64 Real + 64 Imaginary = 128 total size
    
    print(f"Creating QuantumEmbedding(vocab={vocab_size}, d_model={d_model})")
    # This initializes with Polar Coordinates (Magnitude ~1, Phase random)
    embedding = QuantumEmbedding(vocab_size, d_model)
    
    # 2. Dummy Data
    input_ids = torch.tensor([[1, 5, 10, 20], [4, 2, 8, 9]]) # Batch size 2, Seq len 4
    print(f"Input shape: {input_ids.shape}")
    
    # 3. Forward Pass
    # Output will be [Batch, Seq, d_model * 2] -> [2, 4, 128]
    complex_vectors = embedding(input_ids)
    print(f"Output shape (Complex): {complex_vectors.shape}")
    
    # 4. Training (Standard PyTorch loop)
    print("\n--- Training Step Simulation ---")
    optimizer = optim.Adam(embedding.parameters(), lr=0.01)
    
    # Dummy loss: Let's say we want the magnitude of vectors to be close to 1.5
    # Magnitude^2 = Real^2 + Imag^2
    real, imag = torch.chunk(complex_vectors, 2, dim=-1)
    magnitude = torch.sqrt(real**2 + imag**2)
    loss = torch.mean((magnitude - 1.5)**2)
    
    print(f"Initial Loss: {loss.item():.4f}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check if weights changed
    print("Optimizer step executed. Weights updated.")
    
    # Verify new loss
    new_complex = embedding(input_ids)
    r_new, i_new = torch.chunk(new_complex, 2, dim=-1)
    new_mag = torch.sqrt(r_new**2 + i_new**2)
    new_loss = torch.mean((new_mag - 1.5)**2)
    print(f"New Loss: {new_loss.item():.4f} (Should be lower)")


class QuantumToClassicalBridge(nn.Module):
    """
    Adapter to make Quantum Embeddings compatible with Classical Models (like BERT/Llama)
    for Knowledge Distillation.
    """
    def __init__(self, quantum_dim, classical_dim):
        super().__init__()
        # quantum_dim is usually d_model * 2 (Real + Imag)
        self.projection = nn.Linear(quantum_dim * 2, classical_dim)
        
    def forward(self, quantum_states):
        # quantum_states: [Batch, Seq, d_model * 2]
        return self.projection(quantum_states)

def demo_kd_compatibility():
    print("\n=== 2. Compatibility with Classical Models (KD) ===")
    
    # Scenario: 
    # Teacher: TinyLlama (Classical, Hidden Size = 2048)
    # Student: IndraQuantum (Quantum, d_model = 128 -> Total 256)
    
    teacher_dim = 2048
    student_quantum_dim = 128 # 128 Real + 128 Imag = 256 total
    
    print(f"Teacher Dim: {teacher_dim} (Real)")
    print(f"Student Dim: {student_quantum_dim} (Complex -> {student_quantum_dim*2} Real representation)")
    
    # The Student Output
    batch_size = 2
    seq_len = 10
    student_output = torch.randn(batch_size, seq_len, student_quantum_dim * 2)
    
    # The Teacher Output (Target)
    teacher_output = torch.randn(batch_size, seq_len, teacher_dim)
    
    # PROBLEM: Dimensions don't match, can't compute MSE Loss directly.
    # SOLUTION: Use a Bridge (Projection Layer)
    
    bridge = nn.Linear(student_quantum_dim * 2, teacher_dim)
    print("Created Bridge Layer: Linear(256 -> 2048)")
    
    # Project Student -> Teacher Space
    projected_student = bridge(student_output)
    
    # Now we can compute KD Loss (Hidden State Matching)
    kd_loss = nn.MSELoss()(projected_student, teacher_output)
    print(f"KD Loss (Hidden State Matching): {kd_loss.item():.4f}")
    
    print("\n--- Note on Logits KD ---")
    print("For standard KD (distilling output probabilities), no bridge is needed.")
    print("Both Teacher and Student output logits of size [Batch, Vocab_Size].")
    print("You simply use KLDivLoss(Student_Logits, Teacher_Logits).")

if __name__ == "__main__":
    demo_creation_and_training()
    demo_kd_compatibility()
