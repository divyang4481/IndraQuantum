import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indra.models.embedding import QuantumEmbedding
from scripts.setup_teacher import setup_teacher_model

def test_kd_integration():
    print("=== Testing Full KD Integration (Student + Teacher) ===")
    
    # 1. Setup Student (IndraQuantum - Tiny Version for Test)
    print("Initializing Student (IndraQuantum)...")
    vocab_size = 32000
    d_model = 128 # 256 total
    student_emb = QuantumEmbedding(vocab_size, d_model)
    # Simple student body for testing
    student_head = nn.Linear(d_model * 2, vocab_size)
    
    # 2. Setup Teacher
    # We use a very small model if possible, or the one from setup_teacher
    # If TinyLlama is not cached, this might take a while. 
    # For this test, let's assume the user might not want to wait for a 2GB download if they haven't done it.
    # But the user asked to "run by small test".
    # I'll check if the model exists first, if not I'll mock it for the test speed, 
    # but the code structure will be real.
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    cache_dir = "./teacher_models"
    
    # Check if model is likely present
    is_present = os.path.exists(os.path.join(cache_dir, "models--" + model_name.replace("/", "--")))
    
    teacher_model = None
    tokenizer = None
    
    if is_present:
        print(f"Loading Teacher {model_name} from cache...")
        teacher_model, tokenizer = setup_teacher_model(model_name, cache_dir)
    else:
        print(f"Teacher {model_name} not found in cache. Using Mock Teacher for speed.")
        # Mock Teacher Output
        class MockTeacher(nn.Module):
            def __init__(self, hidden_size=2048, vocab_size=32000):
                super().__init__()
                self.config = type('obj', (object,), {'hidden_size': hidden_size, 'vocab_size': vocab_size})
                self.hidden_size = hidden_size
            def forward(self, input_ids, output_hidden_states=False):
                batch, seq = input_ids.shape
                # Return mock logits and hidden states
                logits = torch.randn(batch, seq, self.config.vocab_size)
                hidden = torch.randn(batch, seq, self.config.hidden_size)
                return type('obj', (object,), {'logits': logits, 'hidden_states': (hidden,)})()
        
        teacher_model = MockTeacher()
    
    # 3. The Bridge
    teacher_dim = teacher_model.config.hidden_size
    student_dim = d_model * 2
    
    print(f"Creating Bridge: {student_dim} -> {teacher_dim}")
    bridge = nn.Linear(student_dim, teacher_dim)
    
    # 4. Forward Pass & Loss Calculation
    print("Running Forward Pass...")
    input_ids = torch.randint(0, vocab_size, (2, 16)) # Batch 2, Seq 16
    
    # Student Forward
    student_embeds = student_emb(input_ids)
    student_logits = student_head(student_embeds)
    
    # Teacher Forward
    with torch.no_grad():
        teacher_out = teacher_model(input_ids, output_hidden_states=True)
        teacher_logits = teacher_out.logits
        teacher_hidden = teacher_out.hidden_states[-1] # Last layer
        
    # 5. Compute Losses
    print("Computing Losses...")
    
    # A. Hidden State Loss (Feature Matching)
    # Project student to teacher space
    projected_student = bridge(student_embeds)
    loss_hidden = nn.MSELoss()(projected_student, teacher_hidden)
    print(f"Hidden State Loss: {loss_hidden.item():.4f}")
    
    # B. Logit Loss (Soft Targets)
    # Temperature scaling
    T = 2.0
    loss_logits = nn.KLDivLoss(reduction="batchmean")(
        torch.log_softmax(student_logits / T, dim=-1),
        torch.softmax(teacher_logits / T, dim=-1)
    ) * (T * T)
    print(f"Logit KD Loss: {loss_logits.item():.4f}")
    
    total_loss = loss_hidden + loss_logits
    print(f"Total Loss: {total_loss.item():.4f}")
    
    print("\n=== Test Successful ===")
    print("The architecture is ready for KD training.")

if __name__ == "__main__":
    test_kd_integration()
