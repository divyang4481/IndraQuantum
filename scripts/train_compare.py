import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import sys
import time
import argparse
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indra.models.quantum_core import IndraQuantum
from indra.models.baseline import StandardBaseline
from scripts.setup_teacher import setup_teacher_model

class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long)
        }

def train_model(
    model_type="quantum",
    data_path="data/wikitext",
    epochs=3,
    batch_size=8,
    lr=3e-4,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"=== Starting Training: {model_type.upper()} Model ===")
    
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("CUDA is NOT available. Using CPU.")
        device = "cpu"
        
    print(f"Device: {device}")
    
    # 1. Load Data
    print("Loading data...")
    try:
        with open(os.path.join(data_path, "train.pkl"), "rb") as f:
            train_data = pickle.load(f)
        # Use a subset for quick testing if needed
        # train_data = train_data[:1000] 
        train_dataset = SimpleDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print(f"Loaded {len(train_data)} training samples.")
    except FileNotFoundError:
        print("Data not found! Please run scripts/prepare_data.py first.")
        return

    # 2. Initialize Models
    vocab_size = 32000 # TinyLlama vocab size approx
    d_model = 128      # Base dimension
    
    if model_type == "quantum":
        student = IndraQuantum(vocab_size, d_model).to(device)
    else:
        student = StandardBaseline(vocab_size, d_model).to(device)
        
    print(f"Student Parameters: {student.get_num_params():,}")
    
    # Teacher
    print("Loading Teacher...")
    teacher, _ = setup_teacher_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if teacher is None:
        print("Failed to load teacher. Exiting.")
        return
    teacher = teacher.to(device)
    teacher.eval()
    
    # Bridge (for Hidden State KD)
    # Student dim is d_model*2 (256), Teacher is 2048
    student_dim = d_model * 2
    teacher_dim = teacher.config.hidden_size
    bridge = nn.Linear(student_dim, teacher_dim).to(device)
    
    # Optimizer
    optimizer = optim.Adam(list(student.parameters()) + list(bridge.parameters()), lr=lr)
    
    # Losses
    mse_loss = nn.MSELoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    # Logging
    log_file = f"logs/train_{model_type}.txt"
    if not os.path.exists("logs"): os.makedirs("logs")
    
    history = {'loss': [], 'hidden_loss': [], 'logit_loss': []}
    
    # 3. Training Loop
    student.train()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        total_loss = 0
        pbar = tqdm(train_loader, desc="Training")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask'].to(device) # Not used in simple models yet
            
            optimizer.zero_grad()
            
            # Student Forward
            student_logits = student(input_ids)
            # We need student hidden states for KD. 
            # Current IndraQuantum forward returns logits. 
            # We might need to hack it or just use the embedding/pre-logit state.
            # For now, let's assume we can't easily get hidden states without modifying the model code.
            # I will modify the model code in a second step if needed, but for now let's rely on Logit KD 
            # or just assume we only do Logit KD if hidden is hard to get.
            
            # Actually, let's just use Logit KD for simplicity in this comparison script, 
            # OR we can modify the models to return hidden states.
            # Let's stick to Logit KD + Task Loss (Next Token Prediction) for now to be robust.
            
            # Teacher Forward
            with torch.no_grad():
                teacher_out = teacher(input_ids)
                teacher_logits = teacher_out.logits
            
            # KD Loss (Logits)
            T = 2.0
            loss_kd = kl_loss(
                torch.nn.functional.log_softmax(student_logits / T, dim=-1),
                torch.nn.functional.softmax(teacher_logits / T, dim=-1)
            ) * (T * T)
            
            # Task Loss (Optional, but good for stability)
            # loss_task = nn.CrossEntropyLoss()(student_logits.view(-1, vocab_size), input_ids.view(-1))
            
            # Total Loss
            loss = loss_kd # + 0.5 * loss_task
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        history['loss'].append(avg_loss)
        
        # Save Checkpoint
        torch.save(student.state_dict(), f"checkpoints/{model_type}_epoch_{epoch+1}.pt")
        
    # Save History
    with open(f"logs/history_{model_type}.pkl", "wb") as f:
        pickle.dump(history, f)
        
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="quantum", choices=["quantum", "baseline"])
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    
    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")
    
    train_model(model_type=args.model, epochs=args.epochs)
