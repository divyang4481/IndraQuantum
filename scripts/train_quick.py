import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import sys
import pickle
import logging
from tqdm import tqdm
from transformers import AutoTokenizer
import csv

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indra.models.quantum_core import IndraQuantum
from scripts.setup_teacher import setup_teacher_model

# Setup Logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename='logs/train_quick.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

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

def train_quick():
    # Configuration for Quick Test
    BATCH_SIZE = 8
    EPOCHS = 20
    LR = 1e-3
    TT_RANK = 32
    CHECKPOINT_DIR = "checkpoints/quick_test"
    DATA_PATH = "data/wikitext/train.pkl"
    SUBSET_SIZE = 2000 # Increase size
    
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # 1. Load Data
    logging.info("Loading data...")
    if not os.path.exists(DATA_PATH):
        logging.error(f"Data not found at {DATA_PATH}. Please run scripts/prepare_data.py.")
        return
        
    with open(DATA_PATH, "rb") as f:
        train_data = pickle.load(f)
    
    # Take subset using select to keep it as a Dataset object
    indices = range(min(len(train_data), SUBSET_SIZE))
    train_data = train_data.select(indices)
    logging.info(f"Using subset of {len(train_data)} samples for quick test.")
        
    dataset = SimpleDataset(train_data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Initialize Model
    logging.info(f"Initializing IndraQuantum (TT-Rank={TT_RANK})...")
    vocab_size = 32000 # Match TinyLlama
    d_model = 128
    
    config = {'tt_rank': TT_RANK, 'use_tt_embeddings': True}
    student = IndraQuantum(vocab_size, d_model, config=config).to(device)
    logging.info(f"Student Parameters: {student.get_num_params():,}")
    
    # 3. Load Teacher
    logging.info("Loading Teacher...")
    teacher, _ = setup_teacher_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    teacher = teacher.to(device)
    teacher.eval()
    
    # 4. Optimizer
    optimizer = optim.AdamW(student.parameters(), lr=LR)
    
    # 5. Training Loop
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    # CSV Logging
    metrics_file = 'logs/quick_metrics.csv'
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Batch', 'Total_Loss', 'Loss_Per_Token'])
    
    for epoch in range(EPOCHS):
        student.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            student_logits = student(input_ids)
            
            # Loss: Standard Next Token Prediction
            # Logits: [B, S, V] -> [B, S-1, V]
            # Labels: [B, S] -> [B, S-1] (Shifted)
            
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            
            loss.backward()
            optimizer.step()
            
            # Metrics
            loss_per_token = loss.item() # CrossEntropy is already per-token mean
            
            if batch_idx % 5 == 0:
                 with open(metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch+1, batch_idx, loss.item(), loss_per_token])
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.1f}", 'per_tok': f"{loss_per_token:.2f}"})
            
        avg_loss = total_loss / len(loader)
        logging.info(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, ckpt_path)
        logging.info(f"Saved checkpoint to {ckpt_path}")
        
    logging.info("Quick Test Complete.")

if __name__ == "__main__":
    train_quick()
