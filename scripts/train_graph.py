import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import sys
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indra.models.quantum_graph import IndraQuantumGraph
from indra.graph.builder import TextGraphBuilder
from scripts.setup_teacher import setup_teacher_model

class GraphCollator:
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.builder = TextGraphBuilder(tokenizer, max_seq_len=max_len)
        
    def __call__(self, batch):
        # Batch is list of dicts {'input_ids': ...}
        # We need to decode to text to rebuild graph structure
        # This is inefficient but necessary since we only have tokenized data prepared
        
        input_ids_list = [item['input_ids'] for item in batch]
        texts = self.tokenizer.batch_decode(input_ids_list, skip_special_tokens=True)
        
        batch_input_ids = []
        batch_node_types = []
        batch_masks = []
        batch_graph_masks = []
        
        max_len = 0
        
        processed_graphs = []
        
        for text in texts:
            g = self.builder.build_graph(text)
            processed_graphs.append(g)
            max_len = max(max_len, g['seq_len'])
            
        # Pad and Stack
        for g in processed_graphs:
            l = g['seq_len']
            # Pad Input IDs
            ids = torch.cat([g['input_ids'], torch.zeros(max_len - l, dtype=torch.long)])
            batch_input_ids.append(ids)
            
            # Pad Node Types
            types = torch.cat([g['node_types'], torch.zeros(max_len - l, dtype=torch.long)])
            batch_node_types.append(types)
            
            # Pad Graph Mask
            # Mask is [L, L] -> [Max, Max]
            mask = torch.zeros((max_len, max_len))
            mask[:l, :l] = g['graph_mask']
            batch_graph_masks.append(mask)
            
        return {
            'input_ids': torch.stack(batch_input_ids),
            'node_types': torch.stack(batch_node_types),
            'graph_mask': torch.stack(batch_graph_masks)
        }

def train_graph_model(epochs=1, batch_size=4, lr=3e-4):
    print("=== Training IndraQuantum Graph Model ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load Data
    with open("data/wikitext/train.pkl", "rb") as f:
        train_data = pickle.load(f)
    
    # Convert to list of dicts to avoid Dataset indexing issues
    # The error `KeyError: 2` or `0` likely comes from the Dataset object's __getitem__ behavior
    # when accessed by the DataLoader in a way it doesn't expect (e.g. list of indices).
    train_data = [train_data[i] for i in range(len(train_data))]
    
    # Subset for speed
    train_data = train_data[:200] 
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # DataLoader
    collator = GraphCollator(tokenizer, max_len=128)
    loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collator, shuffle=True)
    
    # Model
    # Vocab size needs to include special graph tokens (SENT, PARA)
    # TinyLlama has 32000. We added 2.
    vocab_size = 32002
    d_model = 128
    model = IndraQuantumGraph(vocab_size, d_model, config={'tt_rank': 4}).to(device)
    print(f"Model Parameters: {model.get_num_params():,}")
    
    # Teacher (for KD)
    print("Loading Teacher...")
    teacher, _ = setup_teacher_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    teacher = teacher.to(device)
    teacher.eval()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    history = {'loss': []}
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for i, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            node_types = batch['node_types'].to(device)
            graph_mask = batch['graph_mask'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            student_logits = model(input_ids, node_types, graph_mask)
            
            # Teacher Forward
            teacher_input_ids = input_ids.clone()
            teacher_input_ids[teacher_input_ids >= 32000] = 0
            
            with torch.no_grad():
                teacher_out = teacher(teacher_input_ids)
                teacher_logits = teacher_out.logits
                
            # CRITICAL FIX: Mask the logits to only consider Token nodes (node_types == 0)
            token_mask = (node_types == 0)
            
            student_logits_tokens = student_logits[token_mask][:, :32000]
            teacher_logits_tokens = teacher_logits[token_mask]
            
            T = 2.0
            loss = kl_loss(
                torch.nn.functional.log_softmax(student_logits_tokens / T, dim=-1),
                torch.nn.functional.softmax(teacher_logits_tokens / T, dim=-1)
            ) * (T * T)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Log every 10 steps
            if (i + 1) % 10 == 0:
                # You can add file logging here if needed, but tqdm handles console output
                pass
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        history['loss'].append(avg_loss)
        
    torch.save(model.state_dict(), "checkpoints/quantum_graph.pt")
    
    # Save History
    if not os.path.exists("logs"): os.makedirs("logs")
    with open("logs/history_graph.pkl", "wb") as f:
        pickle.dump(history, f)
        
    print("Training Complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    
    train_graph_model(epochs=args.epochs)
