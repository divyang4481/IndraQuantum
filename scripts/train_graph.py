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
    # Subset for speed
    train_data = train_data[:200] 
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # DataLoader
    collator = GraphCollator(tokenizer, max_len=128)
    loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collator, shuffle=True)
    
    # Model
    vocab_size = 32000
    d_model = 128
    model = IndraQuantumGraph(vocab_size, d_model).to(device)
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
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            node_types = batch['node_types'].to(device)
            graph_mask = batch['graph_mask'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            student_logits = model(input_ids, node_types, graph_mask)
            
            # Teacher Forward
            # Teacher expects standard input_ids. 
            # Our input_ids contain extra nodes (Sentence/Para).
            # We should filter them out for the teacher OR just pass them and ignore output?
            # Teacher won't understand Sentence/Para tokens (they are UNK or EOS).
            # Strategy: Extract only Token nodes (type 0) for Teacher.
            
            # Mask for tokens
            token_mask = (node_types == 0)
            
            # This is tricky because batch items have different token counts.
            # For simplicity in this demo: We pass the full sequence to Teacher.
            # The Teacher will treat Sentence/Para tokens as whatever ID we gave them (EOS).
            # This is "noisy" but allows dimension matching.
            
            with torch.no_grad():
                teacher_out = teacher(input_ids)
                teacher_logits = teacher_out.logits
                
            # Compute Loss only on Token nodes?
            # Or full sequence?
            # Let's do full sequence KD for simplicity, assuming Teacher's reaction to EOS 
            # is a valid target for Student's reaction to SentenceNode.
            
            T = 2.0
            loss = kl_loss(
                torch.nn.functional.log_softmax(student_logits / T, dim=-1),
                torch.nn.functional.softmax(teacher_logits / T, dim=-1)
            ) * (T * T)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
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
