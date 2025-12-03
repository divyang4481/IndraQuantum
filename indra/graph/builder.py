import re
import torch
from typing import List, Tuple, Dict

class TextGraphBuilder:
    """
    Builds a hierarchical graph structure from text.
    Hierarchy: Token -> Sentence -> Paragraph
    
    This allows treating:
    - Tokens as Quantum Particles
    - Sentences as Local Quantum Systems
    - Paragraphs as Global Quantum Systems
    """
    
    def __init__(self, tokenizer, max_seq_len=512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def build_graph(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Parses text and returns graph components.
        
        Returns:
            dict containing:
            - input_ids: Token IDs including special system tokens
            - attention_mask: Standard mask
            - graph_mask: Adjacency matrix for quantum interactions
            - node_types: Tensor indicating node type (0=Token, 1=Sentence, 2=Para)
        """
        # 1. Split into Paragraphs
        paragraphs = [p for p in text.split('\n') if p.strip()]
        
        all_tokens = []
        node_types = [] # 0=Token, 1=Sentence, 2=Para
        
        # Edges: (Source, Target)
        # We will build an adjacency matrix later
        edges = []
        
        current_idx = 0
        
        # Global Paragraph Node (Root)
        # We can add a "Document" node, or just treat list of paragraphs.
        # Let's stick to Sentence/Para hierarchy.
        
        for para_idx, para in enumerate(paragraphs):
            # Create Paragraph Node
            para_node_idx = len(all_tokens)
            # We need a special token ID for Para/Sent. 
            # For now, let's use the tokenizer's UNK or a specific unused token if available.
            # Assuming tokenizer has a [SEP] or similar we can repurpose, or just 0.
            # Ideally we should extend vocabulary. For this demo, we use ID 1 (usually padding/start) as placeholder.
            all_tokens.append(self.tokenizer.eos_token_id or 1) 
            node_types.append(2) # Para
            
            # Split into Sentences
            # Simple regex split
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sentences = [s for s in sentences if s.strip()]
            
            for sent in sentences:
                sent_node_idx = len(all_tokens)
                all_tokens.append(self.tokenizer.eos_token_id or 1)
                node_types.append(1) # Sentence
                
                # Connect Sentence to Para
                edges.append((sent_node_idx, para_node_idx))
                edges.append((para_node_idx, sent_node_idx))
                
                # Tokenize Sentence
                tokens = self.tokenizer.encode(sent, add_special_tokens=False)
                
                for token in tokens:
                    token_idx = len(all_tokens)
                    all_tokens.append(token)
                    node_types.append(0) # Token
                    
                    # Connect Token to Sentence
                    edges.append((token_idx, sent_node_idx))
                    edges.append((sent_node_idx, token_idx))
                    
                    # Connect Token to Next Token (Sequential)
                    # if len(all_tokens) > 1 and node_types[-2] == 0:
                    #     prev_idx = len(all_tokens) - 2
                    #     edges.append((prev_idx, token_idx))
                    #     edges.append((token_idx, prev_idx))
        
        # Truncate or Pad
        if len(all_tokens) > self.max_seq_len:
            all_tokens = all_tokens[:self.max_seq_len]
            node_types = node_types[:self.max_seq_len]
            # Filter edges
            edges = [(u, v) for u, v in edges if u < self.max_seq_len and v < self.max_seq_len]
            
        seq_len = len(all_tokens)
        
        # Build Tensors
        input_ids = torch.tensor(all_tokens, dtype=torch.long)
        node_types_tensor = torch.tensor(node_types, dtype=torch.long)
        
        # Build Adjacency Matrix (Graph Mask)
        # 1 means connected, 0 means not
        adj = torch.zeros((seq_len, seq_len), dtype=torch.float)
        
        # Add Self Loops
        adj.fill_diagonal_(1.0)
        
        # Add Edges
        for u, v in edges:
            adj[u, v] = 1.0
            
        # Also allow Causal connections for tokens?
        # For Quantum Graph, we might want full visibility within the system.
        # Let's keep it defined by the explicit edges for now.
        
        return {
            "input_ids": input_ids,
            "node_types": node_types_tensor,
            "graph_mask": adj,
            "seq_len": seq_len
        }

def demo_graph_builder():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    builder = TextGraphBuilder(tokenizer)
    text = "Quantum mechanics is fascinating. It explains the behavior of particles. This is a new paragraph. It has another sentence."
    
    graph = builder.build_graph(text)
    
    print("Input IDs shape:", graph['input_ids'].shape)
    print("Node Types:", graph['node_types'])
    print("Graph Mask shape:", graph['graph_mask'].shape)
    
    # Visualize connections for first few nodes
    adj = graph['graph_mask']
    print("\nAdjacency (First 10x10):")
    print(adj[:10, :10])

if __name__ == "__main__":
    demo_graph_builder()
