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
        
        # Define special token IDs for graph nodes
        # We assume these are added to the end of the vocabulary
        self.vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 32000
        self.SENT_TOKEN_ID = self.vocab_size
        self.PARA_TOKEN_ID = self.vocab_size + 1
        
    def build_graph(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Parses text and returns graph components.
        
        Returns:
            dict containing:
            - input_ids: Token IDs including special system tokens
            - attention_mask: Standard mask
            - graph_mask: Adjacency matrix with edge types
            - node_types: Tensor indicating node type (0=Token, 1=Sentence, 2=Para)
        """
        # 1. Split into Paragraphs
        paragraphs = [p for p in text.split('\n') if p.strip()]
        
        all_tokens = []
        node_types = [] # 0=Token, 1=Sentence, 2=Para
        
        # Edges: (Source, Target, Type)
        # Types: 1=Self, 2=Local(Token-Sent), 3=Global(Sent-Para)
        edges = []
        
        current_idx = 0
        
        for para_idx, para in enumerate(paragraphs):
            # Create Paragraph Node
            para_node_idx = len(all_tokens)
            all_tokens.append(self.PARA_TOKEN_ID) 
            node_types.append(2) # Para
            
            # Split into Sentences
            # Simple regex split
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sentences = [s for s in sentences if s.strip()]
            
            for sent in sentences:
                sent_node_idx = len(all_tokens)
                all_tokens.append(self.SENT_TOKEN_ID)
                node_types.append(1) # Sentence
                
                # Connect Sentence to Para (Global Edge)
                edges.append((sent_node_idx, para_node_idx, 3))
                edges.append((para_node_idx, sent_node_idx, 3))
                
                # Tokenize Sentence
                tokens = self.tokenizer.encode(sent, add_special_tokens=False)
                
                for token in tokens:
                    token_idx = len(all_tokens)
                    all_tokens.append(token)
                    node_types.append(0) # Token
                    
                    # Connect Token to Sentence (Local Edge)
                    edges.append((token_idx, sent_node_idx, 2))
                    edges.append((sent_node_idx, token_idx, 2))
        
        # Truncate or Pad
        if len(all_tokens) > self.max_seq_len:
            all_tokens = all_tokens[:self.max_seq_len]
            node_types = node_types[:self.max_seq_len]
            # Filter edges
            edges = [(u, v, t) for u, v, t in edges if u < self.max_seq_len and v < self.max_seq_len]
            
        seq_len = len(all_tokens)
        
        # Build Tensors
        input_ids = torch.tensor(all_tokens, dtype=torch.long)
        node_types_tensor = torch.tensor(node_types, dtype=torch.long)
        
        # Build Adjacency Matrix (Graph Mask) with Edge Types
        # 0 = No Connection
        adj = torch.zeros((seq_len, seq_len), dtype=torch.float)
        
        # Add Self Loops (Type 1)
        adj.fill_diagonal_(1.0)
        
        # Add Edges
        for u, v, t in edges:
            adj[u, v] = float(t)
            
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
