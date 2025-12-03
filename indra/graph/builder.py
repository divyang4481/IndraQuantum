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
        self.vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 32000
        self.SENT_TOKEN_ID = self.vocab_size
        self.PARA_TOKEN_ID = self.vocab_size + 1
        
    def build_graph(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Parses text and returns graph components using Appended Topology.
        Structure: [Tokens... | Sentences... | Paragraphs...]
        
        Returns:
            dict containing:
            - input_ids: Token IDs including special system tokens
            - attention_mask: Standard mask
            - graph_mask: Adjacency matrix with edge types
            - node_types: Tensor indicating node type (0=Token, 1=Sentence, 2=Para)
        """
        # 1. Tokenize full text first (keep tokens contiguous)
        # We need offsets to map sentences to tokens
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids_list = encoding['input_ids']
        offsets = encoding['offset_mapping']
        
        # Truncate tokens if needed (leave room for structure nodes? 
        # For now, let's just truncate tokens to some reasonable limit, e.g. max_len - 20)
        # But we need to be careful. Let's just truncate to max_len for now and handle overflow.
        if len(input_ids_list) > self.max_seq_len:
            input_ids_list = input_ids_list[:self.max_seq_len]
            offsets = offsets[:self.max_seq_len]
            
        num_tokens = len(input_ids_list)
        
        # Lists to build graph
        all_input_ids = list(input_ids_list)
        node_types = [0] * num_tokens # 0=Token
        
        # Edges: (Source, Target, Type)
        # Types: 1=Self, 2=Local(Token-Sent), 3=Global(Sent-Para)
        edges = []
        
        # 2. Split Structure
        paragraphs = [p for p in text.split('\n') if p.strip()]
        
        # We need to map character positions to token indices
        # offsets is list of (start, end)
        
        current_token_idx = 0
        
        # We will append structural nodes to all_input_ids
        
        for para_idx, para in enumerate(paragraphs):
            # Create Paragraph Node
            para_node_idx = len(all_input_ids)
            all_input_ids.append(self.PARA_TOKEN_ID)
            node_types.append(2) # Para
            
            # Find paragraph span in text to help with sentence finding?
            # Actually, we can just iterate sentences and find their tokens.
            
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sentences = [s for s in sentences if s.strip()]
            
            for sent in sentences:
                sent_node_idx = len(all_input_ids)
                all_input_ids.append(self.SENT_TOKEN_ID)
                node_types.append(1) # Sentence
                
                # Connect Sentence to Para (Global Edge)
                edges.append((sent_node_idx, para_node_idx, 3))
                edges.append((para_node_idx, sent_node_idx, 3))
                
                # Find tokens for this sentence
                # We assume sequential processing matches the tokenizer's output
                # This is a heuristic: we look for the sentence text in the original text
                # and map it to tokens.
                # Since we tokenized the FULL text, the tokens should be in order.
                
                # Simple Heuristic:
                # Advance current_token_idx until we find tokens that "look like" this sentence?
                # Or just map by character offsets if we knew the sentence offsets.
                # Let's find sentence offsets in the original text.
                
                # Note: This simple find might fail with repeated sentences. 
                # But since we process sequentially, we can keep a "text_cursor".
                
                # However, `text` passed to tokenizer might differ slightly from `para` splits if we stripped things.
                # Let's rely on the fact that `offsets` map to the `text` string.
                
                # We need to find where `sent` starts and ends in `text`.
                # But `text` is the full string.
                pass 
                
        # RE-IMPLEMENTATION WITH ROBUST OFFSET MAPPING
        # To do this correctly like NyGraphTiny, we need to track char cursor.
        
        # Reset
        all_input_ids = list(input_ids_list)
        node_types = [0] * num_tokens
        edges = []
        
        text_cursor = 0
        token_cursor = 0
        
        for para in paragraphs:
            # Create Para Node
            para_node_idx = len(all_input_ids)
            all_input_ids.append(self.PARA_TOKEN_ID)
            node_types.append(2)
            
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sentences = [s for s in sentences if s.strip()]
            
            for sent in sentences:
                # Find sent in text starting from text_cursor
                # We use `text` (the full string)
                start_char = text.find(sent, text_cursor)
                if start_char == -1:
                    continue # Should not happen
                end_char = start_char + len(sent)
                text_cursor = end_char
                
                # Create Sent Node
                sent_node_idx = len(all_input_ids)
                all_input_ids.append(self.SENT_TOKEN_ID)
                node_types.append(1)
                
                # Edge: Sent <-> Para
                edges.append((sent_node_idx, para_node_idx, 3))
                edges.append((para_node_idx, sent_node_idx, 3))
                
                # Map to Tokens
                # Find tokens that overlap with [start_char, end_char)
                
                # Advance token_cursor to start
                while token_cursor < num_tokens:
                    t_start, t_end = offsets[token_cursor]
                    if t_end <= start_char:
                        token_cursor += 1
                    else:
                        break
                
                sent_start_token = token_cursor
                
                # Find end token
                temp_cursor = token_cursor
                while temp_cursor < num_tokens:
                    t_start, t_end = offsets[temp_cursor]
                    if t_start < end_char:
                        temp_cursor += 1
                    else:
                        break
                
                sent_end_token = temp_cursor
                
                # Connect Sent <-> Tokens
                for i in range(sent_start_token, sent_end_token):
                    edges.append((i, sent_node_idx, 2))
                    edges.append((sent_node_idx, i, 2))
                    
        # Truncate total sequence if needed
        if len(all_input_ids) > self.max_seq_len:
            all_input_ids = all_input_ids[:self.max_seq_len]
            node_types = node_types[:self.max_seq_len]
            # Filter edges
            edges = [(u, v, t) for u, v, t in edges if u < self.max_seq_len and v < self.max_seq_len]
            
        seq_len = len(all_input_ids)
        
        # Build Tensors
        input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        node_types_tensor = torch.tensor(node_types, dtype=torch.long)
        
        # Build Adjacency Matrix
        adj = torch.zeros((seq_len, seq_len), dtype=torch.float)
        adj.fill_diagonal_(1.0)
        
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
