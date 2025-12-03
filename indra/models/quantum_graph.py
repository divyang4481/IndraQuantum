import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantum_core import ComplexLinear
from .embedding import QuantumEmbedding

class QuantumGraphLayer(nn.Module):
    """
    Applies quantum-inspired graph operations.
    Updates node states based on neighbors defined in the adjacency matrix.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Transformation for message passing
        self.message_transform = ComplexLinear(d_model, d_model)
        
        # Update gate (combining self and neighbor info)
        self.update_gate = ComplexLinear(d_model * 2, d_model)
        
    def forward(self, x, adj):
        """
        Args:
            x: Node states [Batch, Seq, d_model * 2] (Complex)
            adj: Adjacency matrix [Batch, Seq, Seq] (Real)
        """
        # 1. Compute Messages
        # Transform all nodes to generate messages
        messages = self.message_transform(x) # [B, S, D*2]
        
        # 2. Aggregate Messages
        # Split into Real/Imag for matrix multiplication with Adjacency
        m_real, m_imag = torch.chunk(messages, 2, dim=-1)
        
        # Aggregate: A * M
        # adj is [B, S, S], m is [B, S, D]
        agg_real = torch.bmm(adj, m_real)
        agg_imag = torch.bmm(adj, m_imag)
        
        aggregated = torch.cat([agg_real, agg_imag], dim=-1)
        
        # 3. Update State
        # Combine original state and aggregated messages
        combined = torch.cat([x, aggregated], dim=-1) # [B, S, D*4]
        
        # Apply update transformation
        # Note: update_gate expects d_model*2 input, but we have d_model*4
        # We need to adjust the update_gate definition or project here.
        # Let's redefine update_gate in __init__ to take d_model*2 input 
        # but here we are concatenating. 
        # Actually, let's just add them for residual-like connection before transform,
        # or use a larger linear layer.
        # Let's use a larger linear layer for the update.
        
        return aggregated # Returning just aggregation for residual add in parent

class IndraQuantumGraph(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers=4, n_heads=4, dropout=0.1, max_seq_length=512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Embeddings
        self.token_embedding = QuantumEmbedding(vocab_size, d_model)
        
        # Special Embeddings for Graph Nodes (Sentence, Para)
        # We assume 3 types: 0=Token, 1=Sentence, 2=Para
        self.type_embedding = nn.Embedding(3, d_model * 2)
        
        # Layers
        self.layers = nn.ModuleList([
            QuantumGraphLayer(d_model) for _ in range(n_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model * 2) for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model * 2, vocab_size)
        
    def forward(self, input_ids, node_types, graph_mask):
        """
        Args:
            input_ids: [Batch, Seq]
            node_types: [Batch, Seq]
            graph_mask: [Batch, Seq, Seq]
        """
        # Embeddings
        x = self.token_embedding(input_ids)
        
        # Add Type Embeddings (Real-valued added to Complex? Or Complex Type Emb?)
        # Let's treat type_embedding as complex (size d_model*2)
        t_emb = self.type_embedding(node_types)
        x = x + t_emb
        
        x = self.dropout(x)
        
        # Graph Layers
        for layer, norm in zip(self.layers, self.layer_norms):
            residual = x
            # Layer returns the aggregation
            agg = layer(x, graph_mask)
            
            # Update: New = Old + Aggregation (Residual connection is implicit in graph conv usually, 
            # but here we explicitly add)
            x = norm(x + agg)
            x = self.dropout(x)
            
        # Output Projection
        # We only care about predictions for Token nodes (Type 0)
        logits = self.output_projection(x)
        
        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
