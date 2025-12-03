import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .quantum_core import ComplexLinear, complex_relu
from .embedding import QuantumEmbedding

class ComplexGraphAttentionLayer(nn.Module):
    """
    Complex-valued Graph Attention Layer with Edge Bias.
    
    Implements:
    1. Complex Attention: A = |Q^dag K|^2
    2. Edge Bias: Adds learned bias based on graph structure
    3. Complex FFN with CReLU
    """
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        # Complex Projections for Q, K, V
        self.q_proj = ComplexLinear(d_model, d_model)
        self.k_proj = ComplexLinear(d_model, d_model)
        self.v_proj = ComplexLinear(d_model, d_model)
        self.o_proj = ComplexLinear(d_model, d_model)
        
        # Edge Bias
        # We assume 3 edge types: 0 (None/Self), 1 (Local/Sentence), 2 (Global/Hierarchy)
        # Or maybe we use the graph_mask values directly?
        # The user mentioned "similar to _build_edge_bias_mask from NyGraphTiny".
        # And "Apply Additive Bias... based on edge type".
        # Let's assume the graph_mask passed in contains edge types (0, 1, 2...).
        # If graph_mask is just adjacency (0/1), we might need to update builder to send types.
        # For now, let's assume graph_mask is the edge type matrix.
        self.num_edge_types = 4 # None, Self, Local, Global
        self.edge_bias = nn.Embedding(self.num_edge_types, n_heads)
        
        # FFN
        self.ffn_1 = ComplexLinear(d_model, d_model * 4)
        self.ffn_2 = ComplexLinear(d_model * 4, d_model)
        
        self.norm1 = nn.LayerNorm(d_model * 2)
        self.norm2 = nn.LayerNorm(d_model * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, graph_mask):
        """
        Args:
            x: [Batch, Seq, d_model * 2] (Interleaved Real/Imag)
            graph_mask: [Batch, Seq, Seq] (Edge types)
        """
        batch_size, seq_len, _ = x.shape
        
        residual = x
        
        # 1. Attention
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for heads: [B, S, H, D_head * 2]
        # We need to handle the interleaved real/imag parts carefully.
        # x is [Re, Im] concatenated? No, ComplexLinear returns [Re, Im] concatenated.
        # So first half is Real, second half is Imag.
        
        def split_heads(tensor):
            # tensor: [B, S, D*2] -> Real: [B, S, D], Imag: [B, S, D]
            t_real, t_imag = torch.chunk(tensor, 2, dim=-1)
            # Reshape to [B, H, S, D_head]
            t_real = t_real.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            t_imag = t_imag.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            return t_real, t_imag
            
        q_r, q_i = split_heads(q) # [B, H, S, Dh]
        k_r, k_i = split_heads(k)
        v_r, v_i = split_heads(v)
        
        # Compute Attention Scores: |Q^dag K|^2
        # Real part of inner product: Q_r K_r^T + Q_i K_i^T
        # Imag part of inner product: Q_r K_i^T - Q_i K_r^T
        
        # Transpose K for matmul: [B, H, Dh, S]
        k_r_t = k_r.transpose(-2, -1)
        k_i_t = k_i.transpose(-2, -1)
        
        ac_real = torch.matmul(q_r, k_r_t) + torch.matmul(q_i, k_i_t)
        ac_imag = torch.matmul(q_r, k_i_t) - torch.matmul(q_i, k_r_t)
        
        # Magnitude squared
        attn_scores = ac_real**2 + ac_imag**2
        
        # Scale
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        
        # Add Edge Bias
        # graph_mask: [B, S, S] containing edge types
        # edge_bias: [NumTypes, H]
        # We want [B, H, S, S]
        
        # Ensure graph_mask is long for embedding lookup
        edge_types = graph_mask.long()
        bias = self.edge_bias(edge_types) # [B, S, S, H]
        bias = bias.permute(0, 3, 1, 2)   # [B, H, S, S]
        
        attn_scores = attn_scores + bias
        
        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply to V
        # Output is Real * V (since probabilities are real)
        # O_r = A * V_r
        # O_i = A * V_i
        
        o_r = torch.matmul(attn_probs, v_r)
        o_i = torch.matmul(attn_probs, v_i)
        
        # Recombine heads
        def combine_heads(t_r, t_i):
            # [B, H, S, Dh] -> [B, S, H, Dh] -> [B, S, D]
            t_r = t_r.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            t_i = t_i.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            return torch.cat([t_r, t_i], dim=-1)
            
        o = combine_heads(o_r, o_i)
        o = self.o_proj(o)
        
        # Residual + Norm
        x = self.norm1(residual + self.dropout(o))
        
        # 2. FFN
        residual = x
        
        # FFN 1
        x = self.ffn_1(x)
        
        # CReLU
        x_r, x_i = torch.chunk(x, 2, dim=-1)
        x_r, x_i = complex_relu(x_r, x_i)
        x = torch.cat([x_r, x_i], dim=-1)
        
        # FFN 2
        x = self.ffn_2(x)
        
        # Residual + Norm
        x = self.norm2(residual + self.dropout(x))
        
        return x

class IndraQuantumGraph(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers=4, n_heads=4, dropout=0.1, max_seq_length=512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Embeddings
        self.token_embedding = QuantumEmbedding(vocab_size, d_model)
        self.position_embedding = QuantumEmbedding(max_seq_length, d_model)
        
        # Special Embeddings for Graph Nodes (Sentence, Para)
        # We assume 3 types: 0=Token, 1=Sentence, 2=Para
        self.type_embedding = nn.Embedding(3, d_model * 2)
        
        # Layers
        self.layers = nn.ModuleList([
            ComplexGraphAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Linear(d_model * 2, vocab_size)
        
    def forward(self, input_ids, node_types, graph_mask):
        """
        Args:
            input_ids: [Batch, Seq]
            node_types: [Batch, Seq]
            graph_mask: [Batch, Seq, Seq] (Edge Types)
        """
        seq_len = input_ids.size(1)
        
        # Embeddings
        x = self.token_embedding(input_ids)
        
        # Position Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        x = x + pos_emb
        
        # Add Type Embeddings
        t_emb = self.type_embedding(node_types)
        x = x + t_emb
        
        # Graph Layers
        for layer in self.layers:
            x = layer(x, graph_mask)
            
        # Output Projection
        logits = self.output_projection(x)
        
        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
