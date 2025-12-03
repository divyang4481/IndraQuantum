"""
Quantum Core Module
Defines the IndraQuantum model and ComplexLinear layer using PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def complex_relu(real: torch.Tensor, imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Complex ReLU: ReLU(Real) + i * ReLU(Imag)
    
    Args:
        real: Real part of the tensor
        imag: Imaginary part of the tensor
        
    Returns:
        Tuple of (real_out, imag_out)
    """
    return F.relu(real), F.relu(imag)



class ComplexLinear(nn.Module):
    """
    Complex-valued linear layer for quantum-inspired transformations.
    
    Processes complex-valued inputs by splitting real and imaginary parts,
    applying separate transformations, and recombining them.
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If True, adds a learnable bias to the output
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Real and imaginary weight matrices
        self.weight_real = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(out_features))
            self.bias_imag = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform initialization"""
        nn.init.kaiming_uniform_(self.weight_real, a=torch.nn.init.calculate_gain('linear'))
        nn.init.kaiming_uniform_(self.weight_imag, a=torch.nn.init.calculate_gain('linear'))
        if self.bias_real is not None:
            nn.init.zeros_(self.bias_real)
            nn.init.zeros_(self.bias_imag)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for complex linear transformation.
        
        Args:
            x: Input tensor of shape (..., in_features * 2) where the last dimension
               contains interleaved real and imaginary parts
        
        Returns:
            Output tensor of shape (..., out_features * 2) with interleaved 
            real and imaginary parts
        """
        # Split input into real and imaginary parts
        x_real, x_imag = torch.chunk(x, 2, dim=-1)
        
        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        out_real = F.linear(x_real, self.weight_real, self.bias_real) - \
                   F.linear(x_imag, self.weight_imag, None)
        out_imag = F.linear(x_real, self.weight_imag, self.bias_imag if self.bias_real is not None else None) + \
                   F.linear(x_imag, self.weight_real, None)
        
        # Concatenate real and imaginary parts
        return torch.cat([out_real, out_imag], dim=-1)
    
    def extra_repr(self) -> str:
        """String representation of the layer"""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias_real is not None}'


class IndraQuantum(nn.Module):
    """
    IndraQuantum: A quantum-inspired language model optimized for low VRAM.
    
    Uses complex-valued embeddings and quantum-inspired operations for 
    extreme parameter efficiency on consumer GPUs.
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of the model embeddings
        n_layers: Number of transformer-like layers
        n_heads: Number of attention heads
        dropout: Dropout probability
        max_seq_length: Maximum sequence length
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 512
    ):
        super(IndraQuantum, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_length = max_seq_length
        
        # Complex-valued embeddings (real and imaginary parts)
        # We use d_model * 2 to store both real and imaginary components
        from .embedding import QuantumEmbedding
        self.token_embedding = QuantumEmbedding(vocab_size, d_model)
        self.position_embedding = QuantumEmbedding(max_seq_length, d_model)
        
        # Quantum-inspired transformation layers
        self.quantum_layers = nn.ModuleList([
            ComplexLinear(d_model, d_model)
            for _ in range(n_layers)
        ])
        
        # Layer normalization (applied separately to real and imaginary parts)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model * 2)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model * 2, vocab_size)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        # Embeddings are already initialized by QuantumEmbedding
        # nn.init.normal_(self.token_embedding.weight, std=0.02)
        # nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.output_projection.weight, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the IndraQuantum model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_length)
            attention_mask: Optional attention mask of shape (batch_size, seq_length)
        
        Returns:
            Logits of shape (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_embeds + position_embeds
        x = self.dropout(x)
        
        # Pass through quantum layers
        for layer, norm in zip(self.quantum_layers, self.layer_norms):
            # Residual connection
            residual = x
            x = layer(x)
            x = norm(x + residual)
            x = self.dropout(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_length)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: If specified, only sample from top k tokens
        
        Returns:
            Generated token IDs of shape (batch_size, seq_length + max_new_tokens)
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate if exceeding max sequence length
                if input_ids.shape[1] > self.max_seq_length:
                    input_ids = input_ids[:, -self.max_seq_length:]
                
                # Forward pass
                logits = self.forward(input_ids)
                
                # Get logits for last token and apply temperature
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def get_num_params(self) -> int:
        """Calculate total number of parameters"""
        return sum(p.numel() for p in self.parameters())
