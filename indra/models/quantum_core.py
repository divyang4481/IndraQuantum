"""
Quantum Core Module
Defines the IndraQuantum model and ComplexLinear layer using PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any

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



class DenseComplexLinear(nn.Module):
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
        super(DenseComplexLinear, self).__init__()
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


class TensorTrainComplexLinear(nn.Module):
    """
    Tensor Train Complex-valued linear layer for quantum-inspired transformations.

    Implements a 2-core Tensor Train decomposition on the Real and Imaginary
    weight matrices to drastically reduce parameter count.
    
    The implementation dynamically factors the in/out features using a 2-core TT
    approximation to replace the massive dense matrix W.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, tt_rank: int = 4):
        super(TensorTrainComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tt_rank = tt_rank

        # Determine optimal factorization for in_features and out_features
        # We simplify the factorization to two cores: D = d1 * d2
        # For D_model=128 (FNN, QKV) -> 16*8
        # For D_model*4=512 (FFN Inner) -> 32*16
        def get_factorization(D):
            if D == 128: return [16, 8]
            if D == 512: return [32, 16]
            # General simple factorization (not optimal, but necessary for dynamic size)
            if D % 16 == 0 and D > 16: return [16, D // 16]
            return [D, 1]

        self.d_in = get_factorization(in_features)
        self.d_out = get_factorization(out_features)
        
        # TT Cores for Real Part (W_real approx G_1^R * G_2^R)
        # G1_R shape: (r0, o1, r1, i1)
        # G2_R shape: (r1, o2, rk, i2)
        r0 = 1
        rk = 1
        self.G1_R = nn.Parameter(torch.Tensor(r0, self.d_out[0], tt_rank, self.d_in[0]))
        self.G2_R = nn.Parameter(torch.Tensor(tt_rank, self.d_out[1], rk, self.d_in[1]))

        # TT Cores for Imaginary Part (W_imag approx G_1^I * G_2^I)
        self.G1_I = nn.Parameter(torch.Tensor(r0, self.d_out[0], tt_rank, self.d_in[0]))
        self.G2_I = nn.Parameter(torch.Tensor(tt_rank, self.d_out[1], rk, self.d_in[1]))

        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(out_features))
            self.bias_imag = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
        
        self.reset_parameters()

    def _unfold_tt(self, G1, G2):
        # Reconstruct the full weight matrix W from the TT cores G1, G2
        # W_tt: (r0, o1, r1, i1) x (r1, o2, rk, i2) -> (r0, o1, o2, rk, i1, i2)
        # Using letters only for einsum compatibility:
        # a=r0, b=o1, c=r1, d=i1, e=o2, f=rk, g=i2
        W_tt = torch.einsum('abcd,cefg->abefdg', G1, G2)
        
        D_out = self.d_out[0] * self.d_out[1]
        D_in = self.d_in[0] * self.d_in[1]
        
        W = W_tt.reshape(D_out, D_in)
        return W

    def reset_parameters(self):
        # Initializing based on TT-rank (simplified)
        std = math.sqrt(2.0 / (self.in_features * self.tt_rank))
        nn.init.uniform_(self.G1_R, -std, std)
        nn.init.uniform_(self.G2_R, -std, std)
        nn.init.uniform_(self.G1_I, -std, std)
        nn.init.uniform_(self.G2_I, -std, std)

        if self.bias_real is not None:
            nn.init.zeros_(self.bias_real)
            nn.init.zeros_(self.bias_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reconstruct weights on the fly 
        W_real = self._unfold_tt(self.G1_R, self.G2_R)
        W_imag = self._unfold_tt(self.G1_I, self.G2_I)

        # Split input into real and imaginary parts
        x_real, x_imag = torch.chunk(x, 2, dim=-1)
        
        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        out_real = F.linear(x_real, W_real, self.bias_real) - \
                   F.linear(x_imag, W_imag, None)
        out_imag = F.linear(x_real, W_imag, self.bias_imag if self.bias_real is not None else None) + \
                   F.linear(x_imag, W_real, None)
        
        return torch.cat([out_real, out_imag], dim=-1)

    def get_num_params(self) -> int:
        num_params_per_core = (
            self.G1_R.numel() + self.G2_R.numel()
        )
        total_params = 2 * num_params_per_core
        if self.bias_real is not None:
            total_params += self.bias_real.numel() + self.bias_imag.numel()
        return total_params

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, tt_rank={self.tt_rank}, bias={self.bias_real is not None}'

class CustomComplexLinear(nn.Module):
    """
    Adapter to switch between Tensor Train (efficient) and Dense (standard) Complex Linear layers.
    """
    def __init__(self, in_features, out_features, bias=True, config=None):
        super().__init__()
        self.config = config if config is not None else {}
        tt_rank = self.config.get('tt_rank', 0)
        
        # Apply TT only if tt_rank > 0 and the dimensions match the designed TT structures
        is_tt_compatible = (tt_rank > 0 and 
                            ((in_features == 128 and out_features == 128) or 
                             (in_features == 128 and out_features == 512) or
                             (in_features == 512 and out_features == 128)))
        
        if is_tt_compatible:
            self.layer = TensorTrainComplexLinear(in_features, out_features, bias=bias, tt_rank=tt_rank)
        else:
            self.layer = DenseComplexLinear(in_features, out_features, bias=bias)
            
    def forward(self, x):
        return self.layer(x)
        
    def get_num_params(self) -> int:
        if hasattr(self.layer, 'get_num_params'):
            return self.layer.get_num_params()
        else:
            return sum(p.numel() for p in self.layer.parameters())


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
        max_seq_length: int = 512,
        config: Optional[Dict[str, Any]] = None
    ):
        super(IndraQuantum, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_length = max_seq_length
        self.config = config if config is not None else {}
        
        # Complex-valued embeddings (real and imaginary parts)
        # We use d_model * 2 to store both real and imaginary components
        from .embedding import QuantumEmbedding
        self.token_embedding = QuantumEmbedding(vocab_size, d_model)
        self.position_embedding = QuantumEmbedding(max_seq_length, d_model)
        
        # Quantum-inspired transformation layers
        self.quantum_layers = nn.ModuleList([
            CustomComplexLinear(d_model, d_model, config=self.config)
            for _ in range(n_layers)
        ])
        
        # Layer normalization (applied separately to real and imaginary parts)
        # Layer normalization (applied separately to real and imaginary parts)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model * 2)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Output projection to vocabulary
        # Weight Tying: Share weights with embedding
        self.output_projection = nn.Linear(d_model * 2, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.embedding.weight
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        # Embeddings are already initialized by QuantumEmbedding
        # nn.init.normal_(self.token_embedding.weight, std=0.02)
        # nn.init.normal_(self.position_embedding.weight, std=0.02)
        # Output projection weights are tied, so no need to init them separately
        pass
    
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
