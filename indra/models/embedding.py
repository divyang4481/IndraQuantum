
import torch
import torch.nn as nn
import math

class QuantumEmbedding(nn.Module):
    """
    Quantum-inspired Embedding Layer.
    
    Instead of standard random initialization, this embedding is initialized
    in polar coordinates (magnitude and phase) and then converted to 
    rectangular coordinates (real and imaginary).
    
    This mimics the quantum state representation |psi> = r * e^(i*theta).
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None):
        """
        Args:
            num_embeddings: Size of the dictionary of embeddings
            embedding_dim: The size of each embedding vector (will be doubled for real+imag)
            padding_idx: If specified, the entries at padding_idx do not contribute to the gradient
        """
        super(QuantumEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # We use a standard embedding layer to store the parameters, 
        # but we interpret them as (Real, Imaginary) parts.
        # The size is embedding_dim * 2.
        self.embedding = nn.Embedding(num_embeddings, embedding_dim * 2, padding_idx=padding_idx)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initialize embeddings using quantum-inspired polar coordinates.
        """
        with torch.no_grad():
            # Initialize magnitudes (r) - typically close to 1 for normalized quantum states
            # We use a slight variation to allow for expressivity
            r = torch.ones(self.num_embeddings, self.embedding_dim) + \
                torch.randn(self.num_embeddings, self.embedding_dim) * 0.1
            
            # Initialize phases (theta) - uniformly distributed between -pi and pi
            theta = torch.rand(self.num_embeddings, self.embedding_dim) * 2 * math.pi - math.pi
            
            # Convert to Cartesian (Real, Imaginary)
            # real = r * cos(theta)
            # imag = r * sin(theta)
            real = r * torch.cos(theta)
            imag = r * torch.sin(theta)
            
            # Interleave real and imaginary parts
            # shape: [num_embeddings, embedding_dim * 2]
            weight = torch.stack([real, imag], dim=-1).view(self.num_embeddings, self.embedding_dim * 2)
            
            self.embedding.weight.copy_(weight)
            
            if self.padding_idx is not None:
                self.embedding.weight[self.padding_idx].fill_(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor containing indices into the embedding matrix
            
        Returns:
            Tensor of shape (..., embedding_dim * 2) containing complex embeddings
        """
        return self.embedding(input_ids)
