"""
Phase Shift Module
Implements graph-induced phase rotation logic for quantum-inspired operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PhaseShift(nn.Module):
    """
    Graph-induced phase rotation layer for quantum-inspired transformations.
    
    Applies phase rotations to complex-valued representations based on 
    graph structure or learned phase parameters. This enables quantum-like
    interference patterns in the model.
    
    Args:
        d_model: Dimension of the model (should match the real component size)
        num_phases: Number of learnable phase parameters
        use_graph: If True, expects graph adjacency information for phase computation
    """
    
    def __init__(
        self,
        d_model: int,
        num_phases: int = 8,
        use_graph: bool = False
    ):
        super(PhaseShift, self).__init__()
        
        self.d_model = d_model
        self.num_phases = num_phases
        self.use_graph = use_graph
        
        # Learnable phase parameters
        self.phase_params = nn.Parameter(torch.randn(num_phases, d_model) * 0.1)
        
        if use_graph:
            # Graph convolution for phase computation
            self.graph_conv = nn.Linear(d_model * 2, num_phases)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.uniform_(self.phase_params, -math.pi, math.pi)
        if self.use_graph:
            nn.init.kaiming_uniform_(self.graph_conv.weight, a=math.sqrt(5))
            nn.init.zeros_(self.graph_conv.bias)
    
    def compute_phases(
        self,
        x: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute phase angles for rotation.
        
        Args:
            x: Input tensor of shape (..., d_model * 2)
            adjacency: Optional adjacency matrix for graph-based phase computation
        
        Returns:
            Phase angles of shape (..., d_model)
        """
        batch_shape = x.shape[:-1]
        
        if self.use_graph and adjacency is not None:
            # Graph-based phase computation
            # Apply graph convolution to get phase weights
            phase_weights = self.graph_conv(x)  # (..., num_phases)
            phase_weights = F.softmax(phase_weights, dim=-1)
            
            # Combine with learnable phase parameters
            phases = torch.einsum('...p,pd->...d', phase_weights, self.phase_params)
        else:
            # Use mean of learnable phases as default
            phases = self.phase_params.mean(dim=0).unsqueeze(0)
            phases = phases.expand(*batch_shape, self.d_model)
        
        return phases
    
    def apply_phase_rotation(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        phases: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply phase rotation to complex numbers.
        
        Phase rotation: (a + bi) * e^(iθ) = (a + bi) * (cos(θ) + i*sin(θ))
                       = (a*cos(θ) - b*sin(θ)) + i*(a*sin(θ) + b*cos(θ))
        
        Args:
            x_real: Real component of shape (..., d_model)
            x_imag: Imaginary component of shape (..., d_model)
            phases: Phase angles of shape (..., d_model)
        
        Returns:
            Tuple of (rotated_real, rotated_imag)
        """
        cos_phase = torch.cos(phases)
        sin_phase = torch.sin(phases)
        
        # Apply rotation
        out_real = x_real * cos_phase - x_imag * sin_phase
        out_imag = x_real * sin_phase + x_imag * cos_phase
        
        return out_real, out_imag
    
    def forward(
        self,
        x: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass applying phase shift to complex-valued input.
        
        Args:
            x: Input tensor of shape (..., d_model * 2) with interleaved 
               real and imaginary components
            adjacency: Optional adjacency matrix for graph-based phase computation
               Shape: (batch_size, seq_length, seq_length) or (seq_length, seq_length)
        
        Returns:
            Phase-shifted tensor of shape (..., d_model * 2)
        """
        # Split into real and imaginary parts
        x_real, x_imag = torch.chunk(x, 2, dim=-1)
        
        # Compute phases
        phases = self.compute_phases(x, adjacency)
        
        # Apply phase rotation
        out_real, out_imag = self.apply_phase_rotation(x_real, x_imag, phases)
        
        # Concatenate back
        return torch.cat([out_real, out_imag], dim=-1)
    
    def get_phase_statistics(self, x: torch.Tensor) -> dict:
        """
        Get statistics about the current phase distribution.
        
        Args:
            x: Input tensor
        
        Returns:
            Dictionary with phase statistics
        """
        phases = self.compute_phases(x, None)
        
        return {
            'mean_phase': phases.mean().item(),
            'std_phase': phases.std().item(),
            'min_phase': phases.min().item(),
            'max_phase': phases.max().item(),
        }
    
    def extra_repr(self) -> str:
        """String representation of the layer"""
        return f'd_model={self.d_model}, num_phases={self.num_phases}, use_graph={self.use_graph}'


class GraphPhaseEncoder(nn.Module):
    """
    Encoder that combines graph structure with phase rotations.
    
    Useful for encoding structural information from graphs into
    quantum-inspired representations.
    
    Args:
        d_model: Model dimension
        num_layers: Number of phase shift layers
        num_phases: Number of phase parameters per layer
    """
    
    def __init__(
        self,
        d_model: int,
        num_layers: int = 3,
        num_phases: int = 8
    ):
        super(GraphPhaseEncoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Stack of phase shift layers
        self.phase_layers = nn.ModuleList([
            PhaseShift(d_model, num_phases, use_graph=(i == 0))
            for i in range(num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model * 2)
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through stacked phase shift layers.
        
        Args:
            x: Input tensor of shape (..., d_model * 2)
            adjacency: Optional adjacency matrix
        
        Returns:
            Encoded tensor of shape (..., d_model * 2)
        """
        for i, (phase_layer, norm) in enumerate(zip(self.phase_layers, self.layer_norms)):
            residual = x
            # Only use adjacency for first layer
            adj = adjacency if i == 0 else None
            x = phase_layer(x, adj)
            x = norm(x + residual)
        
        return x
