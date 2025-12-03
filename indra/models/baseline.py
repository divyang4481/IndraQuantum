import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardBaseline(nn.Module):
    """
    Standard MLP-based Baseline Model.
    Designed to be structurally similar to IndraQuantum but using standard Real-valued layers.
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: The 'effective' dimension. 
                 IndraQuantum uses d_model (complex) -> 2*d_model (real representation).
                 So here we use hidden_dim = 2*d_model to match the vector size.
        n_layers: Number of layers
        dropout: Dropout probability
        max_seq_length: Maximum sequence length
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int, # We interpret this as the 'base' dim, so actual dim is 2*d_model
        n_layers: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 512
    ):
        super(StandardBaseline, self).__init__()
        
        self.hidden_dim = d_model * 2
        self.max_seq_length = max_seq_length
        
        # Standard Embeddings
        self.token_embedding = nn.Embedding(vocab_size, self.hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_length, self.hidden_dim)
        
        # Standard Linear Layers (MLP)
        # To match the "per-token" processing of IndraQuantum (which has no attention yet),
        # we use simple Linear layers with non-linearity.
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(), # Standard activation
                nn.Linear(self.hidden_dim, self.hidden_dim) # Project back
            )
            for _ in range(n_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.output_projection = nn.Linear(self.hidden_dim, vocab_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(positions)
        
        x = token_embeds + position_embeds
        x = self.dropout(x)
        
        # Layers
        for layer, norm in zip(self.layers, self.layer_norms):
            residual = x
            x = layer(x)
            x = norm(x + residual)
            x = self.dropout(x)
            
        logits = self.output_projection(x)
        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
