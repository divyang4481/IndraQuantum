import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantumEmbedding(nn.Module):
    """
    Complex-valued embedding layer.
    
    Initialized with polar coordinates (magnitude ~ 1, phase ~ uniform)
    to represent quantum states.
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super(QuantumEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # We store Real and Imaginary parts interleaved in the last dimension
        # Shape: [num_embeddings, embedding_dim * 2]
        self.embedding = nn.Embedding(num_embeddings, embedding_dim * 2, padding_idx=padding_idx)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize with quantum-inspired polar coordinates
        # Magnitude r ~ 1 (with some small variance)
        # Phase theta ~ Uniform(0, 2pi)
        
        with torch.no_grad():
            # Initialize with more stable values
            # Magnitude r ~ 1 (small variance)
            r = torch.ones(self.num_embeddings, self.embedding_dim) + torch.randn(self.num_embeddings, self.embedding_dim) * 0.02
            # Phase theta ~ Small random noise around 0 initially, to behave like real-valued
            # This helps training start stable and then "drift" into complex space
            theta = torch.randn(self.num_embeddings, self.embedding_dim) * 0.1
            
            # Convert to Cartesian
            real = r * torch.cos(theta)
            imag = r * torch.sin(theta)
            
            weight = torch.cat([real, imag], dim=-1)
            self.embedding.weight.data.copy_(weight)
            
            if self.padding_idx is not None:
                self.embedding.weight.data[self.padding_idx].fill_(0)

    def forward(self, input):
        return self.embedding(input)

class QuantumTTEmbedding(nn.Module):
    """
    Tensor Train Decomposition of the Embedding Matrix.
    
    Replaces the dense [Vocab, D] matrix with a set of core tensors.
    Vocab V is factorized into v1 * v2 * v3
    Dim D is factorized into d1 * d2 * d3
    
    Total parameters reduction: O(V*D) -> O(rank^2 * sum(vi*di))
    """
    def __init__(self, num_embeddings, embedding_dim, tt_rank=4):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tt_rank = tt_rank
        
        # Factorization Strategy for Vocab=32000 and Dim=128
        # We need generic factorization but let's hardcode for the specific case first for robustness,
        # then add fallback.
        
        if num_embeddings >= 32000:
            # Round up to 32000 if close, or handle larger.
            # 32000 = 20 * 40 * 40
            self.vocab_factors = [20, 40, 40]
            # Map any index >= 32000 to 0 or handle modulo? 
            # For safety, we'll clamp or modulo.
        else:
            # Fallback for smaller vocabs (e.g. tests)
            # e.g. 100 -> 4 * 5 * 5
            self.vocab_factors = [int(num_embeddings**(1/3))]*3
            # Adjust last factor
            self.vocab_factors[-1] = num_embeddings // (self.vocab_factors[0]*self.vocab_factors[1])
            # Remainder handling is complex, let's assume standard vocab for now.
            if np.prod(self.vocab_factors) < num_embeddings:
                 self.vocab_factors[-1] += 1
        
        # Factorization for Dim=128
        # 128 = 4 * 4 * 8
        self.dim_factors = [4, 4, 8]
        
        assert np.prod(self.dim_factors) == embedding_dim, f"Dim factors {self.dim_factors} must product to {embedding_dim}"
        
        # TT Cores (Real and Imaginary separate)
        # Core 1: [1, v1, d1, r]
        # Core 2: [r, v2, d2, r]
        # Core 3: [r, v3, d3, 1]
        
        self.cores_real = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.vocab_factors[0], self.dim_factors[0], tt_rank) * 0.1),
            nn.Parameter(torch.randn(tt_rank, self.vocab_factors[1], self.dim_factors[1], tt_rank) * 0.1),
            nn.Parameter(torch.randn(tt_rank, self.vocab_factors[2], self.dim_factors[2], 1) * 0.1)
        ])
        
        self.cores_imag = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.vocab_factors[0], self.dim_factors[0], tt_rank) * 0.1),
            nn.Parameter(torch.randn(tt_rank, self.vocab_factors[1], self.dim_factors[1], tt_rank) * 0.1),
            nn.Parameter(torch.randn(tt_rank, self.vocab_factors[2], self.dim_factors[2], 1) * 0.1)
        ])
        
    def _indices_to_factors(self, indices):
        # Convert flat indices [B, S] to factor indices [B, S, 3]
        # idx = i*v2*v3 + j*v3 + k
        
        v1, v2, v3 = self.vocab_factors
        
        # Handle out of bounds by modulo (safe fallback)
        indices = indices % (v1 * v2 * v3)
        
        k = indices % v3
        indices = indices // v3
        j = indices % v2
        i = indices // v2
        
        return torch.stack([i, j, k], dim=-1)
        
    def forward(self, input_ids):
        # input_ids: [Batch, Seq]
        B, S = input_ids.shape
        
        # 1. Get Factors
        factors = self._indices_to_factors(input_ids) # [B, S, 3]
        
        # 2. Gather Slices
        # We need to gather from cores based on factors.
        # Core 1: [1, v1, d1, r] -> Gather i -> [B, S, 1, d1, r]
        
        def gather_core(core, idx_tensor):
            # core: [r_in, v, d, r_out]
            # idx_tensor: [B, S] (values in 0..v-1)
            
            # Expand core to [B, S, r_in, v, d, r_out] ? No too big.
            # Use embedding lookup logic.
            # Treat core as EmbeddingBag? No.
            # Reshape core to [v, r_in * d * r_out]
            r_in, v, d, r_out = core.shape
            flat_core = core.permute(1, 0, 2, 3).reshape(v, -1)
            
            gathered = F.embedding(idx_tensor, flat_core) # [B, S, r_in*d*r_out]
            return gathered.view(B, S, r_in, d, r_out)

        # Real Part
        c1 = gather_core(self.cores_real[0], factors[..., 0]) # [B, S, 1, d1, r]
        c2 = gather_core(self.cores_real[1], factors[..., 1]) # [B, S, r, d2, r]
        c3 = gather_core(self.cores_real[2], factors[..., 2]) # [B, S, r, d3, 1]
        
        # 3. Contraction
        # We want [B, S, d1*d2*d3]
        # Einsum: 
        # c1: b s a d e (a=1, e=r)
        # c2: b s e f g (e=r, g=r)
        # c3: b s g h i (g=r, i=1)
        # Result: b s d f h
        
        # Contract c1 and c2 along 'e' (rank)
        # [B, S, 1, d1, r] x [B, S, r, d2, r] -> [B, S, 1, d1, d2, r]
        # Reshape to merge d1, d2?
        
        # Let's use einsum per batch/seq item? No, slow.
        # Use matmul.
        
        # Reshape for matmul
        # c1: [BS, 1*d1, r]
        # c2: [BS, r, d2*r] -> Wait, d2 is in middle.
        
        # Let's stick to einsum, it's optimized.
        # But we have 5 dims.
        # 'bsade,bsefg,bsghi->bsdfh'
        # b=batch, s=seq. We can merge them -> N = B*S
        # 'nade,nefg,nghi->ndfh'
        
        c1_flat = c1.view(-1, 1, self.dim_factors[0], self.tt_rank)
        c2_flat = c2.view(-1, self.tt_rank, self.dim_factors[1], self.tt_rank)
        c3_flat = c3.view(-1, self.tt_rank, self.dim_factors[2], 1)
        
        # Contract 1 & 2
        # [N, 1, d1, r] * [N, r, d2, r] -> [N, 1, d1, d2, r]
        # Einsum: 'nadr,nrfr->nadfr' (Wait, indices matching?)
        # c1: n a d r
        # c2: n r f g (using g for output rank)
        # sum over r (input of c2, output of c1)
        temp = torch.einsum('nadr,nrfg->nadfg', c1_flat, c2_flat) 
        # temp: [N, 1, d1, d2, r]
        
        # Contract temp & 3
        # temp: n a d f g (g is rank)
        # c3: n g h i (g is input rank, i is 1)
        final = torch.einsum('nadfg,nghi->nadfhi', temp, c3_flat)
        # final: [N, 1, d1, d2, d3, 1]
        
        real_out = final.reshape(B, S, -1) # [B, S, D]
        
        # Imag Part (Same logic)
        c1_i = gather_core(self.cores_imag[0], factors[..., 0])
        c2_i = gather_core(self.cores_imag[1], factors[..., 1])
        c3_i = gather_core(self.cores_imag[2], factors[..., 2])
        
        c1_flat_i = c1_i.view(-1, 1, self.dim_factors[0], self.tt_rank)
        c2_flat_i = c2_i.view(-1, self.tt_rank, self.dim_factors[1], self.tt_rank)
        c3_flat_i = c3_i.view(-1, self.tt_rank, self.dim_factors[2], 1)
        
        temp_i = torch.einsum('nadr,nrfg->nadfg', c1_flat_i, c2_flat_i)
        final_i = torch.einsum('nadfg,nghi->nadfhi', temp_i, c3_flat_i)
        imag_out = final_i.reshape(B, S, -1)
        
        return torch.cat([real_out, imag_out], dim=-1)
