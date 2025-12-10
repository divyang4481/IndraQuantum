import torch
import torch.nn as nn


class ComplexEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Two real embeddings to make one complex
        self.embed_real = nn.Embedding(num_embeddings, embedding_dim)
        self.embed_imag = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input):
        # input: [Batch, Seq] indices
        r = self.embed_real(input)
        i = self.embed_imag(input)
        return torch.complex(r, i)
