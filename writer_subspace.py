"""Writer-subspace factorization modules for nanoGPT.

These modules factor residual-stream writers (columns of projection matrices) and
optionally fake-quantize the low-rank coefficient writers during training.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


def fake_quant_columns(B, bits):
    """Symmetric per-writer fake quantization for columns of a coefficient matrix."""
    if bits >= 16:
        return B

    qmax = 2 ** (bits - 1) - 1
    scale = B.float().detach().abs().amax(dim=0, keepdim=True)
    scale = scale.clamp_min(1e-12) / qmax

    Bq = (B.float() / scale).round().clamp(-qmax, qmax) * scale
    Bq = Bq.to(B.dtype)
    return B + (Bq - B).detach()


def fake_quant_rows(A, bits):
    """Symmetric fake quantization for rows of a vocabulary coefficient table."""
    if bits >= 16:
        return A

    qmax = 2 ** (bits - 1) - 1
    scale = A.float().detach().abs().amax(dim=1, keepdim=True)
    scale = scale.clamp_min(1e-12) / qmax

    Aq = (A.float() / scale).round().clamp(-qmax, qmax) * scale
    Aq = Aq.to(A.dtype)
    return A + (Aq - A).detach()


class SubspaceLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True, bits=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.bits = bits
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)

    @property
    def weight(self):
        return self.up.weight @ self.down.weight

    @property
    def bias(self):
        return self.up.bias

    def forward(self, x):
        B = fake_quant_columns(self.down.weight, self.bits)
        return self.up(F.linear(x, B))

    @classmethod
    @torch.no_grad()
    def from_linear(cls, layer, rank, bits=16, column_rms=None):
        W = layer.weight.float()
        if rank > min(W.shape):
            raise ValueError(f"rank={rank} exceeds min(weight.shape)={min(W.shape)}")
        X = W if column_rms is None else W * column_rms.reshape(1, -1).to(W.device, W.dtype)

        eigenvalues, eigenvectors = torch.linalg.eigh(X @ X.T)
        del eigenvalues
        U = eigenvectors[:, -rank:].flip(1)
        B = U.T @ W

        result = cls(
            layer.in_features,
            layer.out_features,
            rank,
            bias=layer.bias is not None,
            bits=bits,
        ).to(layer.weight.device, layer.weight.dtype)
        result.up.weight.copy_(U.to(layer.weight.dtype))
        result.down.weight.copy_(B.to(layer.weight.dtype))
        if layer.bias is not None:
            result.up.bias.copy_(layer.bias)
        result.up.weight.requires_grad_(False)
        return result


class SubspaceVocab(nn.Module):
    """Shared low-rank token embedding/unembedding module.

    Factorizes E[V, C] ~= A[V, r] @ U.T[r, C]. Embedding returns A_t U.T;
    logits compute x U A.T. The coefficient rows A are fake-quantized.
    """

    def __init__(self, vocab_size, n_embd, rank, bits=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.rank = rank
        self.bits = bits
        self.coeff = nn.Embedding(vocab_size, rank)
        self.up = nn.Linear(rank, n_embd, bias=False)

    @property
    def weight(self):
        return self.coeff.weight @ self.up.weight.T

    def forward(self, idx):
        A = fake_quant_rows(self.coeff.weight, self.bits)
        return F.embedding(idx, A) @ self.up.weight.T

    def logits(self, x):
        A = fake_quant_rows(self.coeff.weight, self.bits)
        return (x @ self.up.weight) @ A.T

    @classmethod
    @torch.no_grad()
    def from_embedding(cls, embedding, rank, bits=16):
        E = embedding.weight.float()
        if rank > min(E.shape):
            raise ValueError(f"rank={rank} exceeds min(embedding.shape)={min(E.shape)}")
        _, _, Vh = torch.linalg.svd(E, full_matrices=False)
        U = Vh[:rank].T
        A = E @ U
        result = cls(embedding.num_embeddings, embedding.embedding_dim, rank, bits=bits).to(
            embedding.weight.device, embedding.weight.dtype
        )
        result.coeff.weight.copy_(A.to(embedding.weight.dtype))
        result.up.weight.copy_(U.T.to(embedding.weight.dtype))
        result.up.weight.requires_grad_(False)
        return result
