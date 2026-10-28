"""Legacy-equivalent pre-LayerNorm causal Transformer, without experiment logic."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


class Block(nn.Module):
    """Operation ordering, biases and dropout locations match legacy src/model.py."""
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.dropout = config.dropout
        self.ln1 = nn.LayerNorm(config.hidden)
        self.qkv = nn.Linear(config.hidden, 3*config.hidden, bias=False)
        self.attention_out = nn.Linear(config.hidden, config.hidden)
        self.ln2 = nn.LayerNorm(config.hidden)
        self.ff = nn.Sequential(
            nn.Linear(config.hidden, config.feedforward), nn.GELU(),
            nn.Linear(config.feedforward, config.hidden), nn.Dropout(config.dropout))

    def forward(self, x: Tensor) -> Tensor:
        batch, time, hidden = x.shape
        qkv = self.qkv(self.ln1(x)).reshape(batch, time, 3, self.heads, hidden//self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        attended = attention.transpose(1, 2).reshape(batch, time, hidden)
        x = x + F.dropout(self.attention_out(attended), p=self.dropout, training=self.training)
        return x + self.ff(self.ln2(x))


class CausalTransformer(nn.Module):
    def __init__(self, config: ModelConfig, vocab_size: int) -> None:
        super().__init__()
        if type(vocab_size) is not int or vocab_size < 1:
            raise ValueError("Vocabulary size must be a positive integer")
        self.config = config
        # No random draw before legacy body construction: preserve seeded init too.
        self.vocabulary = DenseVocabulary(torch.zeros(vocab_size, config.hidden))
        self.position = nn.Embedding(config.context, config.hidden)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.hidden)
        self.apply(self._initialize)
        nn.init.normal_(self.vocabulary.weight, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids: Tensor) -> Tensor:
        if (token_ids.ndim != 2 or token_ids.dtype != torch.int64
                or token_ids.shape[0] < 1 or not 0 < token_ids.shape[1] <= self.config.context):
            raise ValueError("Require nonempty int64 [batch, time] IDs within model context")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = self.vocabulary.lookup(token_ids) + self.position(positions)
        x = F.dropout(x, p=self.config.dropout, training=self.training)
        for block in self.blocks:
            x = block(x)
        return self.vocabulary.project(self.norm(x))


# Embedding layers
class DenseVocabulary(nn.Module):
    def __init__(self, weight: Tensor) -> None:
        super().__init__()
        if (weight.ndim != 2 or min(weight.shape) < 1 or weight.dtype != torch.float32
                or not torch.isfinite(weight).all()):
            raise ValueError("Require a nonempty finite FP32 embedding matrix")
        self.weight = nn.Parameter(weight.detach().clone())

    @property
    def vocab_size(self) -> int:
        return self.weight.shape[0]

    @property
    def hidden(self) -> int:
        return self.weight.shape[1]

    def lookup(self, token_ids: Tensor) -> Tensor:
        return F.embedding(token_ids, self.weight)

    def project(self, hidden_states: Tensor) -> Tensor:
        return F.linear(hidden_states, self.weight)

    def specification(self) -> dict:
        return {"kind": "dense", "vocab_size": self.vocab_size, "hidden": self.hidden}


def code_dtype(centroids: int) -> torch.dtype:
    if type(centroids) is not int or not 2 <= centroids <= 65536:
        raise ValueError("PQ supports 2..65536 centroids per subspace")
    return torch.uint8 if centroids <= 256 else torch.uint16


class PQVocabulary(nn.Module):
    """Trainable FP32 books and fixed compact assignments; no dense-table cache.

    A nonpersistent int64 index buffer includes subspace offsets. It avoids
    repeated uint16 casts and offset allocations, and is rebuilt on state load.
    Assignments are immutable during recovery; replacing them requires a new
    vocabulary. Runtime accounting includes this derived buffer.
    """
    def __init__(self, codebooks: Tensor, codes: Tensor) -> None:
        super().__init__()
        if (codebooks.ndim != 3 or codebooks.dtype != torch.float32
                or min(codebooks.shape) < 1 or codebooks.shape[0] < 2
                or codebooks.shape[2] < 2 or not torch.isfinite(codebooks).all()):
            raise ValueError("Require finite FP32 PQ books with m>=2 and subvector width>=2")
        m, k, _ = codebooks.shape
        if (codes.ndim != 2 or codes.shape[1] != m or len(codes) < 1
                or codes.dtype != code_dtype(k) or codes.device != codebooks.device
                or (codes.long() >= k).any()):
            raise ValueError("Invalid PQ assignment shape, IDs, device or index dtype")
        self.codebooks = nn.Parameter(codebooks.detach().clone())
        self.register_buffer("codes", codes.detach().clone())
        self.register_buffer("_indices", self._make_indices(), persistent=False)

    def _make_indices(self) -> Tensor:
        return self.codes.long() + torch.arange(
            self.codebooks.shape[0], device=self.codes.device) * self.codebooks.shape[1]

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        if (self.codes.long() >= self.codebooks.shape[1]).any():
            raise ValueError("PQ assignment out of range")
        self._indices = self._make_indices()

    @property
    def vocab_size(self) -> int:
        return len(self.codes)

    @property
    def hidden(self) -> int:
        return self.codebooks.shape[0] * self.codebooks.shape[2]

    def _reconstruct(self, indices: Tensor) -> Tensor:
        books = self.codebooks.flatten(0, 1)
        return F.embedding(indices, books).flatten(-2)

    def lookup(self, token_ids: Tensor) -> Tensor:
        return self._reconstruct(F.embedding(token_ids, self._indices))

    def materialize(self) -> Tensor:
        return self._reconstruct(self._indices)

    def project(self, hidden_states: Tensor) -> Tensor:
        return F.linear(hidden_states, self.materialize())

    def specification(self) -> dict:
        return {"kind": "pq", "vocab_size": self.vocab_size, "hidden": self.hidden,
                "subspaces": self.codebooks.shape[0], "centroids": self.codebooks.shape[1],
                "code_dtype": str(self.codes.dtype), "partition": "contiguous equal-width subvectors"}
