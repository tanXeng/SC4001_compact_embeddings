"""Legacy-equivalent pre-LayerNorm causal Transformer, without experiment logic."""

from __future__ import annotations

import torch
import hashlib
import math
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
def _dimensions(*values):
    if any(type(v) is not int or v < 1 for v in values):
        raise ValueError("Dimensions must be positive integers")


class LowRankVocabulary(nn.Module):
    """E=A B; both input and full-softmax output use these same factors."""
    def __init__(self, vocab_size: int, hidden: int, rank: int):
        super().__init__()
        _dimensions(vocab_size, hidden, rank)
        self.vocab_size, self.hidden, self.rank = vocab_size, hidden, rank
        self.a = nn.Parameter(torch.empty(vocab_size, rank))
        self.b = nn.Parameter(torch.empty(rank, hidden))
        nn.init.normal_(self.a, std=.02)
        nn.init.normal_(self.b, std=rank**-.5)

    def lookup(self, token_ids):
        return F.embedding(token_ids, self.a) @ self.b

    def project(self, hidden_states):
        return (hidden_states @ self.b.T) @ self.a.T

    def specification(self):
        return dict(kind="low_rank", vocab_size=self.vocab_size, hidden=self.hidden,
                    rank=self.rank, initialization="a_normal_0.02_b_normal_inv_sqrt_rank_v1")


class HashVocabulary(nn.Module):
    """Learned weighted shared components, not ordinary feature hashing.

    Serialized indices avoid platform-dependent hashing and are verified on load.
    SHA-256 input is ASCII 'hash_embedding_v1:{seed}:{token_id}'.
    """
    def __init__(self, vocab_size: int, hidden: int, buckets: int, seeds=(17, 29),
                 importance_init="normal"):
        super().__init__()
        _dimensions(vocab_size, hidden, buckets)
        if (not seeds or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds)
                or len(set(seeds)) != len(seeds) or importance_init not in {"normal", "constant"}):
            raise ValueError("Require distinct nonnegative int64 seeds and normal/constant initialization")
        self.vocab_size, self.hidden, self.buckets = vocab_size, hidden, buckets
        self.seeds, self.importance_init = tuple(seeds), importance_init
        indices = [[int.from_bytes(hashlib.sha256(
            f"hash_embedding_v1:{s}:{i}".encode("ascii")).digest(), "big") % buckets
                    for s in seeds] for i in range(vocab_size)]
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.int64))
        self.components = nn.Parameter(torch.empty(buckets, hidden))
        self.importance = nn.Parameter(torch.empty(vocab_size, len(seeds)))
        nn.init.normal_(self.components, std=.02)
        if importance_init == "normal":
            nn.init.normal_(self.importance, std=len(seeds)**-.5)
        else:
            nn.init.constant_(self.importance, len(seeds)**-.5)

    def lookup(self, token_ids):
        codes = self.indices[token_ids]
        return (F.embedding(codes, self.components) * self.importance[token_ids].unsqueeze(-1)).sum(-2)

    def project(self, hidden_states):
        effective = (
            F.embedding(self.indices, self.components)
            * self.importance.unsqueeze(-1)
        ).sum(1)
        return hidden_states @ effective.T

    def specification(self):
        return dict(kind="hash", vocab_size=self.vocab_size, hidden=self.hidden,
                    buckets=self.buckets, seeds=list(self.seeds), importance_init=self.importance_init,
                    hash_version="sha256_ascii_v1", component_std=.02)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        supplied = state_dict.get(prefix + "indices")
        if supplied is not None and not torch.equal(supplied.cpu(), self.indices.cpu()):
            raise ValueError("Hash indices disagree with deterministic specification")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class AdaptiveVocabulary(nn.Module):
    """Frequency bands with projected tables and an exactly tied full softmax.

    Counts must come only from training. Specials occupy the head in supplied
    order; remaining ties in frequency break by original ID. EOS is trainable.
    """
    def __init__(self, counts: Tensor, hidden: int, cutoffs, initial_dim=None,
                 factor=4., special_ids=(0,)):
        super().__init__()
        _dimensions(hidden)
        if (counts.ndim != 1 or counts.dtype != torch.int64 or len(counts) < 1
                or (counts < 0).any()):
            raise ValueError("Require nonnegative int64 training counts")
        self.vocab_size, self.hidden = len(counts), hidden
        self.initial_dim = hidden if initial_dim is None else initial_dim
        _dimensions(self.initial_dim)
        if not math.isfinite(factor) or factor < 1:
            raise ValueError("Reduction factor must be finite and >=1")
        self.factor = float(factor)
        self.cutoffs = tuple(cutoffs) + (len(counts),)
        if (any(type(c) is not int for c in self.cutoffs) or
                any(a >= b for a, b in zip((0,) + self.cutoffs, self.cutoffs))):
            raise ValueError("Cutoffs must increase strictly inside vocabulary")
        self.special_ids = tuple(special_ids)
        if (len(set(special_ids)) != len(special_ids) or len(special_ids) > self.cutoffs[0]
                or any(type(i) is not int or not 0 <= i < len(counts) for i in special_ids)):
            raise ValueError("Special IDs must be distinct and fit the head")
        values = counts.cpu().tolist()
        specials = set(special_ids)
        order = list(special_ids) + sorted((i for i in range(len(counts)) if i not in specials),
                                         key=lambda i: (-values[i], i))
        self.register_buffer("order", torch.tensor(order, dtype=torch.int64))
        self.register_buffer("inverse", self.order.argsort())
        self.tables, self.projections = nn.ParameterList(), nn.ParameterList()
        start = 0
        for j, end in enumerate(self.cutoffs):
            dim = int(self.initial_dim // self.factor**j)
            _dimensions(dim)
            a = nn.Parameter(torch.empty(end-start, dim))
            b = nn.Parameter(torch.empty(dim, hidden))
            nn.init.normal_(a, std=.02)
            nn.init.normal_(b, std=dim**-.5)
            self.tables.append(a)
            self.projections.append(b)
            start = end

    def lookup(self, token_ids):
        positions = self.inverse[token_ids]
        result = self.tables[0].new_zeros((*token_ids.shape, self.hidden))
        start = 0
        for end, a, b in zip(self.cutoffs, self.tables, self.projections):
            mask = (positions >= start) & (positions < end)
            result[mask] = F.embedding(positions[mask]-start, a) @ b
            start = end
        return result

    def project(self, hidden_states):
        ranked = torch.cat([(hidden_states @ b.T) @ a.T
                            for a, b in zip(self.tables, self.projections)], dim=-1)
        return ranked[..., self.inverse]

    def specification(self):
        return dict(kind="adaptive", vocab_size=self.vocab_size, hidden=self.hidden,
                    cutoffs=list(self.cutoffs[:-1]), initial_dim=self.initial_dim,
                    factor=self.factor, special_ids=list(self.special_ids),
                    initialization="a_normal_0.02_b_normal_inv_sqrt_dim_v1")

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        order, inverse = state_dict.get(prefix+"order"), state_dict.get(prefix+"inverse")
        if order is not None and inverse is not None:
            expected = torch.arange(self.vocab_size, device=order.device)
            if (order.dtype != torch.int64 or inverse.dtype != torch.int64
                    or order.shape != expected.shape or inverse.shape != expected.shape
                    or not torch.equal(order.sort().values, expected)
                    or not torch.equal(inverse, order.argsort())
                    or order[:len(self.special_ids)].tolist() != list(self.special_ids)):
                raise ValueError("Invalid adaptive permutation")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


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
