"""Byte-exact document-local validation; no dataset download or test-loading route."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Iterator

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Document:
    """Already encoded [EOS, text..., EOS], with byte lengths aligned to IDs.

    Shard loading and tokenization supply these records. Incomplete UTF-8 token
    fragments retain their actual byte lengths.
    """
    ids: tuple[int, ...]
    byte_lengths: tuple[int, ...]
    raw_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.ids, tuple) or not isinstance(self.byte_lengths, tuple)
                or len(self.ids) < 2 or len(self.ids) != len(self.byte_lengths)
                or any(type(i) is not int or i < 0 for i in self.ids)
                or any(type(n) is not int or n < 0 for n in self.byte_lengths)
                or not isinstance(self.raw_sha256, str) or len(self.raw_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.raw_sha256)):
            raise ValueError("Invalid encoded document, byte lengths or raw SHA-256")


@dataclass(frozen=True)
class Window:
    inputs: tuple[int, ...]
    targets: tuple[int, ...]
    byte_lengths: tuple[int, ...]
    score_from: int
    first_target: int


def windows(document: Document, config) -> Iterator[Window]:
    """Score every target once; overlapping inputs are context only."""
    if (document.ids[0] != config.eos_id or document.ids[-1] != config.eos_id
            or document.byte_lengths[0] != 0 or document.byte_lengths[-1] != 0
            or config.eos_id in document.ids[1:-1]):
        raise ValueError("Require zero-byte prefix/closing EOS and ordinary text between")
    for first in range(1, len(document.ids), config.stride):
        end = min(first + config.stride - 1, len(document.ids) - 1)
        left = max(0, end - config.context)
        yield Window(document.ids[left:end], document.ids[left+1:end+1],
                     document.byte_lengths[left+1:end+1], first-left-1, first)


def bits_per_byte(nll: float, raw_bytes: int) -> float:
    if (type(raw_bytes) is not int or raw_bytes <= 0
            or not math.isfinite(nll) or nll < 0):
        raise ValueError("BPB needs finite nonnegative NLL and positive raw bytes")
    return nll/(math.log(2)*raw_bytes)


@torch.no_grad()
def evaluate(model, documents: Iterable[Document], config,
             *, expected_bytes: int | None = None) -> dict:
    """FP32 forward/CE, FP64 loss reduction, complete per-document coverage."""
    if model.config.context < config.context:
        raise ValueError("Model context is shorter than the scoring context")
    if any(t.is_floating_point() and t.dtype != torch.float32
           for t in (*model.parameters(), *model.buffers())):
        raise ValueError("Validation requires FP32 floating-point model state")
    device = next(model.parameters()).device
    previous = model.training
    nll, tokens, raw_bytes, document_count = 0.0, 0, 0, 0
    model.eval()
    try:
        with torch.autocast(device.type, enabled=False):
            for document in documents:
                count, byte_count = 0, 0
                for window in windows(document, config):
                    x = torch.tensor([window.inputs], dtype=torch.int64, device=device)
                    targets = torch.tensor(window.targets[window.score_from:], device=device)
                    logits = model(x)[0, window.score_from:]
                    losses = F.cross_entropy(logits, targets, reduction="none").double()
                    if not torch.isfinite(losses).all():
                        raise FloatingPointError("Nonfinite validation loss")
                    nll += losses.sum().item()
                    count += len(targets)
                    byte_count += sum(window.byte_lengths[window.score_from:])
                if (count, byte_count) != (len(document.ids)-1, sum(document.byte_lengths)):
                    raise RuntimeError("Per-document target/byte coverage mismatch")
                tokens += count
                raw_bytes += byte_count
                document_count += 1
    finally:
        model.train(previous)
    if expected_bytes is not None and raw_bytes != expected_bytes:
        raise ValueError(f"Validation byte coverage {raw_bytes} != expected {expected_bytes}")
    return {"bpb": bits_per_byte(nll, raw_bytes), "nll": nll, "tokens": tokens,
            "bytes": raw_bytes, "documents": document_count, "token_nll": nll/tokens,
            "protocol": asdict(config)}
