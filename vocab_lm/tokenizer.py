"""Nested byte-level BPE and the fixed FineWeb tokenizer family.

BPE operations are preserved from the pre-cleanup src/tokenizers.py.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable

from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers

from .runtime import file_hash


def byte_alphabet() -> dict[int, str]:
    """GPT-2/ByteLevel's reversible mapping, including incomplete UTF-8 bytes."""
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codes = visible.copy()
    extra = 0
    for value in range(256):
        if value not in visible:
            visible.append(value)
            codes.append(256 + extra)
            extra += 1
    return dict(zip(visible, map(chr, codes)))


BYTE_TO_SYMBOL = byte_alphabet()
SYMBOL_TO_BYTE = {symbol: value for value, symbol in BYTE_TO_SYMBOL.items()}


def symbol_bytes(symbols: str) -> bytes:
    """Invert byte symbols without ever decoding token fragments as UTF-8."""
    return bytes(SYMBOL_TO_BYTE[symbol] for symbol in symbols)


def _payload(tokenizer: Tokenizer) -> dict:
    return json.loads(tokenizer.to_str())


@dataclass
class NestedBPE:
    base: Tokenizer
    expanded: Tokenizer
    eos_token: str

    def __post_init__(self) -> None:
        # Raw literal occurrences of the special-token spelling must remain text.
        # EOS is inserted explicitly by encode_document, never inferred from text.
        self.base.encode_special_tokens = True
        self.expanded.encode_special_tokens = True
        self.verify()

    @property
    def eos_id(self) -> int:
        value = self.base.token_to_id(self.eos_token)
        if value is None:
            raise ValueError("Missing EOS")
        return value

    def verify(self) -> None:
        small, large = _payload(self.base), _payload(self.expanded)
        v0, v1 = small["model"]["vocab"], large["model"]["vocab"]
        if len(v0) >= len(v1):
            raise ValueError("Expanded vocabulary must be larger")
        if sorted(v0.values()) != list(range(len(v0))):
            raise ValueError("Base IDs are not contiguous")
        if sorted(v1.values()) != list(range(len(v1))):
            raise ValueError("Expanded IDs are not contiguous")
        if v0 != {token: idx for token, idx in v1.items() if idx < len(v0)}:
            raise ValueError("Vocabulary IDs are not nested")
        merges = small["model"]["merges"]
        if merges != large["model"]["merges"][:len(merges)]:
            raise ValueError("Base merges are not a prefix")
        if small["added_tokens"] != large["added_tokens"] or self.eos_id != 0:
            raise ValueError("EOS must be the only special token, with stable ID 0")
        if len(small["added_tokens"]) != 1 or self.expanded.token_to_id(self.eos_token) != 0:
            raise ValueError("Invalid special-token configuration")
        for symbol in BYTE_TO_SYMBOL.values():
            if symbol not in v0:
                raise ValueError("Base vocabulary is missing a byte")
        # Every retained merge must create exactly the next stable token ID.
        for model in (small["model"], large["model"]):
            for rank, (left, right) in enumerate(model["merges"]):
                if model["vocab"].get(left + right) != 257 + rank:
                    raise ValueError("Merge order does not match token IDs")

    def encode_document(self, raw: bytes, *, expanded: bool, append_eos: bool = True) -> list[int]:
        tokenizer = self.expanded if expanded else self.base
        ids = tokenizer.encode(raw.decode("utf-8"), add_special_tokens=False).ids
        return ids + [self.eos_id] if append_eos else ids

    def token_bytes(self, token_id: int, *, expanded: bool = True) -> bytes:
        tokenizer = self.expanded if expanded else self.base
        token = tokenizer.id_to_token(token_id)
        if token is None:
            raise ValueError(f"Unknown token ID {token_id}")
        return b"" if token_id == self.eos_id else symbol_bytes(token)

    def reconstruct(self, ids: Iterable[int], *, expanded: bool) -> bytes:
        return b"".join(self.token_bytes(idx, expanded=expanded) for idx in ids)

    def decompositions(self) -> list[list[int]]:
        """Apply base BPE directly to each new token's byte symbols.

        Bypassing pre-tokenization here avoids corrupting incomplete UTF-8 and
        avoids treating byte symbols as ordinary Unicode input a second time.
        Ragged lists preserve repetitions, order, and arbitrary lengths.
        """
        result: list[list[int]] = []
        for idx in range(self.base.get_vocab_size(), self.expanded.get_vocab_size()):
            symbols = self.expanded.id_to_token(idx)
            assert symbols is not None
            pieces = [piece.id for piece in self.base.model.tokenize(symbols)]
            if not pieces or any(piece == self.eos_id or piece >= self.base.get_vocab_size() for piece in pieces):
                raise ValueError(f"Invalid decomposition for {idx}")
            if self.reconstruct(pieces, expanded=False) != symbol_bytes(symbols):
                raise ValueError(f"Decomposition changed bytes for {idx}")
            result.append(pieces)
        return result

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.base.save(str(directory / "base.json"))
        self.expanded.save(str(directory / "expanded.json"))
        (directory / "nested.json").write_text(json.dumps({"eos_token": self.eos_token}) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> NestedBPE:
        settings = json.loads((directory / "nested.json").read_text(encoding="utf-8"))
        return cls(Tokenizer.from_file(str(directory / "base.json")),
                   Tokenizer.from_file(str(directory / "expanded.json")), settings["eos_token"])


def train_nested(documents: Iterable[bytes], *, base_vocab_size: int, expanded_vocab_size: int,
                 min_frequency: int, eos_token: str, add_prefix_space: bool = False,
                 use_regex: bool = True) -> NestedBPE:
    """Train expanded BPE once, then truncate its merge sequence and vocabulary."""
    if not 257 <= base_vocab_size < expanded_vocab_size:
        raise ValueError("Require 257 <= base size < expanded size")
    if add_prefix_space:
        raise ValueError("A synthetic prefix space violates exact byte reconstruction")
    if not eos_token or any(char in SYMBOL_TO_BYTE for char in eos_token) and len(eos_token) == 1:
        raise ValueError("EOS must not collide with a byte symbol")
    expanded = Tokenizer(models.BPE())
    expanded.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=use_regex)
    expanded.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=expanded_vocab_size, min_frequency=min_frequency, show_progress=False,
        special_tokens=[AddedToken(eos_token, special=True, normalized=False)],
        initial_alphabet=list(BYTE_TO_SYMBOL.values()),
    )
    expanded.train_from_iterator((raw.decode("utf-8") for raw in documents), trainer=trainer)
    if expanded.get_vocab_size() != expanded_vocab_size:
        raise ValueError("Training corpus cannot support the requested expanded vocabulary")
    payload = copy.deepcopy(_payload(expanded))
    payload["model"]["vocab"] = {token: idx for token, idx in payload["model"]["vocab"].items() if idx < base_vocab_size}
    payload["model"]["merges"] = payload["model"]["merges"][:base_vocab_size - 257]
    return NestedBPE(Tokenizer.from_str(json.dumps(payload)), expanded, eos_token)


@dataclass(frozen=True)
class FineWebEvaluationConfig:
    protocol: str = "fineweb_document_eos_c256_s128_v1"
    context: int = 256
    stride: int = 128
    eos_id: int = 0

    def __post_init__(self):
        if (self.protocol != "fineweb_document_eos_c256_s128_v1"
                or type(self.context) is not int or self.context != 256
                or type(self.stride) is not int or self.stride != 128
                or type(self.eos_id) is not int or self.eos_id != 0):
            raise ValueError("FineWeb v1 requires context 256, stride 128, EOS 0")


def train_family(raw_train, directory: Path, config: dict) -> NestedBPE:
    """Train once on the exact ordered training-only sample. Never use valid."""
    if directory.exists():
        raise FileExistsError(directory)
    config = dict(config)
    os.environ["TOKENIZERS_PARALLELISM"] = str(config.pop("parallelism")).lower()
    pair = train_nested(iter(raw_train), **config)
    pair.save(directory)
    # A second independent fit must reproduce merge order and shared IDs.
    replay = train_nested(iter(raw_train), **config)
    for a, b in ((pair.base, replay.base), (pair.expanded, replay.expanded)):
        if json.loads(a.to_str()) != json.loads(b.to_str()):
            raise RuntimeError("Tokenizer replay differs; nested construction is not reproducible")
    return pair


def tokenizer_identity(directory: Path, expanded: bool) -> dict:
    path = directory / ("expanded.json" if expanded else "base.json")
    return {"sha256": file_hash(path), "artifact": path.name,
            "vocab_size": Tokenizer.from_file(str(path)).get_vocab_size()}
