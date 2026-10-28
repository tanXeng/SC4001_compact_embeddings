"""CPU document-local windows with an exact next-example cursor.

The indexed Sequence may be a read-only mmap shard reader; documents stay on CPU.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import random
import torch
from collections.abc import Sequence
import struct
from pathlib import Path
import numpy as np
from .runtime import file_hash, read_json, write_json
from .evaluate import Document

PROTOCOL = "document_eos_disjoint_targets_shuffled_v1"


@dataclass
class Cursor:
    epoch: int = 0
    document: int = 0  # position in this epoch's permutation, not source ID
    first_target: int = 1


class TrainingData:
    def __init__(self, documents, *, context: int, stride: int, seed: int,
                 tokenizer: dict, dataset: dict):
        if not documents or not 1 <= stride <= context:
            raise ValueError("Need documents and 1 <= stride <= context")
        if dataset.get("split") != "train":
            raise ValueError("Training accepts only the train split")
        for identity in (tokenizer, dataset):
            digest = identity.get("sha256", "")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Require SHA-256 identities")
        digest = hashlib.sha256()
        total_bytes = 0
        for d in documents:
            if (d.ids[0] != 0 or d.ids[-1] != 0 or 0 in d.ids[1:-1]
                    or d.byte_lengths[0] or d.byte_lengths[-1]):
                raise ValueError("Require zero-byte EOS boundaries")
            # Include encoded IDs and attribution, not just caller-supplied hashes.
            digest.update(repr((d.ids, d.byte_lengths, d.raw_sha256)).encode())
            total_bytes += sum(d.byte_lengths)
        if not total_bytes:
            raise ValueError("Training corpus must contain raw text bytes")
        self.documents, self.context, self.stride, self.seed = documents, context, stride, seed
        self.identity = {"tokenizer": tokenizer, "dataset": dataset, "encoded_sha256": digest.hexdigest(),
                         "protocol": PROTOCOL, "context": context, "stride": stride, "seed": seed}
        self.cursor = Cursor()
        self._epoch = None

    def state_dict(self):
        return {"identity": self.identity, "cursor": asdict(self.cursor)}

    def load_state_dict(self, state):
        if state["identity"] != self.identity:
            raise ValueError("Training data identity/protocol changed")
        c = Cursor(**state["cursor"])
        if c.epoch < 0 or not 0 <= c.document < len(self.documents):
            raise ValueError("Invalid data cursor")
        self.cursor, self._epoch = c, None
        self._order()
        d = self.documents[self.order[c.document]]
        if c.first_target not in range(1, len(d.ids), self.stride):
            raise ValueError("Invalid next target")

    def _order(self):
        if self._epoch != self.cursor.epoch:
            self.order = list(range(len(self.documents)))
            random.Random(self.seed + self.cursor.epoch).shuffle(self.order)
            self._epoch = self.cursor.epoch

    def next_example(self):
        self._order()
        c = self.cursor
        source = self.order[c.document]
        d = self.documents[source]
        first = c.first_target
        end = min(first+self.stride-1, len(d.ids)-1)
        left = max(0, end-self.context)
        x = d.ids[left:end]
        y = [-100]*(first-left-1) + list(d.ids[first:end+1])
        raw_bytes = sum(d.byte_lengths[first:end+1])
        c.first_target += self.stride
        if c.first_target >= len(d.ids):
            c.first_target = 1
            c.document += 1
            if c.document == len(self.documents):
                c.document = 0
                c.epoch += 1
        return x, y, raw_bytes, (source, first)

    def batch(self, size):
        rows = [self.next_example() for _ in range(size)]
        length = max(len(row[0]) for row in rows)
        x = torch.zeros((size, length), dtype=torch.int64)
        y = torch.full_like(x, -100)
        for i, (inputs, targets, _, _) in enumerate(rows):
            x[i, :len(inputs)] = torch.tensor(inputs)
            y[i, :len(targets)] = torch.tensor(targets)
        return {"inputs": x, "targets": y, "tokens": int((y != -100).sum()),
                "bytes": sum(r[2] for r in rows), "examples": [r[3] for r in rows]}


# handling the shards
"""Versioned LE binary token shards, mmap reads and per-document provenance.

Header: <8sIIQ (magic, version, unsigned token width, token count), padded to
64 bytes. IDs follow; JSON metadata stores document offsets and identities.
Per-token byte lengths are looked up from a tokenizer-bound table, not repeated
for every corpus token. Each shard ends at a complete document boundary.
"""
MAGIC, VERSION, HEADER_BYTES = b"VLMTSHRD", 1, 64
HEADER = struct.Struct("<8sIIQ")


def token_dtype(vocab_size: int) -> np.dtype:
    if not 1 <= vocab_size <= 2**32:
        raise ValueError("Vocabulary outside uint32 range")
    return np.dtype("<u2" if vocab_size <= 2**16 else "<u4")


def write_shards(directory: Path, raw_documents, records, pair, *, expanded: bool,
                 tokenizer: dict, dataset: dict, max_tokens: int) -> dict:
    if directory.exists():
        raise FileExistsError(directory)
    if max_tokens < 1 or dataset.get("split") not in {"train", "validation"}:
        raise ValueError("Invalid shard settings/split")
    directory.mkdir(parents=True)
    model = pair.expanded if expanded else pair.base
    vocab_size = model.get_vocab_size()
    if len(raw_documents) != len(records) or not records:
        raise ValueError("Need matching nonempty raw/record sequences")
    if tokenizer.get("vocab_size") != vocab_size:
        raise ValueError("Tokenizer vocabulary identity differs")
    dtype = token_dtype(vocab_size)
    token_bytes = [pair.token_bytes(i, expanded=expanded) for i in range(vocab_size)]
    byte_table = list(map(len, token_bytes))
    shards, docs, count, stream, filename = [], [], 0, None, None

    def close():
        nonlocal stream
        if stream is None:
            return
        stream.seek(0)
        stream.write(HEADER.pack(MAGIC, VERSION, dtype.itemsize, count).ljust(HEADER_BYTES, b"\0"))
        stream.close()
        shards.append({"file": filename, "sha256": file_hash(directory / filename),
                       "tokens": count, "documents": docs})
        stream = None

    try:
        for i, record in enumerate(records):
            raw = raw_documents[i]
            ids = [0] + pair.encode_document(raw, expanded=expanded)
            if (any(t < 0 or t >= vocab_size for t in ids) or 0 in ids[1:-1]
                    or b"".join(token_bytes[t] for t in ids) != raw):
                raise ValueError("Token range/EOS/byte reconstruction mismatch")
            if sum(byte_table[t] for t in ids) != record["raw_bytes"]:
                raise ValueError("Token raw-byte attribution differs")
            if stream is not None and count + len(ids) > max_tokens:
                close()
            if stream is None:
                count, docs = 0, []
                filename = f"tokens-{len(shards):05d}.bin"
                stream = (directory / filename).open("w+b")
                stream.write(bytes(HEADER_BYTES))
            docs.append({k: record[k] for k in ("source_id", "raw_sha256", "raw_bytes", "selection_order")})
            docs[-1].update(offset=count, length=len(ids), tokenizer_sha256=tokenizer["sha256"])
            stream.write(np.asarray(ids, dtype=dtype).tobytes())
            count += len(ids)
        close()
    finally:
        if stream is not None:
            stream.close()
    metadata = {"format": "vocab_lm_token_shards", "version": VERSION,
                "byte_order": "little", "dtype": dtype.str, "tokenizer": tokenizer,
                "dataset": dataset, "vocab_size": vocab_size, "byte_lengths": byte_table,
                "shards": shards}
    write_json(directory / "index.json", metadata)
    return metadata


class ShardedDocuments(Sequence):
    """CPU mmap sequence accepted directly by Gate 2 TrainingData/evaluate.

    Metadata is resident; tokens remain mapped. Only the requested complete
    document becomes Python tuples. Startup verifies every shard hash/range.
    """
    def __init__(self, directory: Path, *, tokenizer_sha256=None, dataset_sha256=None):
        directory = Path(directory)
        self.metadata = m = read_json(directory / "index.json")
        if (m["format"] != "vocab_lm_token_shards" or m["version"] != VERSION
                or m["byte_order"] != "little"):
            raise ValueError("Unsupported shard format")
        if (tokenizer_sha256 is not None and m["tokenizer"]["sha256"] != tokenizer_sha256
                or dataset_sha256 is not None and m["dataset"]["sha256"] != dataset_sha256):
            raise ValueError("Shard identity mismatch")
        if m["dataset"].get("split") not in {"train", "validation"}:
            raise ValueError("Unsupported split")
        dtype = token_dtype(m["vocab_size"])
        if m["dtype"] != dtype.str or len(m["byte_lengths"]) != m["vocab_size"]:
            raise ValueError("Invalid dtype/byte table")
        if m["byte_lengths"][0] != 0 or any(type(n) is not int or n <= 0 for n in m["byte_lengths"][1:]):
            raise ValueError("Invalid token byte lengths")
        self.maps, self.records, self.locations = [], [], []
        self._cached_index, self._cached_document = None, None
        for shard in m["shards"]:
            path = directory / shard["file"]
            if path.parent != directory or file_hash(path) != shard["sha256"]:
                raise ValueError("Shard hash/path mismatch")
            with path.open("rb") as stream:
                header = stream.read(HEADER_BYTES)
            magic, version, width, count = HEADER.unpack(header[:HEADER.size])
            if ((magic, version, width, count) != (MAGIC, VERSION, dtype.itemsize, shard["tokens"])
                    or header[HEADER.size:] != bytes(HEADER_BYTES-HEADER.size)
                    or path.stat().st_size != HEADER_BYTES + count*dtype.itemsize or count == 0):
                raise ValueError("Invalid shard header/length")
            data = np.memmap(path, mode="r", dtype=dtype, offset=HEADER_BYTES, shape=(count,))
            if int(data.max()) >= m["vocab_size"]:
                raise ValueError("Token ID out of range")
            self.maps.append(data)
            offset = 0
            for d in shard["documents"]:
                if (d["offset"] != offset or d["length"] < 2 or offset+d["length"] > count
                        or d["tokenizer_sha256"] != m["tokenizer"]["sha256"]
                        or d["selection_order"] != len(self.records)):
                    raise ValueError("Invalid document boundary/order/identity")
                self.records.append(d)
                self.locations.append((len(self.maps)-1, offset, offset+d["length"]))
                offset += d["length"]
            if offset != count:
                raise ValueError("Document boundaries do not cover shard")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self))) ]
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        if i == self._cached_index:
            return self._cached_document
        shard, start, end = self.locations[i]
        ids = tuple(map(int, self.maps[shard][start:end]))
        lengths = tuple(self.metadata["byte_lengths"][t] for t in ids)
        if ids[0] or ids[-1] or 0 in ids[1:-1] or sum(lengths) != self.records[i]["raw_bytes"]:
            raise ValueError("Document EOS/byte attribution mismatch")
        document = Document(ids, lengths, self.records[i]["raw_sha256"])
        self._cached_index, self._cached_document = i, document
        return document


class DocumentPrefix(Sequence):
    """A smoke view of pilot shards with no token or raw-text duplication."""
    def __init__(self, documents, count):
        if not 0 < count <= len(documents):
            raise ValueError("Invalid prefix")
        self.documents, self.count = documents, count

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        if not 0 <= i < self.count:
            raise IndexError(i)
        return self.documents[i]
