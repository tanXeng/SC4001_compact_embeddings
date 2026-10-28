"""Offline synthetic Gate 4A coverage; no optimizer/model fitting is allowed."""
from dataclasses import asdict
import copy
import json
import math
from pathlib import Path
import struct

import numpy as np
import pytest
import torch

from vocab_lm.tokenizer import NestedBPE
from vocab_lm.config import evaluation_config
from vocab_lm.fineweb import (RawDocuments, byte_prefix, describe, file_hash,
                                 ordered_candidates, read_json, select, sha256, source_id, write_json)
from vocab_lm.data import TrainingData
from vocab_lm.evaluate import Document, evaluate, windows
from vocab_lm.data import (HEADER, HEADER_BYTES, MAGIC, VERSION, DocumentPrefix,
                                 ShardedDocuments, token_dtype, write_shards)
from vocab_lm.tokenizer import FineWebEvaluationConfig, train_family


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Gate 4A prohibits model training")
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)


@pytest.fixture
def raw_fixture(corpus):
    return [b"", b" \t\r\n  ", "e\u0301 \u00e9 \U0001f98a \u6771\u4eac\n".encode(),
            b"literal <|endoftext|> stays ordinary text", *corpus,
            b"long document " * 500]


def candidates(raws, prefix="file-a"):
    return [describe(raw, source_id("HuggingFaceFW/fineweb", "a"*40, prefix, i))
            for i, raw in enumerate(raws)]


def test_source_ids():
    a = source_id("dataset", "a"*40, "path", 0)
    assert a == source_id("dataset", "a"*40, "path", 0)
    assert len({a, source_id("dataset", "a"*40, "path", 1),
                source_id("dataset", "b"*40, "path", 0),
                source_id("dataset", "a"*40, "other", 0)}) == 4
    with pytest.raises(ValueError):
        source_id("dataset", "main", "path", 0)


def test_selection_deterministic_nested_append_and_disjoint():
    chunks = [candidates([f"Document {i}".encode() for i in range(400)])]
    settings = dict(seed=42, validation_modulus=4, smoke_bytes=100, pilot_bytes=1000, validation_bytes=100)
    first = select(chunks, settings)
    reversed_rows = select([list(reversed(chunks[0]))], settings)
    assert first == reversed_rows
    bigger = select(chunks + [candidates([f"More {i}".encode() for i in range(200)], "file-b")],
                    dict(settings, pilot_bytes=3500))
    assert first["train"] == bigger["train"][:len(first["train"])]
    assert first["validation"] == bigger["validation"]
    assert first["train"][:first["smoke_documents"]] == byte_prefix(first["train"], settings["smoke_bytes"])
    for field in ("source_id", "raw_sha256"):
        assert not {r[field] for r in first["train"]} & {r[field] for r in first["validation"]}
    assert sum(r["raw_bytes"] for r in first["train"]) >= settings["pilot_bytes"]
    assert sum(r["raw_bytes"] for r in first["train"][:-1]) < settings["pilot_bytes"]
    assert first != select(chunks, dict(settings, seed=43))


def test_dedup_empty_and_bounded_source_exhaustion():
    chunk = candidates([b"", b"", b"same", b"same", b"different"])
    rows = list(ordered_candidates([chunk], seed=42, validation_modulus=2))
    assert len(rows) == 3
    assert sum(r["raw_bytes"] == 0 for r in rows) == 1
    with pytest.raises(ValueError, match="exhausted"):
        byte_prefix(rows, 10_000)
    with pytest.raises(ValueError, match="Duplicate source"):
        list(ordered_candidates([chunk, chunk], seed=42, validation_modulus=2))


def test_raw_utf8_exact_and_boundaries(raw_fixture, tmp_path):
    records = candidates(raw_fixture)
    offset = 0
    with (tmp_path / "raw.bin").open("wb") as stream:
        for raw, record in zip(raw_fixture, records):
            record["raw_offset"] = offset
            stream.write(raw)
            offset += len(raw)
    reader = RawDocuments(tmp_path, records)
    assert list(reader) == raw_fixture
    assert (tmp_path / "raw.bin").read_bytes() == b"".join(raw_fixture)
    records[1]["raw_sha256"] = "a"*64
    with pytest.raises(ValueError, match="hash"):
        reader[1]


def test_family_replay_nested_ids_and_roundtrips(raw_fixture, pair, tmp_path):
    # Small synthetic family exercises the exact production fit/replay route.
    config = dict(base_vocab_size=300, expanded_vocab_size=400, min_frequency=1,
                  eos_token=pair.eos_token, add_prefix_space=False, use_regex=True, parallelism=False)
    fitted = train_family(raw_fixture, tmp_path / "family", config)
    loaded = NestedBPE.load(tmp_path / "family")
    loaded.verify()
    small, large = json.loads(loaded.base.to_str()), json.loads(loaded.expanded.to_str())
    assert small["model"]["merges"] == large["model"]["merges"][:len(small["model"]["merges"])]
    for token, idx in loaded.base.get_vocab().items():
        assert loaded.expanded.token_to_id(token) == idx
    for expanded in (False, True):
        for raw in raw_fixture:
            ids = loaded.encode_document(raw, expanded=expanded)
            assert ids[-1] == 0 and 0 not in ids[:-1]
            assert loaded.reconstruct(ids, expanded=expanded) == raw
            assert loaded.encode_document(raw, expanded=expanded, append_eos=False) == ids[:-1]
        assert loaded.encode_document(b"", expanded=expanded) == [0]
    assert fitted.base.to_str() == loaded.base.to_str()
    with pytest.raises(FileExistsError):
        train_family(raw_fixture, tmp_path / "family", config)


@pytest.fixture
def prepared(raw_fixture, pair, tmp_path):
    records = [dict(r, selection_order=i) for i, r in enumerate(candidates(raw_fixture))]
    paths = []
    for expanded in (False, True):
        path = tmp_path / str(expanded)
        size = (pair.expanded if expanded else pair.base).get_vocab_size()
        identity = {"sha256": sha256((pair.expanded if expanded else pair.base).to_str().encode()),
                    "vocab_size": size}
        write_shards(path, raw_fixture, records, pair, expanded=expanded, tokenizer=identity,
                     dataset={"sha256": "b"*64, "split": "train"}, max_tokens=100)
        paths.append(path)
    return paths


def test_shards_mmap_serialization_boundaries_bytes_and_identities(prepared, pair, raw_fixture):
    readers = [ShardedDocuments(path) for path in prepared]
    assert readers[0].records[0]["source_id"] == readers[1].records[0]["source_id"]
    for expanded, reader in zip((False, True), readers):
        assert len(reader.maps) > 1
        assert all(isinstance(m, np.memmap) and not m.flags.writeable for m in reader.maps)
        for i, (raw, d) in enumerate(zip(raw_fixture, reader)):
            assert pair.reconstruct(d.ids, expanded=expanded) == raw
            assert sum(d.byte_lengths) == len(raw)
            assert d.ids[0] == d.ids[-1] == 0 and 0 not in d.ids[1:-1]
            assert d.byte_lengths[0] == d.byte_lengths[-1] == 0
            assert d.raw_sha256 == sha256(raw)
            assert readers[0].records[i]["source_id"] == readers[1].records[i]["source_id"]
        assert list(DocumentPrefix(reader, 3)) == [reader[i] for i in range(3)]
        with pytest.raises(ValueError, match="identity"):
            ShardedDocuments(prepared[int(expanded)], tokenizer_sha256="f"*64)


@pytest.mark.parametrize("change", ["payload", "range", "header", "offset", "order", "dtype", "bytes", "version", "hash"])
def test_corrupt_shards_rejected(prepared, change):
    path = prepared[0]
    index = read_json(path / "index.json")
    shard = index["shards"][0]
    token_path = path / shard["file"]
    if change in {"payload", "range", "header"}:
        with token_path.open("r+b") as stream:
            stream.seek(0 if change == "header" else HEADER_BYTES)
            stream.write(b"BADMAGIC" if change == "header" else struct.pack("<H", index["vocab_size"]))
        if change != "payload":
            shard["sha256"] = file_hash(token_path)
    elif change == "offset":
        shard["documents"][0]["offset"] += 1
    elif change == "order":
        shard["documents"][0]["selection_order"] += 1
    elif change == "dtype":
        index["dtype"] = ">u2"
    elif change == "bytes":
        shard["documents"][0]["raw_bytes"] += 1
    elif change == "version":
        index["version"] += 1
    elif change == "hash":
        shard["sha256"] = "f"*64
    write_json(path / "index.json", index)
    with pytest.raises(ValueError):
        list(ShardedDocuments(path))


def test_uint32_future_and_little_endian():
    assert token_dtype(65536).str == "<u2"
    assert token_dtype(65537).str == "<u4"
    assert np.asarray([0, 65535], dtype=token_dtype(65536)).tobytes() == b"\0\0\xff\xff"
    assert np.asarray([65536], dtype=token_dtype(65537)).tobytes() == b"\0\0\x01\0"
    for size in (0, 2**32+1):
        with pytest.raises(ValueError):
            token_dtype(size)


def make_data(documents):
    return TrainingData(documents, context=256, stride=128, seed=42,
        tokenizer=documents.metadata["tokenizer"], dataset=documents.metadata["dataset"])


def test_loader_cursor_exact_order_full_epoch_and_padding(prepared):
    documents = ShardedDocuments(prepared[0])
    data = make_data(documents)
    reference = make_data(documents)
    # Exactly one epoch, including a long document and an empty document.
    rows = []
    while reference.cursor.epoch == 0:
        rows.append(reference.next_example())
    assert sum(r[2] for r in rows) == sum(sum(d.byte_lengths) for d in documents)
    assert sum(sum(t != -100 for t in r[1]) for r in rows) == sum(len(d.ids)-1 for d in documents)
    assert rows == [data.next_example() for _ in rows]
    for position in (0, 1, 5, 17):
        data = make_data(documents)
        for _ in range(position):
            data.next_example()
        snapshot = copy.deepcopy(data.state_dict())
        restored = make_data(documents)
        restored.load_state_dict(snapshot)
        # Runs across document and epoch boundaries, comparing masks and bytes.
        for _ in range(len(rows)*2):
            assert data.next_example() == restored.next_example()
        a, b = data.batch(7), restored.batch(7)
        assert torch.equal(a["inputs"], b["inputs"])
        assert torch.equal(a["targets"], b["targets"])
        assert a["bytes"] == b["bytes"]
    other = make_data(ShardedDocuments(prepared[1]))
    with pytest.raises(ValueError, match="identity"):
        other.load_state_dict(snapshot)


def test_validation_exact_coverage_and_uniform_bpb(prepared):
    documents = ShardedDocuments(prepared[1])
    config = FineWebEvaluationConfig()
    for d in documents:
        scored, byte_count = [], 0
        for w in windows(d, config):
            assert 0 < len(w.inputs) <= 256
            scored.extend(range(w.first_target, w.first_target + len(w.targets)-w.score_from))
            byte_count += sum(w.byte_lengths[w.score_from:])
        assert scored == list(range(1, len(d.ids)))
        assert byte_count == sum(d.byte_lengths)

    class Uniform(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = config

        def forward(self, inputs):
            return torch.zeros((*inputs.shape, documents.metadata["vocab_size"]))

    model = Uniform()
    raw_bytes = sum(sum(d.byte_lengths) for d in documents)
    result = evaluate(model, documents, config, expected_bytes=raw_bytes)
    targets = sum(len(d.ids)-1 for d in documents)
    assert result["tokens"] == targets
    assert result["bytes"] == raw_bytes
    assert result["bpb"] == pytest.approx(targets*math.log2(documents.metadata["vocab_size"])/raw_bytes, rel=1e-6)
    assert model.training


def test_fineweb_protocol_is_fixed():
    expected = dict(protocol="fineweb_document_eos_c256_s128_v1", context=256, stride=128, eos_id=0)
    assert asdict(evaluation_config({})) == expected
    assert asdict(evaluation_config(expected)) == expected
    for change in (dict(protocol="unsupported"), dict(context=128), dict(stride=64),
                   dict(eos_id=1), dict(context=True)):
        with pytest.raises(ValueError):
            evaluation_config({**expected, **change})


@pytest.fixture
def parquet_source(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = tmp_path / "fineweb"
    (root / "source").mkdir(parents=True)
    path = "sample/10BT/fixture.parquet"
    texts = [f" \tDocument {i} e\u0301 \U0001f98a\r\n\n " for i in range(200)] + [""]
    pq.write_table(pa.table({"text": texts, "id": [f"native-{i}" for i in range(len(texts))]}),
                   root / "source" / "fixture.parquet")
    f = dict(path=path, size=(root / "source" / "fixture.parquet").stat().st_size,
             sha256=file_hash(root / "source" / "fixture.parquet"))
    source = dict(dataset="HuggingFaceFW/fineweb", configuration="sample-10BT", revision="a"*40, files=[f])
    write_json(root / "source_manifest.json", dict(source, files=[dict(f, url=
        "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/" + "a"*40 + "/" + path)]))
    write_json(root / "source" / "dataset_api.json", {"sha": "a"*40, "cardData": {
        "configs": [{"config_name": "sample-10BT", "data_files": [{"split": "train", "path": "sample/10BT/*"}]}]}})
    write_json(root / "source" / "config_files.json", [{"path": path, "size": f["size"], "lfs": {"oid": f["sha256"]}}])
    config = dict(source=source, protocol="fineweb_v1", selection=dict(seed=42, validation_modulus=4,
                  smoke_bytes=100, pilot_bytes=1000, validation_bytes=100))
    return root, config, texts


def test_parquet_preparation_raw_preservation_and_replay(parquet_source):
    from vocab_lm.fineweb import source_rows, prepare_raw, scan
    root, config, texts = parquet_source
    rows = list(source_rows(root, config["source"]))
    assert [raw for _, _, _, raw in rows] == [t.encode("utf-8") for t in texts]
    selected = prepare_raw(root, config)
    replay, counts = scan(root, config)
    for split in ("train", "validation"):
        for d, raw in zip(selected[split], RawDocuments(root, selected[split])):
            assert raw == texts[d["row"]].encode("utf-8")
            assert d["native_id"] == f"native-{d['row']}"
            assert d["raw_sha256"] == sha256(raw)
        assert [{k: v for k, v in d.items() if k != "raw_offset"} for d in selected[split]] == replay[split]
    assert (root / "raw.bin").stat().st_size == sum(d["raw_bytes"] for s in ("train", "validation") for d in selected[s])
    with pytest.raises(FileExistsError):
        prepare_raw(root, config)


@pytest.mark.parametrize("corruption", ["revision", "url", "source"])
def test_source_provenance_rejects_changes(parquet_source, corruption):
    from vocab_lm.fineweb import scan
    root, config, _ = parquet_source
    if corruption == "source":
        with (root / "source" / "fixture.parquet").open("ab") as stream:
            stream.write(b"changed")
    else:
        manifest = read_json(root / "source_manifest.json")
        if corruption == "revision":
            manifest["revision"] = "main"
        else:
            manifest["files"][0]["url"] = manifest["files"][0]["url"].replace("a"*40, "main")
        write_json(root / "source_manifest.json", manifest)
    with pytest.raises(ValueError):
        scan(root, config)


def test_uint32_full_shard_roundtrip(tmp_path):
    class Model:
        def get_vocab_size(self):
            return 65537

    class Pair:
        base = expanded = Model()

        def token_bytes(self, i, **kwargs):
            return b"" if i == 0 else b"x"

        def encode_document(self, raw, **kwargs):
            return [65536, 0]

    record = dict(describe(b"x", "synthetic"), selection_order=0)
    path = tmp_path / "uint32"
    write_shards(path, [b"x"], [record], Pair(), expanded=True,
        tokenizer={"sha256": "a"*64, "vocab_size": 65537},
        dataset={"sha256": "b"*64, "split": "train"}, max_tokens=100)
    reader = ShardedDocuments(path)
    assert reader.metadata["dtype"] == "<u4"
    assert reader[0].ids == (0, 65536, 0)
    assert reader[0].byte_lengths == (0, 1, 0)


def test_budget_and_cursor_checkpoint_without_updates(prepared, tmp_path):
    from vocab_lm.train import Counters, budget_value, overshoot
    data = make_data(ShardedDocuments(prepared[0]))
    batch = data.batch(7)
    c = Counters(scored_tokens=batch["tokens"], raw_bytes=batch["bytes"], microsteps=1)
    assert budget_value(c, "optimizer_steps") == 0
    assert budget_value(c, "raw_bytes") == batch["bytes"]
    assert overshoot(c, "raw_bytes", 1) == max(0, batch["bytes"]-1)
    write_json(tmp_path / "cursor.json", data.state_dict())
    restored = make_data(ShardedDocuments(prepared[0]))
    restored.load_state_dict(read_json(tmp_path / "cursor.json"))
    assert [data.next_example() for _ in range(50)] == [restored.next_example() for _ in range(50)]
    assert c.attempted_updates == c.successful_updates == 0
