"""FineWeb acquisition, deterministic preparation and read-only verification.

Only explicit inspection/download stages access the network.
"""
from __future__ import annotations

from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import urllib.request

import numpy as np
import pyarrow.parquet as pq

from .runtime import file_hash, read_json, sha256, source_identity, write_json
from .data import TrainingData, DocumentPrefix, ShardedDocuments, write_shards
from .evaluate import windows
from .tokenizer import NestedBPE, FineWebEvaluationConfig, tokenizer_identity, train_family


# Generic file/hash/JSON helpers
# ------------------------------------------------------------------------
# Shared implementations are imported from runtime for independent data/tokenizer use.


# FineWeb source inspection and download
# ------------------------------------------------------------------------
API = "https://huggingface.co/api/datasets/"
HUB = "https://huggingface.co/datasets/"


def inspect_source(root: Path, config: dict):
    if (root / "selection.json").exists():
        raise FileExistsError("Source provenance is frozen after selection; use offline verify")
    source = config["source"]
    base = API + source["dataset"]

    def fetch(url):
        with urllib.request.urlopen(url, timeout=60) as response:
            return json.load(response)

    metadata = fetch(base + "/revision/" + source["revision"])
    if metadata["sha"] != source["revision"]:
        raise ValueError("Source revision mismatch")
    configs = metadata["cardData"]["configs"]
    selected = next(c for c in configs if c["config_name"] == source["configuration"])
    if selected["data_files"] != [{"split": "train", "path": "sample/10BT/*"}]:
        raise ValueError("Unexpected configuration paths; review required")
    files = fetch(base + "/tree/" + source["revision"] + "/sample/10BT?limit=100")
    # The pinned configuration contains 15 files, so no omitted pagination.
    if len(files) != 15:
        raise ValueError("Source listing changed or requires pagination")
    remote = {f["path"]: f for f in files}
    manifest = dict(source, available_configurations=[c["config_name"] for c in configs], files=[])
    for expected in source["files"]:
        actual = remote[expected["path"]]
        if (actual["size"], actual["lfs"]["oid"]) != (expected["size"], expected["sha256"]):
            raise ValueError("Published source size/SHA-256 mismatch")
        url = HUB + source["dataset"] + "/resolve/" + source["revision"] + "/" + expected["path"]
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
            if response.status != 200 or int(response.headers["Content-Length"]) != expected["size"]:
                raise ValueError("Source inaccessible or size mismatch")
        manifest["files"].append(dict(expected, url=url))
    manifest.update(network_estimate_bytes=sum(f["size"] for f in source["files"]),
                    disk_estimate_bytes=2_000_000_000,
                    scope="bounded pinned files; not global random sampling")
    if manifest["network_estimate_bytes"] > 2_000_000_000:
        raise PermissionError("Download exceeds 2 GB; separate approval required")
    (root / "source").mkdir(parents=True, exist_ok=True)
    write_json(root / "source" / "dataset_api.json", metadata)
    write_json(root / "source" / "config_files.json", files)
    write_json(root / "source_manifest.json", manifest)
    return manifest


def check_source(root: Path, config: dict):
    manifest = read_json(root / "source_manifest.json")
    expected = config["source"]
    if expected["dataset"] != "HuggingFaceFW/fineweb":
        raise ValueError("Only standard FineWeb is authorized")
    if any(manifest[k] != expected[k] for k in ("dataset", "configuration", "revision")):
        raise ValueError("Source manifest identity mismatch")
    files = read_json(root / "source" / "config_files.json")
    remote = {f["path"]: f for f in files}
    metadata = read_json(root / "source" / "dataset_api.json")
    if metadata["sha"] != expected["revision"]:
        raise ValueError("Repository metadata revision mismatch")
    selected = next(c for c in metadata["cardData"]["configs"]
                    if c["config_name"] == expected["configuration"])
    if selected["data_files"] != [{"split": "train", "path": "sample/10BT/*"}]:
        raise ValueError("Source configuration mismatch")
    if len(manifest["files"]) != len(expected["files"]):
        raise ValueError("Source file inventory differs")
    for f, e in zip(manifest["files"], expected["files"]):
        if any(f[k] != e[k] for k in ("path", "size", "sha256")):
            raise ValueError("Source file identity differs")
        if (remote[e["path"]]["size"], remote[e["path"]]["lfs"]["oid"]) != (e["size"], e["sha256"]):
            raise ValueError("Published metadata differs")
        if f["url"] != HUB + expected["dataset"] + "/resolve/" + expected["revision"] + "/" + e["path"]:
            raise ValueError("Unpinned source URL")
    return manifest


def download_source(root: Path, config: dict):
    manifest = check_source(root, config)
    estimate = sum(f["size"] for f in manifest["files"])
    if estimate > 2_000_000_000:
        raise PermissionError("Download exceeds 2 GB; separate approval required")
    if shutil.disk_usage(root).free < manifest["disk_estimate_bytes"]:
        raise OSError("Insufficient space for preparation estimate")
    for f in manifest["files"]:
        path = root / "source" / Path(f["path"]).name
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(f["url"], timeout=120) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, 8 << 20)
            if partial.stat().st_size != f["size"] or file_hash(partial) != f["sha256"]:
                raise ValueError("Downloaded source hash/size mismatch")
            partial.rename(path)
        if path.stat().st_size != f["size"] or file_hash(path) != f["sha256"]:
            raise ValueError("Existing source hash/size mismatch")


# Raw-document representation
# ------------------------------------------------------------------------
def source_id(dataset: str, revision: str, path: str, row: int) -> str:
    if (len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision)
            or type(row) is not int or row < 0):
        raise ValueError("Require immutable revision and nonnegative row")
    return f"{dataset}@{revision}/{path}#row={row}"


def describe(raw: bytes, identifier: str, **metadata) -> dict:
    raw.decode("utf-8", errors="strict")
    return dict(metadata, source_id=identifier, raw_bytes=len(raw), raw_sha256=sha256(raw))


class RawDocuments:
    """One concatenated raw byte file; boundaries never insert separators."""
    def __init__(self, root: Path, records: list[dict]):
        self.root, self.records = Path(root), records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        if not 0 <= i < len(self):
            raise IndexError(i)
        d = self.records[i]
        with (self.root / "raw.bin").open("rb") as stream:
            stream.seek(d["raw_offset"])
            raw = stream.read(d["raw_bytes"])
        if len(raw) != d["raw_bytes"] or sha256(raw) != d["raw_sha256"]:
            raise ValueError("Raw document hash/length mismatch")
        return raw


# Deterministic document selection and splitting
# ------------------------------------------------------------------------
def selection_key(record: dict, seed: int, domain: str) -> str:
    return sha256(f"{seed}:{domain}:{record['source_id']}".encode("utf-8"))


def ordered_candidates(chunks, *, seed: int, validation_modulus: int):
    """Append new pinned files at the end; discard duplicate exact texts.

    Empty documents are retained once (dedup applies to them too). Membership
    hashes are domain-separated from ordering. All reserved validation-bucket
    documents are excluded from train, including unused validation candidates.
    """
    if validation_modulus < 2:
        raise ValueError("Need distinct train/validation buckets")
    seen_ids, seen_texts = set(), set()
    for chunk in chunks:
        for record in sorted(chunk, key=lambda d: (selection_key(d, seed, "order"), d["source_id"])):
            if record["source_id"] in seen_ids:
                raise ValueError("Duplicate source identifier")
            seen_ids.add(record["source_id"])
            if record["raw_sha256"] in seen_texts:
                continue
            seen_texts.add(record["raw_sha256"])
            validation = int(selection_key(record, seed, "split"), 16) % validation_modulus == 0
            yield dict(record, split="validation" if validation else "train")


def byte_prefix(records, target: int) -> list[dict]:
    if target <= 0:
        raise ValueError("Byte target must be positive")
    result, total = [], 0
    for record in records:
        result.append(dict(record, selection_order=len(result)))
        total += record["raw_bytes"]
        if total >= target:
            return result
    raise ValueError(f"Bounded source exhausted: {total} < {target} bytes")


def select(chunks, settings: dict) -> dict:
    records = list(ordered_candidates(chunks, seed=settings["seed"],
                                     validation_modulus=settings["validation_modulus"]))
    train = byte_prefix((d for d in records if d["split"] == "train"), settings["pilot_bytes"])
    # Validation is selected once from the original source batch and then frozen.
    validation = byte_prefix((d for d in records if d["split"] == "validation"), settings["validation_bytes"])
    smoke = byte_prefix(train, settings["smoke_bytes"])
    if len(smoke) > len(train):
        raise ValueError("Smoke must be a pilot prefix")
    assert not {d["raw_sha256"] for d in train} & {d["raw_sha256"] for d in validation}
    return {"train": train, "validation": validation, "smoke_documents": len(smoke)}


# Parquet scanning
# ------------------------------------------------------------------------
def source_rows(root, source):
    for f in source["files"]:
        path = root / "source" / Path(f["path"]).name
        if path.stat().st_size != f["size"] or file_hash(path) != f["sha256"]:
            raise ValueError("Source hash/size mismatch")
        parquet = pq.ParquetFile(path)
        if not {"text", "id"} <= set(parquet.schema_arrow.names):
            raise ValueError("FineWeb text/id schema unavailable")
        row = 0
        for batch in parquet.iter_batches(batch_size=2048, columns=["text", "id"]):
            for value in batch.to_pylist():
                if not isinstance(value["text"], str) or not isinstance(value["id"], str):
                    raise ValueError("Null/non-string FineWeb text/id")
                yield f["path"], row, value["id"], value["text"].encode("utf-8")
                row += 1


def scan(root, config):
    source = check_source(root, config)
    chunks = {f["path"]: [] for f in source["files"]}
    for path, row, native_id, raw in source_rows(root, source):
        identifier = source_id(source["dataset"], source["revision"], path, row)
        chunks[path].append(describe(raw, identifier, source_path=path, row=row, native_id=native_id))
    selected = select(list(chunks.values()), config["selection"])
    return selected, {p: len(c) for p, c in chunks.items()}


# Raw-text preparation
# ------------------------------------------------------------------------
def prepare_raw(root: Path, config: dict):
    if (root / "selection.json").exists() or (root / "raw.bin").exists():
        raise FileExistsError("Raw preparation already exists; use verify, never overwrite")
    selected, counts = scan(root, config)
    records = selected["train"] + selected["validation"]
    wanted = {(r["source_path"], r["row"]): r for r in records}
    with (root / "raw.bin").open("xb") as stream:
        for path, row, native_id, raw in source_rows(root, config["source"]):
            d = wanted.get((path, row))
            if d is not None:
                if (len(raw), sha256(raw), native_id) != (d["raw_bytes"], d["raw_sha256"], d["native_id"]):
                    raise ValueError("Source changed between passes")
                d["raw_offset"] = stream.tell()
                stream.write(raw)
    if any("raw_offset" not in r for r in records):
        raise ValueError("Missing selected source row")
    selected.update(protocol=config["protocol"], settings=config["selection"], source_rows=counts,
                    source_manifest_sha256=file_hash(root / "source_manifest.json"),
                    raw_sha256=file_hash(root / "raw.bin"))
    write_json(root / "selection.json", selected)
    return selected


def dataset_identity(root, split):
    return {"sha256": sha256((file_hash(root / "selection.json") + ":" + split).encode()),
            "selection_sha256": file_hash(root / "selection.json"),
            "source_sha256": file_hash(root / "source_manifest.json"), "split": split}


# Tokenizer preparation
# ------------------------------------------------------------------------
def prepare_tokenizers(root, config):
    selection = read_json(root / "selection.json")
    directory = root / "tokenizers"
    train_family(RawDocuments(root, selection["train"]), directory, config["tokenizer"])
    sample = [{k: d[k] for k in ("source_id", "raw_sha256", "raw_bytes", "selection_order")}
              for d in selection["train"]]
    write_json(directory / "training_sample.json", sample)
    write_json(directory / "protocol.json", {
        "protocol": "fineweb_pilot_train_bpe16k_prefix8k_v1", "config": config["tokenizer"],
        "sample_sha256": file_hash(directory / "training_sample.json"),
        "selection_sha256": file_hash(root / "selection.json"),
        "documents": len(sample), "bytes": sum(d["raw_bytes"] for d in sample),
        "artifacts": {name: file_hash(directory / name) for name in ("base.json", "expanded.json", "nested.json")},
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ("tokenizers", "numpy", "pyarrow")},
        **source_identity(),
        "replay": "Two independent full-sample fits have identical parsed artifacts; single thread",
        "future": "32K/64K require a new declared family/sample protocol; retrain maximum vocabulary once and derive prefixes. No claim of fitted-16K extension."})


# Token-shard preparation
# ------------------------------------------------------------------------
def prepare_shards(root, config):
    selection = read_json(root / "selection.json")
    pair = NestedBPE.load(root / "tokenizers")
    for expanded, size in ((False, 8192), (True, 16384)):
        for split in ("train", "validation"):
            write_shards(root / "shards" / str(size) / split,
                         RawDocuments(root, selection[split]), selection[split], pair,
                         expanded=expanded, tokenizer=tokenizer_identity(root / "tokenizers", expanded),
                         dataset=dataset_identity(root, split), **config["shards"])
    write_json(root / "smoke.json", {"protocol": "pilot_prefix_view_v1",
               "documents": selection["smoke_documents"], "storage": "view of pilot train shards",
               "source_ids": [d["source_id"] for d in selection["train"][:selection["smoke_documents"]]]})


# Dataset statistics
# ------------------------------------------------------------------------
def percentiles(values):
    return {str(p): float(np.percentile(values, p)) for p in (0, 25, 50, 75, 90, 95, 99, 100)}


def statistics(documents, *, context, stride):
    vocab = documents.metadata["vocab_size"] if isinstance(documents, ShardedDocuments) else documents.documents.metadata["vocab_size"]
    frequency = np.zeros(vocab, dtype=np.int64)
    sizes, lengths, empty, windows_count = [], [], 0, 0
    for d in documents:
        sizes.append(sum(d.byte_lengths))
        lengths.append(len(d.ids)-2)
        empty += len(d.ids) == 2
        # Count one closing EOS as a corpus token; the prefix is context only.
        frequency += np.bincount(d.ids[1:], minlength=vocab)
        windows_count += math.ceil((len(d.ids)-1)/stride)
    raw_bytes, tokens = sum(sizes), int(frequency.sum())
    return {"raw_bytes": raw_bytes, "documents": len(sizes), "text_tokens": sum(lengths),
            "tokens_including_closing_eos": tokens, "eos_count": int(frequency[0]),
            "empty_documents": empty, "tokens_per_byte": tokens/raw_bytes, "bytes_per_token": raw_bytes/tokens,
            "raw_document_byte_percentiles": percentiles(sizes), "text_token_length_percentiles": percentiles(lengths),
            "vocabulary_used": int((frequency > 0).sum()), "vocabulary_utilization": float((frequency > 0).mean()),
            "token_frequencies": frequency.tolist(),
            "rare_token_types": {str(n): int(((frequency > 0) & (frequency <= n)).sum()) for n in (1, 5, 10, 100)},
            "unused_token_types": int((frequency == 0).sum()),
            "long_documents": {"over_context": sum(n > context for n in lengths),
                               "over_4096_tokens": sum(n > 4096 for n in lengths),
                               "over_65536_bytes": sum(n > 65536 for n in sizes),
                               "maximum_bytes": max(sizes), "maximum_text_tokens": max(lengths)},
            "estimated_training_windows": windows_count, "context": context, "stride": stride}


# Dataset verification and provenance
# ------------------------------------------------------------------------
class CodeProvenanceMismatch(ValueError):
    """Artifact integrity passed, but its recorded producing code differs."""


def implementation_fingerprints():
    project = Path(__file__).resolve().parents[1]
    paths = ["vocab_lm/" + name + ".py" for name in
             ("runtime", "tokenizer", "data", "evaluate", "fineweb")]
    paths += ["scripts/prepare_fineweb.py", "configs/fineweb.yaml"]
    return {p: file_hash(project / p) for p in paths}


def verify(root, config, *, publish=False, allow_code_provenance_mismatch=False):
    """Read-only replay by default; publishing is for a new generation only."""
    if publish and (root / "artifact_hashes.json").exists():
        raise FileExistsError("Never replace an existing artifact seal")
    if (root / "artifact_hashes.json").exists():
        sealed = read_json(root / "artifact_hashes.json")
        inventory = {p.relative_to(root).as_posix() for p in root.rglob("*")
                     if p.is_file() and p.name != "artifact_hashes.json"}
        if inventory != set(sealed):
            raise ValueError("Prepared artifact inventory differs from seal")
        for relative, record in sealed.items():
            path = root / relative
            if path.stat().st_size != record["bytes"] or file_hash(path) != record["sha256"]:
                raise ValueError(f"Prepared artifact hash differs: {relative}")
    fingerprints = implementation_fingerprints()
    previous = read_json(root / "verification.json") if (root / "verification.json").exists() else None
    recorded = previous.get("implementation_sha256") if previous else None
    provenance = {"status": "unrecorded" if recorded is None else
                  "match" if recorded == fingerprints else "mismatch",
                  "recorded": recorded, "current": fingerprints}
    if provenance["status"] != "match" and not (publish or allow_code_provenance_mismatch):
        raise CodeProvenanceMismatch(
            "Code provenance differs or is unrecorded; existing artifact seal checks passed. "
            "Review provenance, then explicitly allow read-only compatibility replay.")
    source = check_source(root, config)
    selection = read_json(root / "selection.json")
    if (selection["source_manifest_sha256"] != file_hash(root / "source_manifest.json")
            or selection["settings"] != config["selection"]
            or selection["raw_sha256"] != file_hash(root / "raw.bin")):
        raise ValueError("Selection/source/raw identity changed")
    replay, counts = scan(root, config)
    for split in ("train", "validation"):
        saved = [{k: v for k, v in d.items() if k != "raw_offset"} for d in selection[split]]
        if replay[split] != saved:
            raise ValueError("Deterministic selection replay differs")
    if counts != selection["source_rows"] or replay["smoke_documents"] != selection["smoke_documents"]:
        raise ValueError("Source/smoke prefix changed")
    raw_total = sum(d["raw_bytes"] for s in ("train", "validation") for d in selection[s])
    intervals = sorted((d["raw_offset"], d["raw_bytes"]) for s in ("train", "validation") for d in selection[s])
    offset = 0
    for start, size in intervals:
        if start != offset:
            raise ValueError("Raw offsets overlap or leave gaps")
        offset += size
    if offset != raw_total or (root / "raw.bin").stat().st_size != raw_total:
        raise ValueError("Raw byte totals differ")
    train_ids = {d["source_id"] for d in selection["train"]}
    valid_ids = {d["source_id"] for d in selection["validation"]}
    if train_ids & valid_ids:
        raise ValueError("Train/validation overlap")
    protocol = read_json(root / "tokenizers" / "protocol.json")
    for name, digest in protocol["artifacts"].items():
        if file_hash(root / "tokenizers" / name) != digest:
            raise ValueError("Tokenizer artifact mismatch")
    if file_hash(root / "tokenizers" / "training_sample.json") != protocol["sample_sha256"]:
        raise ValueError("Tokenizer sample hash differs")
    sample = read_json(root / "tokenizers" / "training_sample.json")
    if sample != [{k: d[k] for k in ("source_id", "raw_sha256", "raw_bytes", "selection_order")}
                  for d in selection["train"]]:
        raise ValueError("Tokenizer sample is not exactly pilot train")
    pair = NestedBPE.load(root / "tokenizers")
    pair.verify()
    evaluation = FineWebEvaluationConfig(**config["validation"])
    stats, coverage = {}, {}
    for expanded, size in ((False, 8192), (True, 16384)):
        table = [pair.token_bytes(i, expanded=expanded) for i in range(size)]
        stats[str(size)], coverage[str(size)] = {}, {}
        for split in ("train", "validation"):
            identity = tokenizer_identity(root / "tokenizers", expanded)
            documents = ShardedDocuments(root / "shards" / str(size) / split,
                         tokenizer_sha256=identity["sha256"], dataset_sha256=dataset_identity(root, split)["sha256"])
            if documents.metadata["byte_lengths"] != list(map(len, table)):
                raise ValueError("Shard token-byte table differs from tokenizer")
            if len(documents) != len(selection[split]):
                raise ValueError("Cross-tokenizer document count mismatch")
            raws = RawDocuments(root, selection[split])
            targets, byte_count = 0, 0
            for i, d in enumerate(documents):
                record = documents.records[i]
                if any(record[k] != selection[split][i][k] for k in ("source_id", "raw_sha256", "raw_bytes", "selection_order")):
                    raise ValueError("Cross-tokenizer document identity mismatch")
                raw = raws[i]
                if b"".join(table[t] for t in d.ids) != raw:
                    raise ValueError("Exact byte reconstruction differs")
                # Independently verify encoding and serialized sequence.
                if tuple([0] + pair.encode_document(raw, expanded=expanded)) != d.ids:
                    raise ValueError("Token sequence differs from tokenizer")
                if split == "validation":
                    next_target, doc_bytes = 1, 0
                    for w in windows(d, evaluation):
                        if w.first_target != next_target:
                            raise ValueError("Missing/repeated validation target")
                        next_target += len(w.targets)-w.score_from
                        doc_bytes += sum(w.byte_lengths[w.score_from:])
                    if next_target != len(d.ids) or doc_bytes != len(raw):
                        raise ValueError("Validation coverage differs")
                    targets += next_target-1
                    byte_count += doc_bytes
            stats[str(size)][split] = statistics(documents, **{k: config["loader"][k] for k in ("context", "stride")})
            if split == "train":
                smoke = DocumentPrefix(documents, selection["smoke_documents"])
                stats[str(size)]["smoke"] = statistics(smoke, **{k: config["loader"][k] for k in ("context", "stride")})
                data = TrainingData(documents, **config["loader"], tokenizer=identity, dataset=dataset_identity(root, split))
                for _ in range(17):
                    data.next_example()
                state = data.state_dict()
                expected = [data.next_example() for _ in range(37)]
                restored = TrainingData(documents, **config["loader"], tokenizer=identity, dataset=dataset_identity(root, split))
                restored.load_state_dict(state)
                if expected != [restored.next_example() for _ in range(37)]:
                    raise ValueError("Interrupted loader continuation differs")
            coverage[str(size)][split] = {"documents": len(documents), "raw_bytes": stats[str(size)][split]["raw_bytes"],
                       "validation_targets": targets if split == "validation" else None,
                       "validation_scored_bytes": byte_count if split == "validation" else None}
    stats["savings"] = {split: {"token_reduction_fraction": 1-stats["16384"][split]["tokens_including_closing_eos"]/stats["8192"][split]["tokens_including_closing_eos"],
                      "token_count_ratio_8k_over_16k": stats["8192"][split]["tokens_including_closing_eos"]/stats["16384"][split]["tokens_including_closing_eos"]}
                      for split in ("smoke", "train", "validation")}
    receipt = {"source": source, "coverage": coverage, "evaluation": asdict(evaluation),
               "checks": "source hashes, replay, disjointness, prefixes, raw hashes, nested artifacts, all token roundtrips, all validation targets, mmap, exact cursor continuation",
               "model_training_updates": 0, "model_evaluation": "none; coverage only, synthetic evaluator test separately",
               "implementation_sha256": fingerprints, "code_provenance": provenance,
               "statistics": stats}
    if publish:
        write_json(root / "statistics.json", stats)
        write_json(root / "verification.json", receipt)
        write_json(root / "artifact_hashes.json", {p.relative_to(root).as_posix():
            {"sha256": file_hash(p), "bytes": p.stat().st_size} for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "artifact_hashes.json"})
    return receipt


# Training-data construction
# ------------------------------------------------------------------------
def load_training_data(root: Path, config: dict, *, vocab_size: int, tier: str):
    """Construct loader/cursor only; never construct a model or optimizer."""
    if vocab_size not in {8192, 16384} or tier not in {"smoke", "pilot"}:
        raise ValueError("Only Gate 4A vocabularies and prepared tiers are available")
    identity = tokenizer_identity(root / "tokenizers", vocab_size == 16384)
    dataset = dataset_identity(root, "train")
    documents = ShardedDocuments(root / "shards" / str(vocab_size) / "train",
                tokenizer_sha256=identity["sha256"], dataset_sha256=dataset["sha256"])
    if tier == "smoke":
        count = read_json(root / "selection.json")["smoke_documents"]
        documents = DocumentPrefix(documents, count)
        dataset = dict(dataset, sha256=sha256((dataset["sha256"] + f":prefix:{count}").encode()),
                       tier=tier, documents=count)
    return TrainingData(documents, **config["loader"], tokenizer=identity, dataset=dataset)


