"""FineWeb acquisition, deterministic preparation and read-only verification.

Only explicit inspection/download stages access the network.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import urllib.request

import numpy as np
import pyarrow.parquet as pq

from .runtime import file_hash, read_json, sha256, write_json
from .data import ShardedDocuments, write_shards
from .tokenizer import NestedBPE, tokenizer_identity


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
                    disk_estimate_bytes=config.get('disk_estimate_bytes',2_000_000_000),
                    scope="bounded pinned files; not global random sampling")
    if manifest["network_estimate_bytes"] > config.get('download_limit_bytes',2_000_000_000):
        raise PermissionError("Download exceeds declared preparation limit")
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
    if estimate > config.get('download_limit_bytes',2_000_000_000):
        raise PermissionError("Download exceeds declared preparation limit")
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
    # An interrupted unpublished raw build may be restarted; final data never overwritten.
    with (root / "raw.partial").open("wb") as stream:
        for path, row, native_id, raw in source_rows(root, config["source"]):
            d = wanted.get((path, row))
            if d is not None:
                if (len(raw), sha256(raw), native_id) != (d["raw_bytes"], d["raw_sha256"], d["native_id"]):
                    raise ValueError("Source changed between passes")
                d["raw_offset"] = stream.tell()
                stream.write(raw)
    if any("raw_offset" not in r for r in records):
        raise ValueError("Missing selected source row")
    (root / "raw.partial").rename(root / "raw.bin")
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


# Token-shard preparation
# ------------------------------------------------------------------------


def verify_scaling_data(root, directory, config):
    """Only train/validation, complete ordered byte/token coverage for a declared family."""
    root, directory = Path(root), Path(directory)
    selection = read_json(root/'selection.json')
    if (selection['settings'] != config['selection'] or
            selection['source_manifest_sha256'] != file_hash(root/'source_manifest.json') or
            selection['raw_sha256'] != file_hash(root/'raw.bin')):
        raise ValueError('Prepared selection/source/raw identity mismatch')
    check_source(root, config)
    pair=NestedBPE.load(directory)
    protocol=read_json(directory/'protocol.json')
    for name,digest in protocol['artifacts'].items():
        if file_hash(directory/name)!=digest:
            raise ValueError('Frozen tokenizer hash mismatch')
    for key in ('raw_sha256','source_id'):
        if {d[key] for d in selection['train']} & {d[key] for d in selection['validation']}:
            raise ValueError('Train/validation overlap')
    hashes={name:file_hash(root/name) for name in ('selection.json','source_manifest.json','raw.bin')}
    coverage={}; frequencies={}
    for expanded,size in ((False,pair.base.get_vocab_size()),(True,pair.expanded.get_vocab_size())):
        pieces=[pair.token_bytes(i,expanded=expanded) for i in range(size)]
        coverage[str(size)]={}
        for split in ('train','validation'):
            folder=root/'shards'/str(size)/split
            docs=ShardedDocuments(folder,tokenizer_sha256=tokenizer_identity(directory,expanded)['sha256'],
                                  dataset_sha256=dataset_identity(root,split)['sha256'])
            records=selection[split]
            if len(docs)!=len(records):
                raise ValueError('Incomplete document coverage')
            counts=np.zeros(size,dtype=np.int64); total=0
            for d,r in zip(docs,records):
                if (d.ids[0]!=0 or d.ids[-1]!=0 or 0 in d.ids[1:-1] or d.raw_sha256!=r['raw_sha256'] or
                    sha256(b''.join(pieces[i] for i in d.ids))!=r['raw_sha256']):
                    raise ValueError('Document order/EOS/byte reconstruction mismatch')
                total+=sum(d.byte_lengths)
                counts+=np.bincount(d.ids[1:],minlength=size)
            if total!=sum(r['raw_bytes'] for r in records):
                raise ValueError('Raw-byte coverage mismatch')
            coverage[str(size)][split]=dict(documents=len(docs),raw_bytes=total,tokens=int(counts.sum()))
            if split=='train': frequencies[str(size)]=counts.tolist()
            hashes[(folder/'index.json').relative_to(root).as_posix()]=file_hash(folder/'index.json')
            for shard in docs.metadata['shards']:
                hashes[(folder/shard['file']).relative_to(root).as_posix()]=shard['sha256']
    return dict(format='fineweb_large_data_v1',hashes=hashes,coverage=coverage,training_counts=frequencies,
                tokenizer_hashes=protocol['artifacts'],sealed_test_access=False)


def prepare_scaling_data(root, directory, config):
    """Idempotent staged reuse of the existing acquisition/selection/shard pipeline."""
    root, directory = Path(root), Path(directory)
    if set(config['source']) != {'dataset','configuration','revision','files'} or config['source']['dataset']!='HuggingFaceFW/fineweb':
        raise ValueError('Only the declared standard FineWeb training source is supported')
    root.mkdir(parents=True,exist_ok=True)
    seal=root/'preparation.json'
    if seal.exists():
        old=read_json(seal)
        if old.get('preparation_config')!=config:
            raise ValueError('Prepared configuration mismatch; never replace an existing generation')
        result=verify_scaling_data(root,directory,config)
        if any(result[k]!=old[k] for k in result):
            raise ValueError('Prepared artifact seal mismatch')
        return old
    if not (root/'source_manifest.json').exists():
        # Reuse cached, pinned source metadata on this workstation; fresh clones inspect remotely.
        cached=Path('data/fineweb_v1')
        if (cached/'source/dataset_api.json').exists():
            (root/'source').mkdir(exist_ok=True)
            for name in ('dataset_api.json','config_files.json'):
                shutil.copyfile(cached/'source'/name,root/'source'/name)
            source=config['source']
            write_json(root/'source_manifest.json',dict(source,files=[dict(f,url=HUB+source['dataset']+'/resolve/'+source['revision']+'/'+f['path']) for f in source['files']],
                disk_estimate_bytes=config['disk_estimate_bytes']))
        else:
            inspect_source(root,config)
    check_source(root,config)
    canonical=dict(config['source'],files=[dict(f,url=HUB+config['source']['dataset']+'/resolve/'+config['source']['revision']+'/'+f['path']) for f in config['source']['files']],
                   disk_estimate_bytes=config['disk_estimate_bytes'])
    if read_json(root/'source_manifest.json')!=canonical:
        if (root/'selection.json').exists():
            raise ValueError('Frozen source manifest differs from canonical preparation')
        write_json(root/'source_manifest.json',canonical)
    for f in config['source']['files']:
        target=root/'source'/Path(f['path']).name
        cached=Path('data/vocab_scaling_large/source')/target.name
        if not cached.exists(): cached=Path('data/fineweb_v1/source')/target.name
        if not target.exists() and cached.exists() and file_hash(cached)==f['sha256']:
            # Hard link is local reuse only; it does not change either generation.
            import os
            os.link(cached,target)
    download_source(root,config)
    if not (root/'selection.json').exists():
        prepare_raw(root,config)
    selection=read_json(root/'selection.json')
    pair=NestedBPE.load(directory)
    for expanded,size in ((False,pair.base.get_vocab_size()),(True,pair.expanded.get_vocab_size())):
        for split in ('train','validation'):
            target=root/'shards'/str(size)/split
            if target.exists():
                # Existing complete stages are verified below; incomplete stages fail explicitly.
                if not (target/'index.json').exists():
                    raise ValueError(f'Incomplete shard stage {target}; preserve evidence and contact owner')
                continue
            write_shards(target,RawDocuments(root,selection[split]),selection[split],pair,
                expanded=expanded,tokenizer=tokenizer_identity(directory,expanded),dataset=dataset_identity(root,split),**config['shards'])
    result=verify_scaling_data(root,directory,config)
    result['preparation_config']=config
    write_json(seal,result)
    return result
