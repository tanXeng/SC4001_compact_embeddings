"""Standalone configuration/CLI tests with local synthetic byte-BPE shards only."""
from dataclasses import asdict
from pathlib import Path
import json
import sys

import pytest
import torch
import yaml

from scripts.train_dense import main, load_data
from vocab_lm.checkpoint import load_training_checkpoint
from vocab_lm.config import load_config
from vocab_lm.fineweb import describe, read_json, sha256, write_json
from vocab_lm.fineweb import (CodeProvenanceMismatch, dataset_identity,
                              implementation_fingerprints, verify)
from vocab_lm.data import write_shards
from vocab_lm.tokenizer import tokenizer_identity


@pytest.fixture
def recipe(tmp_path, pair, corpus):
    root = tmp_path / "data"
    pair.save(root / "tokenizers")
    raws = {"train": corpus, "validation": ["validation e\u0301 \U0001f98a\n".encode()]}
    selection = {split: [dict(describe(raw, f"{split}-{i}"), selection_order=i)
                         for i, raw in enumerate(values)] for split, values in raws.items()}
    selection["smoke_documents"] = 1
    write_json(root / "selection.json", selection)
    write_json(root / "source_manifest.json", {"synthetic": True})
    vocab = pair.expanded.get_vocab_size()
    for split, values in raws.items():
        write_shards(root / "shards" / str(vocab) / split, values, selection[split], pair,
                     expanded=True, tokenizer=tokenizer_identity(root / "tokenizers", True),
                     dataset=dataset_identity(root, split), max_tokens=1000)
    settings = yaml.safe_load(Path("configs/cfg.yaml").read_text())
    settings.update(model=dict(layers=1, hidden=16, heads=2, feedforward=32, context=256, dropout=0.1),
                    vocab_size=vocab, device="cpu")
    settings["training"].update(context=16, stride=8, accumulation=2, fp16=False,
                                budget_limit=2, validation_interval=1, checkpoint_interval=1)
    settings["data"].update(root=str(root), tokenizer=str(root / "tokenizers"))
    settings["checkpoint"]["output"] = str(tmp_path / "checkpoints")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    return path, settings


def test_cli_train_resume_evaluate(recipe, capsys):
    path, settings = recipe
    main(["--config", str(path), "--validate-only"])
    assert "valid" in capsys.readouterr().out
    main(["--config", str(path)])
    result = json.loads(capsys.readouterr().out)
    assert result["counters"]["successful_updates"] == 2
    assert result["validation"]["bytes"] == len("validation e\u0301 \U0001f98a\n".encode())
    checkpoint = Path(settings["checkpoint"]["output"]) / "last.pt"
    _, payload = load_training_checkpoint(checkpoint)
    assert payload["training_state"]["environment"]["source_sha256"]
    main(["--config", str(path), "--resume", str(checkpoint)])
    assert json.loads(capsys.readouterr().out)["counters"]["successful_updates"] == 2
    main(["--config", str(path), "--evaluate", str(checkpoint)])
    assert json.loads(capsys.readouterr().out)["bpb"] == result["validation"]["bpb"]
    with pytest.raises(FileExistsError):
        main(["--config", str(path)])
    assert not any(n == "src" or n.startswith("src.") or n.startswith("archive.") for n in sys.modules)


@pytest.mark.parametrize("section,key,value", [
    ("model", "dim", 512), ("training", "gradient_accumulation", 2),
    ("training", "fp16", True), ("training", "budget_policy", "tokens"),
    ("training", "budget_limit", 0), ("model", "context", 128),
    ("evaluation", "stride", 64), ("evaluation", "protocol", "unsupported"), ("data", "tier", "test"),
    ("checkpoint", "output", ""),
])
def test_config_rejects_mismatch(recipe, section, key, value):
    path, settings = recipe
    settings[section][key] = value
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises((ValueError, TypeError)):
        load_config(path)


def test_cli_rejects_changed_tokenizer_byte_table(recipe):
    path, settings = recipe
    config = load_config(path)
    index = Path(config.data.root) / "shards" / str(config.vocab_size) / "validation" / "index.json"
    metadata = read_json(index)
    metadata["byte_lengths"][1] += 1
    write_json(index, metadata)
    with pytest.raises(ValueError, match="byte table"):
        load_data(config)


def test_provenance_mismatch_distinct_from_corruption(tmp_path):
    write_json(tmp_path / "verification.json", {"implementation_sha256": {"old/path.py": "a"*64}})
    from vocab_lm.fineweb import file_hash
    write_json(tmp_path / "artifact_hashes.json", {"verification.json": {
        "sha256": file_hash(tmp_path / "verification.json"), "bytes": (tmp_path / "verification.json").stat().st_size}})
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(CodeProvenanceMismatch):
        verify(tmp_path, {})
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert all(Path(p).exists() for p in implementation_fingerprints())
    with pytest.raises(FileExistsError):
        verify(tmp_path, {}, publish=True)
    (tmp_path / "verification.json").write_text("corruption")
    with pytest.raises(ValueError, match="artifact hash") as error:
        verify(tmp_path, {}, allow_code_provenance_mismatch=True)
    assert not isinstance(error.value, CodeProvenanceMismatch)
