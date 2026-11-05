"""Configuration and prepared-data binding on synthetic byte-BPE shards."""
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml

from vocab_lm.data import load_data
from vocab_lm.config import TrainConfig
from vocab_lm.config import load_config
from vocab_lm.fineweb import describe, read_json, write_json
from vocab_lm.fineweb import dataset_identity
from vocab_lm.data import write_shards
from vocab_lm.tokenizer import tokenizer_identity, FineWebEvaluationConfig


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
    settings = dict(training=asdict(TrainConfig()),data=dict(tier="pilot"),checkpoint=dict(output="unused"), evaluation=asdict(FineWebEvaluationConfig()))
    settings.update(model=dict(layers=1, hidden=16, heads=2, feedforward=32, context=256, dropout=0.1),
                    vocab_size=vocab, device="cpu")
    settings["training"].update(context=16, stride=8, accumulation=2, fp16=False,
                                budget_limit=2, validation_interval=1, checkpoint_interval=1)
    settings["data"].update(root=str(root), tokenizer=str(root / "tokenizers"))
    settings["checkpoint"]["output"] = str(tmp_path / "checkpoints")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    return path, settings


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


def test_loader_rejects_changed_tokenizer_byte_table(recipe):
    path, settings = recipe
    config = load_config(path)
    index = Path(config.data.root) / "shards" / str(config.vocab_size) / "validation" / "index.json"
    metadata = read_json(index)
    metadata["byte_lengths"][1] += 1
    write_json(index, metadata)
    with pytest.raises(ValueError, match="byte table"):
        load_data(config)




def test_loader_smoke_prefix_identity(recipe):
    path,settings=recipe
    cfg=load_config(path)
    full,_,_=load_data(cfg)
    short,_,_=load_data(replace(cfg,data=replace(cfg.data,tier='smoke')))
    assert len(short.documents)==1<len(full.documents)
    assert short.identity['dataset']['sha256']!=full.identity['dataset']['sha256']
