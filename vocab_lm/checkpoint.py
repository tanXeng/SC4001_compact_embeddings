"""Versioned inference and training archives."""
from __future__ import annotations

from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import tempfile
import time
import sys
import warnings

import torch

from .config import evaluation_config, ModelConfig
from .model import CausalTransformer
from .model import PQVocabulary, LowRankVocabulary, HashVocabulary, AdaptiveVocabulary


def _identity(value: dict, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Require {name} identity")
    digest = value.get("sha256", "")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError(f"Require {name} SHA-256")
    if name == "dataset":
        if not isinstance(value.get("protocol"), dict) or value.get("split") != "validation":
            raise ValueError("Require an identified validation dataset/protocol")
        evaluation_config(value["protocol"])
    return value


def inference_payload(model: CausalTransformer, *, tokenizer: dict, dataset: dict,
                      training_metadata: dict | None = None,
                      provenance: dict | None = None) -> dict:
    specification = model.vocabulary.specification()
    metadata = {"tokenizer": _identity(tokenizer, "tokenizer"),
                "dataset": _identity(dataset, "dataset"),
                "training_metadata": training_metadata or {}, "provenance": provenance or {}}
    # Primitive metadata permits restricted weights_only loading; copy caller data.
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if any(t.is_floating_point() and (t.dtype != torch.float32 or not torch.isfinite(t).all())
           for t in state.values()):
        raise ValueError("Inference checkpoint requires finite FP32 state")
    return {"format": "vocab_lm_inference", "format_version": 1,
            "model_config": asdict(model.config), "vocabulary": specification,
            "model": state, "trainable_parameters": [n for n, p in model.named_parameters() if p.requires_grad],
            "training_state": None, **metadata}


def archive_bytes(payload: dict) -> bytes:
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def save_checkpoint(path: Path, model: CausalTransformer, **metadata) -> int:
    """Publish a complete archive atomically; never replace any existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    data = archive_bytes(inference_payload(model, **metadata))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and fails if another writer won the name.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(data)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[CausalTransformer, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "vocab_lm_inference" or payload.get("format_version") != 1:
        raise ValueError("Unsupported inference checkpoint format/version")
    if payload.get("training_state") is not None:
        raise ValueError("Inference archives cannot contain training state")
    _identity(payload["tokenizer"], "tokenizer")
    _identity(payload["dataset"], "dataset")
    model = _restore_model(payload)
    state = payload["model"]
    _load_model_state(model, state)
    trainable = payload["trainable_parameters"]
    names = dict(model.named_parameters())
    if (not isinstance(trainable, list) or any(not isinstance(n, str) for n in trainable)
            or len(set(trainable)) != len(trainable) or not set(trainable) <= names.keys()):
        raise ValueError("Invalid parameter trainability metadata")
    model.requires_grad_(False)
    for name in trainable:
        names[name].requires_grad_(True)
    return model.to(device).eval(), payload


def _restore_model(payload):
    spec, state = payload["vocabulary"], payload["model"]
    config = ModelConfig(**payload["model_config"])
    if spec.get("kind") == "dense":
        model = CausalTransformer(config, spec["vocab_size"])
    elif spec.get("kind") == "pq":
        if not {"vocabulary.codebooks", "vocabulary.codes"} <= state.keys():
            raise ValueError("PQ specification requires codebooks and assignments")
        # Construct a one-row placeholder, never a full dense vocabulary table.
        model = CausalTransformer(config, 1)
        model.vocabulary = PQVocabulary(state["vocabulary.codebooks"], state["vocabulary.codes"])
    elif spec.get("kind") in {"low_rank", "hash", "adaptive"}:
        model = CausalTransformer(config, 1)
        if spec["kind"] == "low_rank":
            model.vocabulary = LowRankVocabulary(spec["vocab_size"], spec["hidden"], spec["rank"])
        elif spec["kind"] == "hash":
            model.vocabulary = HashVocabulary(spec["vocab_size"], spec["hidden"],
                spec["buckets"], spec["seeds"], spec["importance_init"])
        else:
            # Counts are not needed for inference: the verified stored permutation
            # is authoritative. Strict state loading below validates both mappings.
            model.vocabulary = AdaptiveVocabulary(
                torch.zeros(spec["vocab_size"], dtype=torch.int64), spec["hidden"],
                spec["cutoffs"], spec["initial_dim"], spec["factor"], spec["special_ids"])
    else:
        raise ValueError("Unsupported vocabulary kind")
    if model.vocabulary.specification() != spec or model.vocabulary.hidden != config.hidden:
        raise ValueError("Vocabulary and architecture specifications disagree")
    return model


def _load_model_state(model, state):
    expected = model.state_dict()
    if any(not isinstance(t, torch.Tensor) or n not in expected or t.dtype != expected[n].dtype
           or (t.is_floating_point() and not torch.isfinite(t).all()) for n, t in state.items()):
        raise ValueError("Require finite FP32 model tensors and exact assignment dtypes")
    model.load_state_dict(state, strict=True)


def _replace_with_retry(temporary: Path, path: Path) -> None:
    """Retry brief access denials without removing the last atomic checkpoint."""
    original = None
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            if original is None:
                original = error
            if attempt == 4:
                raise original
            time.sleep(0.05 * 2**attempt)


def save_training_checkpoint(path: Path, model: CausalTransformer, training_state: dict,
                             *, tokenizer: dict, dataset: dict) -> int:
    """Atomic replace for this training format only; inference archives stay immutable."""
    path = Path(path)
    if path.exists():
        previous = torch.load(path, map_location="cpu", weights_only=True)
        if previous.get("format") != "vocab_lm_training":
            raise ValueError("Refuse to replace a non-training archive")
    payload = inference_payload(model, tokenizer=tokenizer, dataset=dataset)
    # Serialization is synchronous at an update boundary; avoid duplicating
    # AdamW's GPU tensors merely to make an unnecessary deep copy.
    payload.update(format="vocab_lm_training", training_state=training_state)
    data = archive_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        failing = sys.exc_info()[0] is not None
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                if not failing:
                    raise
                warnings.warn(f"Checkpoint temporary file retained at {temporary}: {error}")
    return len(data)


def load_training_checkpoint(path: Path, device: str = "cpu"):
    """Load model only; train() restores optimizer/cursor/RNG after construction."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "vocab_lm_training" or payload.get("format_version") != 1:
        raise ValueError("Unsupported training checkpoint")
    _identity(payload["tokenizer"], "tokenizer")
    _identity(payload["dataset"], "dataset")
    model = _restore_model(payload)
    _load_model_state(model, payload["model"])
    names = dict(model.named_parameters())
    trainable = payload["trainable_parameters"]
    if len(set(trainable)) != len(trainable) or not set(trainable) <= names.keys():
        raise ValueError("Invalid trainability metadata")
    for name, parameter in names.items():
        parameter.requires_grad_(name in trainable)
    return model.to(device), payload
