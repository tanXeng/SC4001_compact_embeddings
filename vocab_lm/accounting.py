"""Persistent model/storage accounting, separate from GPU memory measurements."""
from __future__ import annotations

from pathlib import Path
import json
import torch

from .checkpoint import archive_bytes, inference_payload


def parameter_counts(model) -> dict[str, int]:
    # parameters() deduplicates Parameter objects, including tied aliases.
    parameters = list(model.parameters())
    vocabulary = list(model.vocabulary.parameters())
    total = sum(p.numel() for p in parameters)
    vocab = sum(p.numel() for p in vocabulary)
    return {"backbone_parameters": total-vocab, "vocabulary_parameters": vocab,
            "total_parameters": total,
            "trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad)}


def persistent_buffer_bytes(model) -> int:
    state = model.state_dict()
    return sum(buffer.numel()*buffer.element_size() for name, buffer in model.named_buffers()
               if name in state)


def serialized_inference_bytes(model, **metadata) -> int:
    return len(archive_bytes(inference_payload(model, **metadata)))


def tokenizer_artifact_bytes(paths) -> dict[str, int]:
    paths = [Path(path) for path in paths]
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Tokenizer artifacts must not be counted twice")
    return {p.as_posix(): p.stat().st_size for p in paths}


def tensor_bytes(tree) -> int:
    """Tensor payload in an optimizer state, excluding Python/archive overhead."""
    if isinstance(tree, torch.Tensor):
        return tree.numel()*tree.element_size()
    if isinstance(tree, dict):
        return sum(tensor_bytes(v) for v in tree.values())
    if isinstance(tree, (list, tuple)):
        return sum(tensor_bytes(v) for v in tree)
    return 0


def storage_report(model, *, tokenizer_paths=(), optimizer_state=None, **metadata) -> dict:
    tokenizers = tokenizer_artifact_bytes(tokenizer_paths)
    archive = serialized_inference_bytes(model, **metadata)
    vocabulary = model.vocabulary
    spec = vocabulary.specification()
    pq = spec["kind"] == "pq"
    parameter_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
    fixed_bytes = persistent_buffer_bytes(model)
    return {**parameter_counts(model), "persistent_buffer_bytes": persistent_buffer_bytes(model),
            "parameter_payload_bytes": parameter_bytes,
            "total_model_tensor_bytes": parameter_bytes + fixed_bytes,
            "dense_vocabulary_bytes": tensor_bytes(vocabulary.weight) if spec["kind"] == "dense" else 0,
            **vocabulary_storage_report(vocabulary),
            "dense_equivalent_vocabulary_bytes": vocabulary.vocab_size*vocabulary.hidden*4,
            "pq_codebook_bytes": tensor_bytes(vocabulary.codebooks) if pq else 0,
            "pq_assignment_bytes": tensor_bytes(vocabulary.codes) if pq else 0,
            "required_representation_metadata_json_bytes": len(json.dumps(spec, sort_keys=True).encode()),
            "fixed_buffer_bytes": fixed_bytes,
            "runtime_derived_buffer_bytes": sum(tensor_bytes(b) for n, b in model.named_buffers()
                                                if n not in model.state_dict()),
            "optimizer_state_tensor_bytes": tensor_bytes(optimizer_state),
            "temporary_reconstructed_table_bytes": vocabulary.vocab_size*vocabulary.hidden*4 if pq else 0,
            "serialized_inference_checkpoint_bytes": archive,
            "tokenizer_artifact_bytes": tokenizers, "tokenizer_bytes": sum(tokenizers.values()),
            "deployment_bytes": archive+sum(tokenizers.values())}


def vocabulary_storage_report(vocabulary) -> dict:
    """Standalone inference representation archive, not a model-archive delta.

    Includes specification and persistent state, excludes optimizer and runtime
    activations. Archive overhead is measured rather than guessed from parameters.
    """
    state = {n: t.detach().cpu().clone() for n, t in vocabulary.state_dict().items()}
    spec = vocabulary.specification()
    buffers = {n: b for n, b in vocabulary.named_buffers() if n in state}
    return dict(
        vocabulary_trainable_parameters=sum(p.numel() for p in vocabulary.parameters() if p.requires_grad),
        vocabulary_parameter_bytes=sum(tensor_bytes(p) for p in vocabulary.parameters()),
        vocabulary_fixed_buffer_bytes=tensor_bytes(buffers),
        vocabulary_integer_buffer_bytes=sum(tensor_bytes(b) for b in buffers.values() if not b.is_floating_point()),
        vocabulary_metadata_json_bytes=len(json.dumps(spec, sort_keys=True).encode()),
        vocabulary_serialized_inference_bytes=len(archive_bytes(dict(specification=spec, state_dict=state))))
