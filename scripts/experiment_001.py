"""Gate A only: synthetic measurements, optional diagnostics; no corpus training."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import platform
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from vocab_lm.model import (CausalTransformer, LowRankVocabulary, HashVocabulary,
                            AdaptiveVocabulary)
from vocab_lm.config import ModelConfig
from vocab_lm.accounting import storage_report
from vocab_lm.train import training_flops, compressed_vocabulary_flops
from vocab_lm.tokenizer import FineWebEvaluationConfig


def diagnostics(vocabulary, sample_size=128, interference_tokens=4, training_counts=None):
    """Optional small-subset diagnostics. Never called by ordinary training.

    Interference isolates lookup-only gradients for a common fixed probe, not the
    full LM loss. Each sampled token requires an extra autograd.grad call. Gradients
    are flattened sequentially for this tiny subset, never for the full vocabulary.
    """
    device = next(vocabulary.parameters()).device
    ids = torch.linspace(0, vocabulary.vocab_size-1,
                         min(sample_size, vocabulary.vocab_size), device=device).long()
    e = vocabulary.lookup(ids)
    detached = e.detach().float()
    singular = torch.linalg.svdvals(detached)
    probabilities = singular / singular.sum().clamp_min(1e-30)
    result = dict(sample_ids=ids.tolist(), mean=detached.mean().item(),
                  variance=detached.var(unbiased=False).item(),
                  norm_mean=detached.norm(dim=-1).mean().item(),
                  singular_values=singular.tolist(),
                  entropy_effective_rank=(-(probabilities * probabilities.clamp_min(1e-30).log()).sum()).exp().item(),
                  numerical_rank=torch.linalg.matrix_rank(detached).item(),
                  condition_number=(singular[0]/singular[-1].clamp_min(1e-30)).item(),
                  parameter_norms={n: p.detach().norm().item() for n, p in vocabulary.named_parameters()},
                  gradient_norms={n: None if p.grad is None else p.grad.norm().item()
                                  for n, p in vocabulary.named_parameters()},
                  gradients_finite=all(p.grad is None or torch.isfinite(p.grad).all().item()
                                       for p in vocabulary.parameters()))
    shared = ([vocabulary.components] if isinstance(vocabulary, HashVocabulary) else
              [vocabulary.b] if isinstance(vocabulary, LowRankVocabulary) else
              list(vocabulary.projections) if isinstance(vocabulary, AdaptiveVocabulary) else [])
    started = time.perf_counter()
    gradients = []
    probe = torch.linspace(-1, 1, vocabulary.hidden, device=device)
    for token in ids[:interference_tokens]:
        if not shared:
            break
        grads = torch.autograd.grad((vocabulary.lookup(token.reshape(1))*probe).sum(),
                                    shared, allow_unused=True)
        gradients.append(torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                                    for p, g in zip(shared, grads)]))
    if gradients:
        g = torch.stack(gradients)
        norms = g.norm(dim=1)
        result["interference"] = dict(tokens=ids[:len(g)].tolist(),
            reinforcement_ratio=(g.sum(0).norm()/norms.sum().clamp_min(1e-30)).item(),
            cosine_matrix=(F.normalize(g, dim=1) @ F.normalize(g, dim=1).T).tolist(),
            seconds=time.perf_counter()-started,
            limitation="lookup-only fixed probe; few evenly spaced IDs, not frequency-weighted LM gradients")
    if isinstance(vocabulary, HashVocabulary):
        _, counts = vocabulary.indices.unique(dim=0, return_counts=True)
        used = vocabulary.indices.unique().numel()
        result["hash"] = dict(duplicate_excess_rate=1-len(counts)/vocabulary.vocab_size,
            tokens_in_duplicate_groups_rate=counts[counts>1].sum().item()/vocabulary.vocab_size,
            bucket_utilization=used/vocabulary.buckets,
            importance_mean=vocabulary.importance.detach().mean().item(),
            importance_std=vocabulary.importance.detach().std(unbiased=False).item(),
            importance_min=vocabulary.importance.detach().min().item(),
            importance_max=vocabulary.importance.detach().max().item())
    if isinstance(vocabulary, AdaptiveVocabulary):
        result["bands"] = []
        start = 0
        for end, a, b in zip(vocabulary.cutoffs, vocabulary.tables, vocabulary.projections):
            ids_band = vocabulary.order[start:min(end, start+sample_size)]
            result["bands"].append(dict(tokens=end-start, parameters=a.numel()+b.numel(),
                training_occurrences=None if training_counts is None else
                    training_counts[vocabulary.order[start:end].cpu()].sum().item(),
                sampled_norm_mean=vocabulary.lookup(ids_band).detach().norm(dim=-1).mean().item(),
                table_gradient_norm=None if a.grad is None else a.grad.norm().item(),
                projection_gradient_norm=None if b.grad is None else b.grad.norm().item()))
            start = end
    return result


def stratified_metrics(logits, targets, training_counts, groups=3, vocabulary=None):
    """Aggregate prediction NLL/accuracy by training frequency, not target counts."""
    order = sorted(range(len(training_counts)), key=lambda i: (-int(training_counts[i]), i))
    rank = torch.empty(len(order), dtype=torch.long, device=targets.device)
    rank[torch.tensor(order, device=targets.device)] = torch.arange(len(order), device=targets.device)
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.flatten(), reduction="none")
    correct = logits.argmax(-1).flatten() == targets.flatten()
    result = []
    if isinstance(vocabulary, AdaptiveVocabulary):
        positions = vocabulary.inverse[targets.flatten()]
        memberships = torch.bucketize(positions, torch.tensor(
            vocabulary.cutoffs[:-1], device=targets.device), right=True)
        groups = len(vocabulary.cutoffs)
    else:
        memberships = rank[targets.flatten()]*groups//len(order)
    for band in range(groups):
        mask = memberships == band
        result.append(dict(positions=mask.sum().item(),
            nll=losses[mask].mean().item() if mask.any() else None,
            accuracy=correct[mask].float().mean().item() if mask.any() else None))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/experiment_001/synthetic.json"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    architecture = ModelConfig(6, 256, 4, 1024, 512, .1)
    results = dict(seed=42, architecture=asdict(architecture), synthetic_only=True,
        environment=dict(python=platform.python_version(), torch=torch.__version__,
            cuda=torch.version.cuda, device=args.device,
            device_name=torch.cuda.get_device_name() if args.device == "cuda" else platform.processor(),
            cpu_threads=torch.get_num_threads()), runs=[])
    metadata = dict(tokenizer={"sha256": "a"*64}, dataset={"sha256": "b"*64,
        "split": "validation", "protocol": asdict(FineWebEvaluationConfig())})
    for v in (8192, 16384):
        # Artificial descending counts only; this does not inspect prepared data.
        counts = torch.arange(v, 0, -1, dtype=torch.int64)
        torch.manual_seed(42)
        reference = CausalTransformer(architecture, v)
        backbone = {n: p.clone() for n, p in reference.state_dict().items() if not n.startswith("vocabulary.")}
        del reference
        for kind in ("dense", "low_rank", "hash", "adaptive"):
            torch.manual_seed(42)
            model = CausalTransformer(architecture, v)
            if kind == "low_rank":
                model.vocabulary = LowRankVocabulary(v, 256, 96)
            elif kind == "hash":
                model.vocabulary = HashVocabulary(v, 256, v//4)
            elif kind == "adaptive":
                model.vocabulary = AdaptiveVocabulary(counts, 256, [v//8, v//2])
            assert all(torch.equal(model.state_dict()[n], p) for n, p in backbone.items())
            model = model.to(args.device).eval()
            # One small forward/backward only, no optimizer or data loader.
            ids = torch.arange(16, device=args.device).reshape(2, 8)
            targets = (ids+1) % v
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(ids)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            logits.retain_grad()
            loss.backward()
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            ceiling = (4 if v == 8192 else 8)*1024**2
            storage = storage_report(model, **metadata)
            row = dict(vocab_size=v, kind=kind, backbone_identical=True,
                storage=storage, ceiling_bytes=None if kind == "dense" else ceiling,
                slack_bytes=None if kind == "dense" else ceiling-storage["vocabulary_serialized_inference_bytes"],
                loss=loss.item(), forward_backward_seconds=elapsed,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                cpu_peak_memory="not measured", estimated_training_flops=training_flops(model, 2, 8),
                frequency_metrics=stratified_metrics(logits.detach(), targets, counts,
                                                     vocabulary=model.vocabulary))
            if kind != "dense":
                row["flop_details"] = compressed_vocabulary_flops(model.vocabulary, 16)
            if args.diagnostics:
                # Gradient w.r.t. effective output rows: dL/dE = dL/dlogits^T H.
                # This is a sampled output-path diagnostic, excluding lookup path.
                captured = []
                handle = model.norm.register_forward_hook(lambda m, a, out: captured.append(out.detach()))
                with torch.no_grad():
                    model(ids)
                handle.remove()
                sample = torch.arange(min(128, v), device=args.device)
                effective_grad = logits.grad.flatten(0, 1)[:, sample].T @ captured[0].flatten(0, 1)
                row["effective_output_embedding_gradient_norm"] = effective_grad.norm().item()
                row["diagnostics"] = diagnostics(model.vocabulary, training_counts=counts)
            results["runs"].append(row)
            del model, logits, loss
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
