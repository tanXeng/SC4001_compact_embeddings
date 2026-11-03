"""FP32 vocabulary scaling orchestration; all updates use vocab_lm.train."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from scripts.train_dense import load_data
from vocab_lm.accounting import storage_report, vocabulary_storage_report
from vocab_lm.checkpoint import archive_bytes, load_training_checkpoint, save_training_checkpoint
from vocab_lm.config import ModelConfig, TrainConfig, DataConfig, evaluation_config
from vocab_lm.data import TrainingData
from vocab_lm.evaluate import evaluate
from vocab_lm.model import CausalTransformer, LowRankVocabulary, DenseVocabulary
from vocab_lm.runtime import environment, seed_all, rng_state, restore_rng, sha256, write_json
from vocab_lm.train import train, scaling_training_flops, FLOP_ESTIMATOR_VERSION, Counters


STORAGE_VERSION = "standalone_torch_save_v1"
# Optional additions keyed by representation kind; callbacks receive the model.
DIAGNOSTIC_EXTENSIONS = {}


def load_settings(path):
    s = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"model", "training", "data", "evaluation", "arms", "baseline", "device",
                "output", "evaluation_fractions", "diagnostic_fractions", "screen_documents",
                "confirmation_documents", "threads", "profile_updates", "profile_warmup"}
    if set(s) != required:
        raise ValueError(f"Configuration keys must be {sorted(required)}")
    c, t = ModelConfig(**s["model"]), TrainConfig(**s["training"])
    e = evaluation_config(s["evaluation"])
    if t.fp16 or t.budget_policy != "flops" or t.flop_estimator != FLOP_ESTIMATOR_VERSION:
        raise ValueError("Scaling requires FP32 and the frozen FLOP estimator/budget")
    if max(t.context, e.context) > c.context or t.validation_interval:
        raise ValueError("Invalid contexts; evaluation is scheduled by FLOPs")
    for key in ("evaluation_fractions", "diagnostic_fractions"):
        if s[key] != sorted(set(s[key])) or any(not 0 < x <= 1 for x in s[key]):
            raise ValueError("Fractions must increase in (0,1]")
    if s["baseline"] not in s["arms"] or s["arms"][s["baseline"]]["kind"] != "dense":
        raise ValueError("Require dense baseline")
    for arm in s["arms"].values():
        if arm["kind"] not in {"dense", "low_rank"} or arm["vocab_size"] < 1:
            raise ValueError("Primary arms require dense or low_rank")
    return s


def construct(s, arm):
    """Measure real archives before quality; maximal rank by serialized binary search."""
    seed_all(s["training"]["seed"])
    c = ModelConfig(**s["model"])
    baseline = s["arms"][s["baseline"]]["vocab_size"]
    ceiling = vocabulary_storage_report(DenseVocabulary(torch.zeros(baseline, c.hidden)))["vocabulary_serialized_inference_bytes"]
    a = s["arms"][arm]
    if a["kind"] == "dense":
        model = CausalTransformer(c, a["vocab_size"])
    else:
        lo, hi = 0, min(c.hidden, a["vocab_size"])
        while lo < hi:
            mid = (lo+hi+1)//2
            size = vocabulary_storage_report(LowRankVocabulary(a["vocab_size"], c.hidden, mid))["vocabulary_serialized_inference_bytes"]
            if size <= ceiling:
                lo = mid
            else:
                hi = mid-1
        if lo < 1:
            raise ValueError("No rank fits")
        seed_all(s["training"]["seed"])
        model = CausalTransformer(c, 1)
        model.vocabulary = LowRankVocabulary(a["vocab_size"], c.hidden, lo)
    return model, ceiling


def prepared(s, arm):
    a = s["arms"][arm]
    if not a.get("tokenizer"):
        raise ValueError(f"{arm}: tokenizer is not prepared; configure artifacts first")
    cfg = SimpleNamespace(data=DataConfig(**s["data"], tokenizer=a["tokenizer"]),
                          vocab_size=a["vocab_size"], training=TrainConfig(**s["training"]),
                          evaluation=evaluation_config(s["evaluation"]))
    data, valid, identity = load_data(cfg)
    n, m = s["screen_documents"], s["confirmation_documents"]
    if min(n, m) < 1 or n+m > len(valid):
        raise ValueError("Need disjoint screening and confirmation documents")
    screen, confirmation = [valid[i] for i in range(n)], [valid[i] for i in range(n, n+m)]
    train_hashes = {d.raw_sha256 for d in data.documents}
    if train_hashes & {d.raw_sha256 for d in screen+confirmation}:
        raise ValueError("Train/validation overlap")
    if {d.raw_sha256 for d in screen} & {d.raw_sha256 for d in confirmation}:
        raise ValueError("Screen/confirmation overlap")
    identity = dict(identity, bytes=sum(sum(d.byte_lengths) for d in screen),
                    screen_hashes=[d.raw_sha256 for d in screen],
                    confirmation_hashes=[d.raw_sha256 for d in confirmation])
    counts = torch.zeros(a["vocab_size"], dtype=torch.int64)
    for d in data.documents:
        counts += torch.bincount(torch.tensor(d.ids[1:]), minlength=len(counts))
    order = sorted(range(len(counts)), key=lambda i: (-int(counts[i]), i))
    groups = torch.empty_like(counts)
    groups[torch.tensor(order)] = torch.arange(len(counts))*3//len(counts)
    return data, screen, confirmation, identity, groups


def append(path, row):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, allow_nan=False)+"\n")


def gpu_memory(device):
    if torch.device(device).type != "cuda":
        return {k: None for k in ("cuda_allocated", "cuda_reserved", "peak_allocated", "peak_reserved")}
    return dict(cuda_allocated=torch.cuda.memory_allocated(), cuda_reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())


def diagnose(model, sample=128, diagnostic_batch=None):
    """Deterministic bounded effective-row spectrum; preserves RNG and training state."""
    with torch.no_grad():
        v = model.vocabulary
        ids = torch.linspace(0, v.vocab_size-1, min(sample, v.vocab_size), device=next(v.parameters()).device).long()
        e = v.lookup(ids).float()
        spectrum = torch.linalg.svdvals(e)
        prob = spectrum/spectrum.sum().clamp_min(1e-30)
        result = dict(sample_ids=ids.tolist(), singular_values=spectrum.tolist(),
                      numerical_rank=int(torch.linalg.matrix_rank(e)),
                      entropy_rank=float((-(prob*prob.clamp_min(1e-30).log()).sum()).exp()),
                      effective_norms=e.norm(dim=1).tolist(), coordinate_variance=e.var(0, unbiased=False).tolist(),
                      factor_norms={n: float(p.norm()) for n, p in v.named_parameters()},
                      gradient_norms={n: float(p.grad.norm()) if p.grad is not None else None for n, p in v.named_parameters()})
        if hasattr(v, "b"):
            sv = torch.linalg.svdvals(v.b)
            result.update(projection_condition=float(sv[0]/sv[-1]), rank_utilization=result["numerical_rank"]/v.rank)
        if hasattr(v, "indices"):
            _, counts = v.indices.unique(dim=0, return_counts=True)
            occupancy = torch.bincount(v.indices.flatten(), minlength=v.buckets)
            sharing = []
            for column in v.indices.T:
                c = torch.bincount(column, minlength=v.buckets)
                sharing.append(int((c*(c-1)//2).sum()))
            result["hash"] = dict(bucket_occupancy=occupancy.tolist(), bucket_utilization=float((occupancy>0).float().mean()),
                                  full_code_collision_excess=int((counts-1).sum()), partial_code_shared_pairs=sharing,
                                  importance_quantiles=torch.quantile(v.importance.flatten(),torch.tensor([0.,.25,.5,.75,1.],device=e.device)).tolist())
        if hasattr(v, "tables"):
            result["bands"] = []
            for table, projection in zip(v.tables, v.projections):
                sv = torch.linalg.svdvals(projection)
                result["bands"].append(dict(table_norm=float(table.norm()), projection_norm=float(projection.norm()),
                       projection_condition=float(sv[0]/sv[-1]), parameters=table.numel()+projection.numel(),
                       parameter_bytes=4*(table.numel()+projection.numel()), table_rank=int(torch.linalg.matrix_rank(table)),
                       table_gradient_norm=None if table.grad is None else float(table.grad.norm()),
                       projection_gradient_norm=None if projection.grad is None else float(projection.grad.norm())))
    if hasattr(v, "indices") and diagnostic_batch is not None:
        # At most four actual causal-LM target losses, including both tied paths.
        rng, previous = rng_state(), model.training
        gradients = []
        try:
            model.eval()
            x, y = diagnostic_batch
            logits = model(x)
            losses = torch.nn.functional.cross_entropy(logits.flatten(0,1), y.flatten(), reduction="none",ignore_index=-100)
            for index in (y.flatten()!=-100).nonzero().flatten()[:4]:
                g = torch.autograd.grad(losses[index], tuple(v.parameters()), retain_graph=True)
                gradients.append(torch.cat([t.flatten() for t in g]))
            if gradients:
                g = torch.stack(gradients)
                norms = g.norm(dim=1)
                result["lm_gradient_interference"] = dict(targets=len(g), cancellation=float(1-g.sum(0).norm()/norms.sum().clamp_min(1e-30)),
                    dominance=float(norms.max()/norms.sum().clamp_min(1e-30)), cosine_matrix=(torch.nn.functional.normalize(g,dim=1)@torch.nn.functional.normalize(g,dim=1).T).tolist())
        finally:
            model.train(previous)
            restore_rng(rng)
    extension = DIAGNOSTIC_EXTENSIONS.get(v.specification()["kind"])
    if extension is not None:
        result["extension"] = extension(model)
    return result


def preflight(s):
    manifests = {}
    reference = None
    for arm, a in s["arms"].items():
        if a.get("optional") and not a.get("tokenizer"):
            continue
        data, screen, confirmation, identity, groups = prepared(s, arm)
        raw = [(d.raw_sha256, sum(d.byte_lengths)) for d in data.documents]
        raw += [(d.raw_sha256, sum(d.byte_lengths)) for d in screen+confirmation]
        if reference is not None and raw != reference:
            raise ValueError("Cross-vocabulary raw document ordering/coverage mismatch")
        reference = raw
        # Reconstruct bytes from actual tokenizer pieces, validate SHA, not just lengths.
        from vocab_lm.tokenizer import NestedBPE
        pair = NestedBPE.load(Path(a["tokenizer"]))
        expanded = a["vocab_size"] == pair.expanded.get_vocab_size()
        pieces = [pair.token_bytes(i, expanded=expanded) for i in range(a["vocab_size"])]
        for d in list(data.documents)+screen+confirmation:
            if sha256(b"".join(pieces[i] for i in d.ids)) != d.raw_sha256:
                raise ValueError("Tokenizer byte reconstruction mismatch")
        model, ceiling = construct(s, arm)
        manifests[arm] = dict(ceiling=ceiling, specification=model.vocabulary.specification(),
                              storage=vocabulary_storage_report(model.vocabulary), tokenizer=data.identity["tokenizer"],
                              validation=identity, raw_documents=len(raw))
    return manifests


def run(s, arm, *, resume=False, max_attempts=None, profile=False, operation="train"):
    output = Path(s["output"])/arm/("profile" if profile else "run")
    if output.exists() and any(output.iterdir()) and not resume and operation == "train":
        raise FileExistsError("Fresh run requires an empty destination")
    output.mkdir(parents=True, exist_ok=True)
    data, screen, confirmation, validation, groups = prepared(s, arm)
    model, ceiling = construct(s, arm)
    device = torch.device(s["device"])
    model = model.to(device)
    manifest = dict(config=s, arm=arm, model=asdict(model.config), representation=model.vocabulary.specification(),
                    ceiling=ceiling, dataset=data.identity, validation=validation, environment=environment(device),
                    flop_estimator=FLOP_ESTIMATOR_VERSION, storage_version=STORAGE_VERSION, precision="FP32; TF32 disabled",
                    cuda_determinism="same environment; numerical tolerance, no cross-device bitwise guarantee")
    manifest = json.loads(json.dumps(manifest))
    identity = sha256(json.dumps(manifest, sort_keys=True).encode())
    payload = None
    if resume or operation != "train":
        if json.loads((output/"manifest.json").read_text()) != manifest:
            raise ValueError("Frozen manifest changed")
        model, payload = load_training_checkpoint(output/"last.pt", str(device))
        if resume:
            # Checkpoint is authoritative after a crash between log append and publication.
            limit = payload["training_state"]["counters"]["attempted_updates"]
            for name in ("training.jsonl", "validation.jsonl", "diagnostics.jsonl"):
                path = output/name
                if path.exists():
                    retained = []
                    for line in path.read_text(encoding="utf-8").splitlines():
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            break  # interrupted final append
                        if row["attempted_updates"] <= limit:
                            retained.append(line)
                    path.write_text("\n".join(retained)+("\n" if retained else ""), encoding="utf-8")
    else:
        write_json(output/"manifest.json", manifest)
    t = TrainConfig(**s["training"])
    def ev(state, subset, label):
        started = time.perf_counter()
        result = evaluate(model, subset, evaluation_config(s["evaluation"]), frequency_groups=groups)
        row = dict(**result, **state["counters"], subset=label, evaluation_seconds=time.perf_counter()-started)
        append(output/"validation.jsonl", row)
        return row
    def diag(state):
        rng = rng_state()
        started = time.perf_counter()
        try:
            values = diagnose(model)
        finally:
            restore_rng(rng)
        append(output/"diagnostics.jsonl", dict(**state["counters"], name="effective_rows", version=1,
               values=values, seconds=time.perf_counter()-started, additional_memory=None))
    if operation == "evaluate":
        return ev(payload["training_state"], screen, "screen")
    if operation == "diagnose":
        diag(payload["training_state"])
        return
    if operation == "confirm":
        # Confirmation is a separate, once-only operation after BOTH arms finish.
        for other in s["arms"]:
            other_path = Path(s["output"])/other/"run"
            result = json.loads((other_path/"result.json").read_text())
            if result.get("completion") != "budget_boundary":
                raise ValueError("Both arms must finish before confirmation")
        history = output/"validation.jsonl"
        if any(json.loads(line)["subset"] == "confirmation_final" for line in history.read_text().splitlines()):
            raise ValueError("Confirmation already recorded")
        return ev(payload["training_state"], confirmation, "confirmation_final")
    previous = payload["training_state"]["counters"]["estimated_flops"] if payload else 0
    best = payload["training_state"].get("best_bpb") if payload else None
    def boundary(state, model):
        nonlocal previous, best
        current = state["counters"]["estimated_flops"]
        if not profile and any(previous < int(f*t.budget_limit) <= current for f in s["evaluation_fractions"]):
            row = ev(state, screen, "screen")
            if best is None or row["bpb"] < best:
                best = row["bpb"]
                state["best_bpb"] = best
                save_training_checkpoint(output/"best.pt", model, state, tokenizer=data.identity["tokenizer"], dataset=validation)
        if not profile and any(previous < int(f*t.budget_limit) <= current for f in s["diagnostic_fractions"]):
            diag(state)
        previous = current
    rows = []
    def update(row):
        row.pop("examples")
        row.update(gpu_memory(device))
        row.update(raw_bytes_per_second=row["update_bytes"]/row["update_seconds"],
                   tokens_per_second=row["update_tokens"]/row["update_seconds"], nonfinite_events=0,
                   host_peak_ram_bytes=None)
        append(output/"training.jsonl", row)
        rows.append(row)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    start = time.perf_counter()
    try:
        if not profile and payload is None:
            ev({"counters": asdict(Counters())}, screen, "initial_screen")
        if profile:
            # Warm full AdamW updates; profiler adds overhead only to this dedicated operation.
            count = s["profile_updates"]+s["profile_warmup"]
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
                state = train(model, data, t, validation_identity=validation, max_attempts=count,
                              on_metrics=update, experiment_identity=identity)
            write_json(output/"profiler.json", dict(reported_flops=sum(e.flops for e in prof.key_averages()),
                       analytical_flops=state["counters"]["estimated_flops"],
                       limitation="Profiler omits SDPA/backward kernels; timings here include profiler overhead"))
            # A separate warm timing session without profiler, same initialized model.
            rows.clear()
            state = train(model, data, t, validation_identity=validation, resume=state,
                          max_attempts=count, on_metrics=update, experiment_identity=identity)
        else:
            state = train(model, data, t, validation_identity=validation, output=output,
                          resume=payload["training_state"] if payload else None, max_attempts=max_attempts,
                          on_metrics=update, on_boundary=boundary, should_stop=lambda: stopped, experiment_identity=identity)
            if not stopped and max_attempts is None:
                # Do not repeat a scheduled 100% observation at an exact endpoint.
                if previous != t.budget_limit:
                    ev(state, screen, "final_screen")
                if previous != t.budget_limit or 1.0 not in s['diagnostic_fractions']:
                    diag(state)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    storage = storage_report(model, optimizer_state=state["optimizer"], tokenizer=data.identity["tokenizer"], dataset=validation)
    kept = rows[s["profile_warmup"]:] if profile else rows
    import statistics
    result = dict(counters=state["counters"], budget=t.budget_limit,
                  completion="job_boundary" if stopped or max_attempts is not None else "budget_boundary",
                  unused_flops=t.budget_limit-state["counters"]["estimated_flops"], storage=storage,
                  session_elapsed_seconds=time.perf_counter()-start,
                  median_update_seconds=statistics.median(r["update_seconds"] for r in kept) if kept else None,
                  peak_memory=gpu_memory(device), training_memory=rows[-1] if rows else None)
    write_json(output/("profile.json" if profile else "result.json"), result)
    return result


def report(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(s["output"])
    summaries, lines = [], ["# FP32 IsoFLOPs feasibility", "", "Observed screening results; confirmation uses separate documents. No quality conclusion from feasibility.", ""]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for arm in s["arms"]:
        p = root/arm/"run"
        if not (p/"result.json").exists():
            continue
        result = json.loads((p/"result.json").read_text())
        validations = [json.loads(x) for x in (p/"validation.jsonl").read_text().splitlines()]
        training = [json.loads(x) for x in (p/"training.jsonl").read_text().splitlines()]
        vals = [v for v in validations if v["subset"] in {"screen", "final_screen"}]
        final = vals[-1]
        row = dict(arm=arm, final_bpb=final["bpb"], best_screen_bpb=min(v["bpb"] for v in vals),
                   confirmation_bpb=next((v["bpb"] for v in reversed(validations) if v["subset"] == "confirmation_final"), None),
                   **result["counters"], **result["storage"], **result["peak_memory"],
                   median_update_seconds=result["median_update_seconds"], unused_flops=result["unused_flops"])
        summaries.append(row)
        lines.extend([f"## {arm}", "", f"Final screening BPB: {row['final_bpb']:.6f}; best screening: {row['best_screen_bpb']:.6f}; confirmation final: {row['confirmation_bpb']}",
                      f"Consumed FLOPs: {row['estimated_flops']}; unused: {row['unused_flops']}; raw bytes: {row['raw_bytes']}",
                      f"Vocabulary archive bytes: {row['vocabulary_serialized_inference_bytes']}; median update seconds: {row['median_update_seconds']}",
                      f"Peak memory: {result['peak_memory']}. Complete storage/training-memory accounting: run/result.json; sparse spectra and factor diagnostics: diagnostics.jsonl.", ""])
        for ax, key, label in zip(axes.flat, ("estimated_flops", "raw_bytes", "active_seconds"), ("Estimated training FLOPs", "Raw training bytes", "Active training seconds")):
            ax.scatter([v[key] for v in vals], [v["bpb"] for v in vals], label=arm)
            ax.set(xlabel=label, ylabel="Validation bits / raw UTF-8 byte")
        axes[1, 1].plot([v["estimated_flops"] for v in training], [v["loss"] for v in training], label=arm)
    axes[1, 1].set(xlabel="Estimated training FLOPs", ylabel="Training mean token NLL")
    for ax in axes.flat:
        ax.legend()
    root.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(root/"curves.png")
    plt.close(fig)
    if summaries:
        with (root/"summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    (root/"report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["validate", "preflight", "profile", "train", "resume", "evaluate", "confirm", "diagnose", "report"])
    parser.add_argument("--config", default="configs/vocab_scaling_toy.yaml")
    parser.add_argument("--arm", default="baseline8k")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-attempts", type=int, help="Graceful simulated job boundary; global budget remains frozen")
    args = parser.parse_args(argv)
    s = load_settings(args.config)
    torch.set_num_threads(s["threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.operation == "validate":
        print("Configuration valid; artifact availability is checked by preflight")
    elif args.operation == "preflight":
        result = preflight(s)
        Path(s["output"]).mkdir(parents=True, exist_ok=True)
        write_json(Path(s["output"])/"preflight.json", result)
        print(json.dumps(result, indent=2))
    elif args.operation == "report":
        report(s)
    else:
        result = run(s, args.arm, resume=args.resume or args.operation == "resume", max_attempts=args.max_attempts,
                     profile=args.operation == "profile", operation="train" if args.operation in {"train", "resume", "profile"} else args.operation)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

