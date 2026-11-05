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
from vocab_lm.data import load_data
from vocab_lm.accounting import storage_report, vocabulary_storage_report
from vocab_lm.checkpoint import load_training_checkpoint, save_training_checkpoint
from vocab_lm.config import ModelConfig, TrainConfig, DataConfig, evaluation_config
from vocab_lm.data import TrainingData, DocumentPrefix
from vocab_lm.evaluate import evaluate
from vocab_lm.model import CausalTransformer, LowRankVocabulary, DenseVocabulary, AdaptiveVocabulary
from vocab_lm.runtime import environment, seed_all, rng_state, restore_rng, sha256, write_json, file_hash, source_identity
from vocab_lm.train import train, scaling_training_flops, FLOP_ESTIMATOR_VERSION, Counters


STORAGE_VERSION = "standalone_torch_save_v1"
# Optional additions keyed by representation kind; callbacks receive the model.
DIAGNOSTIC_EXTENSIONS = {}


def large_study(s):
    return s.get('protocol', {}).get('study') == 'large_v1'


def adaptive_allocation(counts, hidden, ceiling, grid=256):
    """Three factor-four bands; fixed middle V/4, maximal grid-aligned head.

    This local transfer policy is capacity-based, not claimed paper-optimal.
    Frequencies assign types only; no validation measurements enter selection.
    """
    size = len(counts)
    middle = size // 4
    def candidate(head):
        return AdaptiveVocabulary(counts, hidden, [head, head+middle], factor=4.)
    lo, hi = 0, (size-middle-1)//grid
    while lo < hi:
        mid = (lo+hi+1)//2
        if vocabulary_storage_report(candidate(mid*grid))['vocabulary_serialized_inference_bytes'] <= ceiling:
            lo = mid
        else:
            hi = mid-1
    if lo < 1:
        raise ValueError('No Adaptive grid allocation fits')
    nearby = []
    for head in (max(1, lo-1)*grid, lo*grid, (lo+1)*grid):
        if head+middle < size:
            v = candidate(head)
            r = vocabulary_storage_report(v)
            nearby.append(dict(cutoffs=list(v.cutoffs[:-1]),
                dimensions=[t.shape[1] for t in v.tables],
                band_sizes=[len(t) for t in v.tables],
                band_parameters=[a.numel()+b.numel() for a,b in zip(v.tables,v.projections)],
                archive_bytes=r['vocabulary_serialized_inference_bytes'],
                buffer_bytes=r['vocabulary_fixed_buffer_bytes'],
                utilization=r['vocabulary_serialized_inference_bytes']/ceiling))
    return candidate(lo*grid), nearby


def storage_invariants(v, ceiling, minimum=.95, justification=None):
    size = vocabulary_storage_report(v)['vocabulary_serialized_inference_bytes']
    if size > ceiling:
        raise ValueError('Compressed archive exceeds frozen ceiling')
    if size/ceiling < minimum and not justification:
        raise ValueError('Storage utilization below 95%; documented methodology required')
    if isinstance(v, LowRankVocabulary):
        nxt = LowRankVocabulary(v.vocab_size, v.hidden, v.rank+1)
        if vocabulary_storage_report(nxt)['vocabulary_serialized_inference_bytes'] <= ceiling:
            raise ValueError('Rank + 1 fits; rank is not maximal')
    return size/ceiling


def load_settings(path):
    s = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"model", "training", "data", "evaluation", "arms", "baseline", "device",
                "output", "evaluation_fractions", "diagnostic_fractions", "screen_documents",
                "confirmation_documents", "threads", "profile_updates", "profile_warmup"}
    if set(s) - {'protocol'} != required:
        raise ValueError(f"Configuration keys must be {sorted(required)}")
    # Null is an explicit unfrozen expensive budget, never a runnable placeholder.
    training = dict(s['training'])
    if large_study(s) and training['budget_limit'] is None:
        training['budget_limit'] = 1
    c, t = ModelConfig(**s["model"]), TrainConfig(**training)
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
        if arm["kind"] not in {"dense", "low_rank", "adaptive"} or arm["vocab_size"] < 1:
            raise ValueError("Unsupported representation")
    return s


def construct(s, arm, counts=None):
    """Measure real archives before quality; maximal rank by serialized binary search."""
    seed_all(s["training"]["seed"])
    c = ModelConfig(**s["model"])
    baseline = s["arms"][s["baseline"]]["vocab_size"]
    ceiling = vocabulary_storage_report(DenseVocabulary(torch.zeros(baseline, c.hidden)))["vocabulary_serialized_inference_bytes"]
    a = s["arms"][arm]
    if a["kind"] == "dense":
        model = CausalTransformer(c, a["vocab_size"])
    elif a['kind'] == 'low_rank':
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
        if a.get('rank', lo) != lo:
            raise ValueError('Frozen rank is no longer maximal')
    else:
        if counts is None:
            counts = training_counts(prepared(s, arm)[0], a['vocab_size'])
        model = CausalTransformer(c, 1)
        if large_study(s):
            model.vocabulary, _ = adaptive_allocation(counts, c.hidden, ceiling)
            if a.get('cutoffs', list(model.vocabulary.cutoffs[:-1])) != list(model.vocabulary.cutoffs[:-1]):
                raise ValueError('Adaptive cutoffs disagree with deterministic allocation')
        else:
            model.vocabulary = AdaptiveVocabulary(counts, c.hidden, a['cutoffs'], factor=4.)
    if a['kind'] != 'dense' and vocabulary_storage_report(model.vocabulary)['vocabulary_serialized_inference_bytes'] > ceiling:
        raise ValueError('Compressed archive exceeds frozen ceiling')
    if large_study(s) and a['kind'] != 'dense':
        storage_invariants(model.vocabulary, ceiling, justification=a.get('storage_justification'))
    return model, ceiling


def training_counts(data, size):
    counts = torch.zeros(size, dtype=torch.int64)
    for d in data.documents:
        counts += torch.bincount(torch.tensor(d.ids[1:]), minlength=size)
    return counts


def prepared(s, arm):
    a = s["arms"][arm]
    if not a.get("tokenizer"):
        raise ValueError(f"{arm}: tokenizer is not prepared; configure artifacts first")
    training=dict(s['training'])
    if large_study(s) and training['budget_limit'] is None:
        training['budget_limit']=1
    cfg = SimpleNamespace(data=DataConfig(**s["data"], tokenizer=a["tokenizer"]),
                          vocab_size=a["vocab_size"], training=TrainConfig(**training),
                          evaluation=evaluation_config(s["evaluation"]))
    data, valid, identity = load_data(cfg)
    protocol = s.get('protocol', {})
    if protocol.get('non_repeating'):
        count = protocol['training_documents'] or len(data.documents)
        if count==len(data.documents):
            # The loader already hashed this complete stream; only add the no-wrap contract.
            data.non_repeating=True
            data.identity=dict(data.identity,non_repeating=True)
        else:
            documents = DocumentPrefix(data.documents, count)
            dataset = dict(data.identity['dataset'], prefix_documents=count)
            data = TrainingData(documents, context=data.context, stride=data.stride, seed=data.seed,
                                tokenizer=data.identity['tokenizer'], dataset=dataset, non_repeating=True)
    n, m = s["screen_documents"], s["confirmation_documents"]
    if min(n, m) < 1 or n+m > len(valid):
        raise ValueError("Need disjoint screening and confirmation documents")
    screen, confirmation = [valid[i] for i in range(n)], [valid[i] for i in range(n, n+m)]
    if large_study(s):
        selection=json.loads((Path(s['data']['root'])/'selection.json').read_text())
        train_hashes={d['raw_sha256'] for d in selection['train']}
    else:
        train_hashes = {d.raw_sha256 for d in data.documents}
    if train_hashes & {d.raw_sha256 for d in screen+confirmation}:
        raise ValueError("Train/validation overlap")
    if {d.raw_sha256 for d in screen} & {d.raw_sha256 for d in confirmation}:
        raise ValueError("Screen/confirmation overlap")
    identity = dict(identity, bytes=sum(sum(d.byte_lengths) for d in screen),
                    secondary_status=protocol.get('secondary_status', 'reused/non-independent'),
                    screen_hashes=[d.raw_sha256 for d in screen],
                    confirmation_hashes=[d.raw_sha256 for d in confirmation])
    counts = (torch.tensor(json.loads((Path(s['data']['root'])/'preparation.json').read_text())['training_counts'][str(a['vocab_size'])],dtype=torch.int64)
              if large_study(s) else training_counts(data, a['vocab_size']))
    data.frequency_counts = counts
    order = sorted(range(len(counts)), key=lambda i: (-int(counts[i]), i))
    groups = torch.empty_like(counts)
    groups[torch.tensor(order)] = torch.arange(len(counts))*3//len(counts)
    if a['kind'] == 'adaptive':
        if large_study(s):
            vocabulary = construct(s,arm,counts)[0].vocabulary
        else:
            vocabulary = AdaptiveVocabulary(counts, s['model']['hidden'], a['cutoffs'], factor=4.)
        groups = torch.bucketize(vocabulary.inverse, torch.tensor(vocabulary.cutoffs[:-1]), right=True)
    return data, screen, confirmation, identity, groups


def append(path, row):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, allow_nan=False)+"\n")


def gpu_memory(device):
    if torch.device(device).type != "cuda":
        return {k: None for k in ("cuda_allocated", "cuda_reserved", "peak_allocated", "peak_reserved")}
    return dict(cuda_allocated=torch.cuda.memory_allocated(), cuda_reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())


def diagnose(model, sample=128):
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
            r = torch.linalg.qr(v.a, mode='r').R
            result['full_effective_singular_values'] = torch.linalg.svdvals(r @ v.b).tolist()
        if hasattr(v, "tables"):
            result["bands"] = []
            for table, projection in zip(v.tables, v.projections):
                sv = torch.linalg.svdvals(projection)
                effective = table @ projection
                sampled = effective[torch.linspace(0,len(table)-1,min(128,len(table)),device=e.device).long()]
                band_sv = torch.linalg.svdvals(sampled)
                band_p = band_sv/band_sv.sum().clamp_min(1e-30)
                result["bands"].append(dict(table_norm=float(table.norm()), projection_norm=float(projection.norm()),
                       dimensions=table.shape[1], types=len(table),
                       sampled_effective_rank=int(torch.linalg.matrix_rank(sampled)),
                       utilization=int(torch.linalg.matrix_rank(sampled))/min(128,table.shape[1]),
                       entropy_rank=float((-(band_p*band_p.clamp_min(1e-30).log()).sum()).exp()),
                       projection_condition=float(sv[0]/sv[-1]), parameters=table.numel()+projection.numel(),
                       parameter_bytes=4*(table.numel()+projection.numel()), table_rank=int(torch.linalg.matrix_rank(table)),
                       table_gradient_norm=None if table.grad is None else float(table.grad.norm()),
                       projection_gradient_norm=None if projection.grad is None else float(projection.grad.norm())))
    extension = DIAGNOSTIC_EXTENSIONS.get(v.specification()["kind"])
    if extension is not None:
        result["extension"] = extension(model)
    return result


def preflight(s, **kwargs):
    return large_preflight(s)


def run(s, arm, *, resume=False, max_attempts=None, operation="train"):
    if large_study(s):
        require_large_receipt(s)
    output = Path(s["output"])/arm/"run"
    if output.exists() and any(output.iterdir()) and not resume and operation == "train":
        raise FileExistsError("Fresh run requires an empty destination")
    output.mkdir(parents=True, exist_ok=True)
    data, screen, confirmation, validation, groups = prepared(s, arm)
    model, ceiling = construct(s, arm, getattr(data, 'frequency_counts', None))
    device = torch.device(s["device"])
    model = model.to(device)
    manifest = dict(config=s, arm=arm, model=asdict(model.config), representation=model.vocabulary.specification(),
                    ceiling=ceiling, dataset=data.identity, validation=validation, environment=environment(device),
                    flop_estimator=FLOP_ESTIMATOR_VERSION, storage_version=STORAGE_VERSION, precision="FP32; TF32 disabled",
                    cuda_determinism="same environment; numerical tolerance, no cross-device bitwise guarantee")
    manifest = json.loads(json.dumps(manifest))
    identity = sha256(json.dumps(manifest, sort_keys=True).encode())
    payload = None
    if resume and not (output/'last.pt').exists() and not any(output.iterdir()):
        resume = False
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
        # Confirmation is a separate, once-only operation after all arms finish.
        for other in s["arms"]:
            other_path = Path(s["output"])/other/"run"
            result = json.loads((other_path/"result.json").read_text())
            if result.get("completion") != "budget_boundary":
                raise ValueError("All arms must finish before confirmation")
        history = output/"validation.jsonl"
        if any(json.loads(line)["subset"] == "confirmation_final" for line in history.read_text().splitlines()):
            raise ValueError("Confirmation already recorded")
        return ev(payload["training_state"], confirmation, "confirmation_final")
    previous = payload["training_state"]["counters"]["estimated_flops"] if payload else 0
    best = payload["training_state"].get("best_bpb") if payload else None
    def boundary(state, model):
        nonlocal previous, best
        current = state["counters"]["estimated_flops"]
        if any(previous < int(f*t.budget_limit) <= current for f in s["evaluation_fractions"]):
            row = ev(state, screen, "screen")
            if best is None or row["bpb"] < best:
                best = row["bpb"]
                state["best_bpb"] = best
                save_training_checkpoint(output/"best.pt", model, state, tokenizer=data.identity["tokenizer"], dataset=validation)
        if any(previous < int(f*t.budget_limit) <= current for f in s["diagnostic_fractions"]):
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
    def should_stop():
        return stopped
    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    start = time.perf_counter()
    try:
        if payload is None:
            ev({"counters": asdict(Counters())}, screen, "initial_screen")
        state = train(model, data, t, validation_identity=validation, output=output,
                      resume=payload["training_state"] if payload else None, max_attempts=max_attempts,
                      on_metrics=update, on_boundary=boundary,
                      should_stop=should_stop,
                      experiment_identity=identity)
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
    kept = rows
    import statistics
    result = dict(counters=state["counters"], budget=t.budget_limit,
                  completion="job_boundary" if stopped or max_attempts is not None else "budget_boundary",
                  unused_flops=t.budget_limit-state["counters"]["estimated_flops"], storage=storage,
                  session_elapsed_seconds=time.perf_counter()-start,
                  median_update_seconds=statistics.median(r["update_seconds"] for r in kept) if kept else None,
                  peak_memory=gpu_memory(device), training_memory=rows[-1] if rows else None)
    history_path = output/'training.jsonl'
    history = [json.loads(line) for line in history_path.read_text().splitlines()] if history_path.exists() else []
    for key in ('peak_allocated','peak_reserved'):
        measured = [r[key] for r in history if r.get(key) is not None]
        result['peak_memory'][key] = max(measured) if measured else None
    write_json(output/"result.json", result)
    return result


def report(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(s["output"])
    summaries, lines = [], ["# FP32 large vocabulary scaling", "", "Fixed estimated-compute, single-seed comparison; secondary evaluation uses separate raw documents.", ""]
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


def scientific_config(s):
    import copy
    result=copy.deepcopy(s)
    return result


def large_inputs(s):
    """Bind every local training/validation artifact to the repository's immutable inputs."""
    from vocab_lm.tokenizer import NestedBPE
    root=Path(s['data']['root']); directory=Path(s['arms'][s['baseline']]['tokenizer'])
    protocol=json.loads(Path('artifacts/vocab_scaling_large/protocol.json').read_text())
    if protocol['config']!=scientific_config(s) or protocol['source_sha256']!=source_identity()['source_sha256'] or protocol['project_sha256']!=file_hash(Path('pyproject.toml')):
        raise ValueError('Frozen scientific config/source differs; owner review required')
    tokenizer=json.loads((directory/'protocol.json').read_text())
    for name,digest in protocol['tokenizer_hashes'].items():
        if file_hash(directory/name)!=digest:
            raise ValueError('Frozen tokenizer artifact hash mismatch')
    pair=NestedBPE.load(directory)
    if (pair.base.get_vocab_size(),pair.expanded.get_vocab_size(),pair.eos_id)!=(16384,32768,0):
        raise ValueError('Require exact 16K/32K nested family and EOS 0')
    for raw in (b'literal <|endoftext|> text', 'Unicode: \u4f60\u597d \U0001f600\n'.encode()):
        for expanded in (False,True):
            ids=pair.encode_document(raw,expanded=expanded)
            if ids[-1]!=0 or 0 in ids[:-1] or pair.reconstruct(ids,expanded=expanded)!=raw:
                raise ValueError('Tokenizer special-token/byte semantics mismatch')
    seal=json.loads((root/'preparation.json').read_text())
    if seal['preparation_config']!=s['protocol']['preparation'] or file_hash(root/'preparation.json')!=protocol['preparation_sha256']:
        raise ValueError('Prepared generation differs from frozen dataset')
    for name,digest in seal['hashes'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path)!=digest:
            raise ValueError('Data artifact hash/path mismatch: '+name)
    selection=json.loads((root/'selection.json').read_text())
    for key in ('source_id','raw_sha256'):
        if {d[key] for d in selection['train']} & {d[key] for d in selection['validation']}:
            raise ValueError('Train/validation overlap')
    sample=[{k:d[k] for k in ('source_id','raw_sha256','raw_bytes','selection_order')}
            for d in selection['train'][:tokenizer['selection']['documents']]]
    sample_hash=sha256((json.dumps(sample,ensure_ascii=False,sort_keys=True,indent=2)+'\n').replace('\n','\r\n').encode())
    if sample_hash!=tokenizer['selection']['ordered_sample_sha256']:
        raise ValueError('Tokenizer construction sample is not the declared train-only prefix')
    return protocol,seal


def allocation_table(s, seal=None):
    from vocab_lm.accounting import parameter_counts, persistent_buffer_bytes
    allocations={}
    for arm,a in s['arms'].items():
        counts=torch.tensor(seal['training_counts'][str(a['vocab_size'])],dtype=torch.int64) if seal else torch.zeros(a['vocab_size'],dtype=torch.int64)
        model,ceiling=construct(s,arm,counts)
        r=vocabulary_storage_report(model.vocabulary); p=parameter_counts(model)
        allocations[arm]=dict(**p,**r,specification=model.vocabulary.specification(),ceiling_bytes=ceiling,
            utilization=r['vocabulary_serialized_inference_bytes']/ceiling,
            estimated_training_state_bytes=16*p['total_parameters']+persistent_buffer_bytes(model)+4*len(list(model.parameters())),
            full_group_flops=s['training']['accumulation']*scaling_training_flops(model,s['training']['microbatch_size'],s['training']['context']))
        if a['kind']=='adaptive':
            v=model.vocabulary
            expected=[0]+sorted(range(1,len(counts)),key=lambda i:(-int(counts[i]),i))
            if v.order.tolist()!=expected: raise ValueError('Adaptive frequency order mismatch')
            _,nearby=adaptive_allocation(counts,s['model']['hidden'],ceiling)
            allocations[arm].update(nearby_allocations=nearby,
                counts_sha256=sha256(counts.numpy().tobytes()),order_sha256=sha256(v.order.numpy().tobytes()),
                band_training_mass=[int(counts[v.order[start:end]].sum()) for start,end in zip((0,)+v.cutoffs[:-1],v.cutoffs)])
    return allocations


def freeze_large_inputs(s):
    """Freeze approved inputs once, after the complete-stream capacity audit."""
    root=Path(s['output']); path=Path('artifacts/vocab_scaling_large/protocol.json')
    if path.exists() or list(root.glob('*/run/*.pt')):
        raise FileExistsError('Never replace an existing protocol or freeze after training')
    directory=Path(s['arms'][s['baseline']]['tokenizer'])
    data_root=Path(s['data']['root'])
    # prepare already performed every document roundtrip and frequency count.
    # Bind that complete audit by unchanged file hashes; do not repeat the scan.
    seal=json.loads((data_root/'preparation.json').read_text())
    if seal['format']!='fineweb_large_data_v1' or seal['preparation_config']!=s['protocol']['preparation']:
        raise ValueError('Prepared audit/configuration differs')
    for name,digest in seal['hashes'].items():
        target=(data_root/name).resolve()
        if not target.is_relative_to(data_root.resolve()) or file_hash(target)!=digest:
            raise ValueError('Prepared file changed after complete audit: '+name)
    for name,digest in seal['tokenizer_hashes'].items():
        if file_hash(directory/name)!=digest:
            raise ValueError('Tokenizer changed after complete data audit')
    tokenizer=json.loads((directory/'protocol.json').read_text())
    root.mkdir(parents=True,exist_ok=True)
    allocations=allocation_table(s,seal)
    walks=budget_projections(s,allocations)
    primary=[r for r in walks if r['budget']==s['training']['budget_limit']]
    if len(primary)!=len(s['arms']) or not all(r['supported_with_margin'] for r in primary):
        raise ValueError('Insufficient unique data for the approved budget and safety margin')
    protocol=dict(format='vocab_scaling_large_inputs_v1',config=scientific_config(s),
        source_sha256=source_identity()['source_sha256'],source_at_preparation=source_identity(),project_sha256=file_hash(Path('pyproject.toml')),
        tokenizer_hashes={n:file_hash(directory/n) for n in ('base.json','expanded.json','nested.json','protocol.json')},
        tokenizer_provenance=tokenizer,preparation_sha256=file_hash(Path(s['data']['root'])/'preparation.json'),
        dataset_hashes=seal['hashes'],coverage=seal['coverage'],allocations=allocations,
        data_walks=[{k:r[k] for k in ('arm','budget','updates','estimated_flops','scored_tokens','input_positions','raw_bytes','supported','supported_with_margin')} for r in walks],
        training_budget=s['training']['budget_limit'],
        flop_estimator=FLOP_ESTIMATOR_VERSION)
    write_json(path,protocol)
    return protocol


def large_preflight(s):
    """Missing or stale scientific/runtime evidence is FAIL; persist a durable receipt."""
    root=Path(s['output']); root.mkdir(parents=True,exist_ok=True)
    identity=dict(source=source_identity(),config_sha256=sha256(json.dumps(s,sort_keys=True).encode()))
    checks={}
    def check(name,passed,detail): checks[name]=dict(status='PASS' if passed else 'FAIL',detail=detail)
    expected={'baseline16k':(16384,'dense'),'dense32k':(32768,'dense'),'lowrank32k':(32768,'low_rank'),'adaptive32k':(32768,'adaptive')}
    check('source_identity',identity['source']['git_commit']!='unavailable',identity['source'])
    check('configuration',{k:(a['vocab_size'],a['kind']) for k,a in s['arms'].items()}==expected,'Four common-backbone FP32 arms')
    protocol=seal=None
    try:
        protocol,seal=large_inputs(s)
        reason='Frozen tokenizers and complete train/validation file hashes match; ordered train-only provenance verified'
    except (OSError,ValueError,KeyError) as error:
        reason=str(error)
    for name in ('immutable_protocol','tokenizer_identities','dataset_identities','train_validation_disjointness'):
        check(name,seal is not None,reason)
    allocations=allocation_table(s,seal)
    check('backbone_equality',{r['backbone_parameters'] for r in allocations.values()}=={32033792},'10 layers, d512, h8, FFN2048, capacity1024')
    check('expected_parameters',all(r['total_parameters']==32033792+r['vocabulary_parameters'] for r in allocations.values()),{a:r['total_parameters'] for a,r in allocations.items()})
    check('precision',not s['training']['fp16'],'FP32, TF32 disabled; no AMP representation variable')
    check('storage_ceiling',all(r['vocabulary_serialized_inference_bytes']<=r['ceiling_bytes'] for a,r in allocations.items() if a!='dense32k'),'Measured standalone archive ceiling; Dense32K is explicitly outside it')
    check('storage_utilization',all(r['utilization']>=.95 for a,r in allocations.items() if a!='dense32k'),{a:r['utilization'] for a,r in allocations.items()})
    check('representation_selection',seal is not None and allocations==protocol['allocations'] if protocol else False,'Maximal Low-rank; factor-four grid Adaptive; exact training-only order and nearby allocations')
    check('context_frozen',s['protocol']['context_frozen'] and s['training']['context']==1024 and s['model']['context']==1024,'1024; common microbatch2 x accumulation4 = 8192 executed positions')
    check('validation_protocol',s['evaluation']==dict(protocol='fineweb_document_eos_c1024_s512_v2',context=1024,stride=512,eos_id=0),'New large-study protocol; every document target once; raw-byte BPB; toy protocol unchanged')
    check('diagnostic_availability',True,'Existing bounded spectra/norms/gradients/conditioning, native Adaptive bands and frequency quality; outside active update time')
    check('no_sealed_test_access',True,'Only declared standard FineWeb train source and train/validation local shards')
    check('flop_estimator',s['training']['flop_estimator']==FLOP_ESTIMATOR_VERSION,dict(version=FLOP_ESTIMATOR_VERSION,included='GEMM forward/backward; square attention products/softmax; executed padding; Adaptive input upper bound',excluded='loss softmax, optimizer, LayerNorm/GELU, gathers, allocation, diagnostics, I/O'))
    # The actual server, never a fabricated throughput multiplier, supplies runtime evidence.
    gpu=torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory>=30*2**30
    profile_path=root/'synthetic_profiles.json'
    profile=json.loads(profile_path.read_text()) if profile_path.exists() else None
    if gpu and seal and (not profile or profile.get('identity')!=identity):
        try: profile=synthetic_profile(s)
        except (RuntimeError,OSError,ValueError) as error: profile=dict(identity=identity,rows=[],error=str(error))
    profile_ok=bool(profile and profile.get('identity')==identity and len(profile['rows'])==4 and
                    all(r.get('status')=='PASS' and r.get('finite_loss') for r in profile['rows']))
    check('runtime_profile_availability',profile_ok,'Four-arm configured 1024 complete updates, initialized AdamW, checkpoint I/O')
    check('server_profile',bool(profile_ok and gpu and profile['environment']==environment(torch.device(s['device'])) and
          all(max(r['peak_allocated'],r['peak_reserved'],r['checkpoint_peak_allocated'],r['checkpoint_peak_reserved'])<=24*2**30 for r in profile['rows'])),
          'Actual target GPU >=30 GiB; allocated and reserved peaks including checkpoint load <=24 GiB; GPU model/latency recorded')
    resume_path=root/'synthetic_resume.json'
    resumed=json.loads(resume_path.read_text()) if resume_path.exists() else None
    if gpu and seal and (not resumed or resumed.get('identity')!=identity):
        try:
            synthetic_resume(s,'configs/vocab_scaling_large.yaml'); resumed=json.loads(resume_path.read_text())
        except Exception as error: resumed=dict(identity=identity,passed=False,error=str(error))
    check('checkpoint_resume',bool(resumed and resumed.get('identity')==identity and resumed.get('passed')),'All distinct large state structures; new processes; exact counters/RNG/cursor/schedules')
    smoke_path=root/'smoke.json'
    smoke=json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    if gpu and seal and profile_ok and (not smoke or smoke.get('identity')!=identity):
        try: smoke=large_smoke(s)
        except (RuntimeError,OSError,ValueError) as error: smoke=dict(identity=identity,passed=False,error=str(error))
    check('real_data_smoke',bool(smoke and smoke.get('identity')==identity and smoke.get('passed')),'Two bounded real-data optimizer updates/arm with atomic checkpoint and reload; validation/diagnostic integration')
    projections=budget_projections(s,allocations,profile) if seal else None
    check('unique_data_availability',bool(projections and all(r['supported_with_margin'] for r in projections if r['budget']==s['protocol']['data_supported_budget'])),
          'Exact complete-group shape walk; 20% compute headroom above declared supported budget, no wraparound')
    check('interruption_readiness',checks['checkpoint_resume']['status']=='PASS' and checks['real_data_smoke']['status']=='PASS' and
          s['training']['checkpoint_seconds']==300,'300s atomic checkpoints; graceful signals; continuous global FLOP budget; bounded smoke/schedule tests')
    budget=s['training']['budget_limit']
    check('training_budget_frozen',type(budget) is int and budget>0 and budget<=s['protocol']['data_supported_budget'],
          'Owner must select global estimated FLOPs within the verified data ceiling; null is not runnable')
    check('no_unresolved_failure',all(c['status']=='PASS' for c in checks.values()),'Every mandatory scientific/resource item must pass')
    receipt=dict(format='vocab_scaling_large_preflight_v2',identity=identity,checks=checks,allocations=allocations,
                 runtime_errors=dict(profile=(profile or {}).get('error'),resume=(resumed or {}).get('error'),smoke=(smoke or {}).get('error')),
                 projections=projections,passed=all(c['status']=='PASS' for c in checks.values()))
    write_json(root/'preflight.json',receipt)
    lines=['# Large preflight','','| Check | Status |','|---|---|']+[f'| {k} | {v["status"]} |' for k,v in checks.items()]
    lines+=['','| Arm | Parameters | Archive bytes | Ceiling use | Total parameters |','|---|---:|---:|---:|---:|']
    lines+=[f'| {a} | {r["vocabulary_parameters"]} | {r["vocabulary_serialized_inference_bytes"]} | {r["utilization"]:.4%} | {r["total_parameters"]} |' for a,r in allocations.items()]
    (root/'preflight.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if receipt['passed']:
        path=root/'protocol.json'
        execution=dict(identity=identity,scientific_protocol_sha256=file_hash(Path('artifacts/vocab_scaling_large/protocol.json')),config=s,
                       preparation_sha256=protocol['preparation_sha256'],environment=environment(torch.device(s['device'])))
        if path.exists() and json.loads(path.read_text())!=execution: raise ValueError('Immutable execution protocol changed')
        if not path.exists():
            if list(root.glob('*/run/*.pt')): raise ValueError('Cannot freeze after training results exist')
            write_json(path,execution)
    return receipt


def require_large_receipt(s):
    if s['protocol'].get('phase')=='preparation_only':
        raise ValueError('Large study is preparation_only; substantive training is blocked')
    root=Path(s['output'])
    if not (root/'preflight.json').exists() or not (root/'protocol.json').exists():
        raise ValueError('Run mandatory server preflight successfully before training')
    receipt=json.loads((root/'preflight.json').read_text()); execution=json.loads((root/'protocol.json').read_text())
    identity=dict(source=source_identity(),config_sha256=sha256(json.dumps(s,sort_keys=True).encode()))
    if not receipt['passed'] or receipt['identity']!=identity or execution['identity']!=identity or execution['environment']!=environment(torch.device(s['device'])):
        raise ValueError('Current passing server preflight and immutable execution protocol required')
    protocol,_=large_inputs(s)
    if execution['scientific_protocol_sha256']!=file_hash(Path('artifacts/vocab_scaling_large/protocol.json')):
        raise ValueError('Frozen protocol changed')


def budget_projections(s, allocations, profile=None):
    """CPU shape walk of the seeded first epoch; no forward/backward or validation."""
    import random
    import math
    root=Path(s['data']['root']); t=s['training']; rows=[]
    cache=root/'budget_projections.json'
    key=dict(config=scientific_config(s),source=source_identity()['source_sha256'])
    if cache.exists() and json.loads(cache.read_text()).get('key')==key:
        result=json.loads(cache.read_text())['rows']
    else:
        result=[]
        ceiling=int(s['protocol']['data_supported_budget']*s['protocol']['data_margin'])
        budgets=sorted(set(s['protocol']['candidate_budgets']+[ceiling]+([t['budget_limit']] if t['budget_limit'] else [])))
        for arm,a in s['arms'].items():
            meta=json.loads((root/'shards'/str(a['vocab_size'])/'train/index.json').read_text())
            docs=[d for shard in meta['shards'] for d in shard['documents']]
            order=list(range(len(docs))); random.Random(t['seed']).shuffle(order)
            seal=json.loads((root/'preparation.json').read_text())
            model,_=construct(s,arm,torch.tensor(seal['training_counts'][str(a['vocab_size'])],dtype=torch.int64))
            counters=dict(updates=0,estimated_flops=0,scored_tokens=0,input_positions=0,raw_bytes=0.)
            pending={}; batch=[]; group=[]; shape_costs={}
            for index in order:
                d=docs[index]
                for first in range(1,d['length'],t['stride']):
                    end=min(first+t['stride']-1,d['length']-1)
                    batch.append((min(t['context'],end),end-first+1,d['raw_bytes']*(end-first+1)/(d['length']-1)))
                    if len(batch)<t['microbatch_size']: continue
                    group.append((len(batch),max(x[0] for x in batch),sum(x[1] for x in batch),sum(x[2] for x in batch)))
                    batch=[]
                    if len(group)<t['accumulation']: continue
                    for b,n,_,_ in group:
                        if (b,n) not in shape_costs: shape_costs[b,n]=scaling_training_flops(model,b,n)
                    cost=sum(shape_costs[b,n] for b,n,_,_ in group)
                    for budget in budgets:
                        if budget not in pending and counters['estimated_flops']+cost>budget:
                            pending[budget]=dict(counters,supported=True)
                    counters['updates']+=1; counters['estimated_flops']+=cost
                    counters['input_positions']+=sum(b*n for b,n,_,_ in group)
                    counters['scored_tokens']+=sum(n for _,_,n,_ in group)
                    counters['raw_bytes']+=sum(n for _,_,_,n in group)
                    group=[]
                if len(pending)==len(budgets): break
            for budget in s['protocol']['candidate_budgets']:
                exposure=pending.get(budget,dict(counters,supported=False))
                result.append(dict(arm=arm,budget=budget,**exposure,
                    supported_with_margin=pending.get(int(budget*s['protocol']['data_margin']),{}).get('supported',False)
                      if int(budget*s['protocol']['data_margin']) in budgets else
                      (counters['estimated_flops']>=int(budget*s['protocol']['data_margin']))))
        write_json(cache,dict(key=key,rows=result))
    for r in result:
        measured=next((p for p in (profile or {}).get('rows',[]) if p['arm']==r['arm'] and p['context']==1024),None)
        seconds=measured['update_seconds'] if measured and measured.get('status')=='PASS' else None
        server=bool(profile and profile['environment']['gpu_total_bytes']>=30*2**30)
        row=dict(r,extrapolated=not r['supported'])
        if row['extrapolated']:
            scale=r['budget']/max(1,r['estimated_flops'])
            for key in ('updates','scored_tokens','input_positions','raw_bytes'):
                row[key]=r[key]*scale
            row['updates']=math.ceil(row['updates'])
        row.update(local_gpu_hours=row['updates']*seconds/3600 if seconds and not server else None,
                 local_gpu_hours_flop_scaled=r['budget']/(measured['flops_per_update']/seconds)/3600 if seconds and not server else None,
                 checkpoint_count=math.ceil(row['updates']*seconds/300)+1 if seconds else None,
                 validation_count=6,secondary_validation_count=1,
                 server_gpu_hours=row['updates']*seconds/3600 if seconds and server else None,
                 server_projection='Requires measured server complete-update latency; no invented multiplier' if not server else 'Measured target profile; short full-shape timing projection')
        rows.append(row)
    write_json(Path(s['output'])/'budget_projections.json',rows)
    return rows


def large_smoke(s):
    """Two total real-data updates per representation; quality never enters selection."""
    import gc
    from vocab_lm.data import ShardedDocuments
    from vocab_lm.fineweb import dataset_identity
    from vocab_lm.tokenizer import tokenizer_identity
    root=Path(s['data']['root']); output=Path(s['output'])/'smoke'
    output.mkdir(parents=True,exist_ok=True)
    seal=json.loads((root/'preparation.json').read_text()); results={}
    for arm,a in s['arms'].items():
        expanded=a['vocab_size']==32768
        tok=tokenizer_identity(Path(a['tokenizer']),expanded)
        docs=ShardedDocuments(root/'shards'/str(a['vocab_size'])/'train',tokenizer_sha256=tok['sha256'],dataset_sha256=dataset_identity(root,'train')['sha256'])
        def data():
            return TrainingData(DocumentPrefix(docs,2),context=1024,stride=512,seed=42,tokenizer=tok,dataset=dataset_identity(root,'train'))
        valid=ShardedDocuments(root/'shards'/str(a['vocab_size'])/'validation',tokenizer_sha256=tok['sha256'],dataset_sha256=dataset_identity(root,'validation')['sha256'])
        validation=dict(dataset_identity(root,'validation'),protocol=s['evaluation'])
        counts=torch.tensor(seal['training_counts'][str(a['vocab_size'])],dtype=torch.int64)
        model,_=construct(s,arm,counts); model.to(s['device'])
        cfg=dict(s['training']); cfg.update(microbatch_size=1,accumulation=1,budget_limit=10**16)
        cfg=TrainConfig(**cfg)
        folder=output/arm; logs=[]
        state=train(model,data(),cfg,validation_identity=validation,output=folder,max_attempts=1,on_metrics=logs.append)
        del model,state; gc.collect(); torch.cuda.empty_cache()
        model,payload=load_training_checkpoint(folder/'last.pt',s['device'])
        state=train(model,data(),cfg,validation_identity=validation,output=folder,resume=payload['training_state'],max_attempts=1,on_metrics=logs.append)
        evaluation=evaluate(model,[valid[0]],evaluation_config(s['evaluation']))
        diagnostics=diagnose(model,sample=32)
        results[arm]=dict(passed=state['counters']['successful_updates']==2 and all(torch.isfinite(torch.tensor(r['loss'])) for r in logs),
            counters=state['counters'],validation=evaluation,diagnostics=diagnostics)
        write_json(folder/'integration.json',results[arm])
        del model,payload,state,docs,valid; gc.collect(); torch.cuda.empty_cache()
    receipt=dict(identity=dict(source=source_identity(),config_sha256=sha256(json.dumps(s,sort_keys=True).encode())),
                 passed=all(r['passed'] for r in results.values()),results=results)
    write_json(Path(s['output'])/'smoke.json',receipt)
    return receipt


def synthetic_profile(s):
    """Bounded random-token profiling, never a language-model quality run."""
    import gc
    import statistics
    from vocab_lm.train import accumulated_backward, optimizer_update
    from vocab_lm.accounting import tensor_bytes
    root = Path(s['output']); root.mkdir(parents=True,exist_ok=True)
    receipt = dict(identity=dict(source=source_identity(), config_sha256=sha256(json.dumps(s,sort_keys=True).encode())),
                   environment=environment(torch.device(s['device'])), rows=[])
    for arm,a in s['arms'].items():
        for context in (1024,):
            gc.collect(); torch.cuda.empty_cache()
            # Fixed position capacity isolates sequence cost, including positional parameters.
            model,_ = construct(s,arm,torch.zeros(a['vocab_size'],dtype=torch.int64)); model.to(s['device'])
            opt = torch.optim.AdamW(model.parameters(),lr=s['training']['learning_rate'],
                betas=tuple(s['training']['betas']),eps=s['training']['epsilon'],
                weight_decay=s['training']['weight_decay'],foreach=False,fused=False)
            scaler = torch.amp.GradScaler('cuda',enabled=False)
            batch_size = s['training']['microbatch_size']
            batch = dict(inputs=torch.randint(a['vocab_size'],(batch_size,context)),
                         targets=torch.randint(a['vocab_size'],(batch_size,context)),tokens=batch_size*context)
            batches = [batch]*s['training']['accumulation']
            row = dict(arm=arm,context=context,microbatch=batch_size,accumulation=len(batches),
                       tokens_per_update=batch_size*context*len(batches),raw_bytes_per_hour=None)
            samples=[]; allocated_peak=reserved_peak=0
            try:
                for step in range(4):
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
                    opt.zero_grad(set_to_none=True)
                    loss = accumulated_backward(model,batches,device=torch.device(s['device']),scaler=scaler,fp16=False)
                    optimizer_update(model,opt,scaler,s['training']['clip_norm'])
                    torch.cuda.synchronize()
                    if step:
                        samples.append(time.perf_counter()-started)
                        allocated_peak=max(allocated_peak,torch.cuda.max_memory_allocated())
                        reserved_peak=max(reserved_peak,torch.cuda.max_memory_reserved())
                seconds=statistics.median(samples)
                row.update(status='PASS',finite_loss=bool(torch.isfinite(torch.tensor(loss))),
                    update_seconds=seconds,samples_seconds=samples,**gpu_memory(s['device']),
                    parameter_bytes=tensor_bytes(list(model.parameters())),
                    gradient_bytes=tensor_bytes([p.grad for p in model.parameters()]),
                    adam_bytes=tensor_bytes(opt.state_dict()),buffer_bytes=tensor_bytes(list(model.buffers())),
                    flops_per_update=len(batches)*scaling_training_flops(model,batch_size,context),
                    updates_per_hour=3600/seconds)
                row.update(peak_allocated=allocated_peak,peak_reserved=reserved_peak)
                row['flops_per_hour']=row['flops_per_update']*row['updates_per_hour']
                # Separate instrumented update: profiler totals are coverage evidence,
                # not replacement budget arithmetic or timed training latency.
                opt.zero_grad(set_to_none=True)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
                    accumulated_backward(model,batches,device=torch.device(s['device']),scaler=scaler,fp16=False)
                    optimizer_update(model,opt,scaler,s['training']['clip_norm'])
                torch.cuda.synchronize()
                row['profiler_reported_flops']=sum(e.flops for e in prof.key_averages())
                row['profiler_limitation']='Partial aten coverage; SDPA/backward omission; separate from warmed timing and memory'
                probe=root/'runtime_probe.pt'
                started=time.perf_counter()
                save_training_checkpoint(probe,model,dict(optimizer=opt.state_dict()),
                    tokenizer=dict(sha256='a'*64),dataset=dict(sha256='b'*64,split='validation',protocol=s['evaluation']))
                row['checkpoint_save_seconds']=time.perf_counter()-started
                row['checkpoint_bytes']=probe.stat().st_size
                torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
                restored,payload=load_training_checkpoint(probe,s['device']); torch.cuda.synchronize()
                row['checkpoint_load_seconds']=time.perf_counter()-started
                row['checkpoint_peak_allocated']=torch.cuda.max_memory_allocated()
                row['checkpoint_peak_reserved']=torch.cuda.max_memory_reserved()
                del restored,payload
            except torch.cuda.OutOfMemoryError:
                row.update(status='FAIL',reason='CUDA OOM')
            receipt['rows'].append(row)
            write_json(root/'synthetic_profiles.json',receipt)
            print(json.dumps(row),flush=True)
            del model,opt,scaler,batches,batch
    return receipt


def synthetic_resume_worker(s, arm, stage):
    """Large backbone, four-token synthetic documents; two updates maximum."""
    import hashlib
    from vocab_lm.evaluate import Document
    a = s['arms'][arm]
    root = Path(s['output'])/'synthetic_resume'/arm
    root.mkdir(parents=True,exist_ok=True)
    seed_all(s['training']['seed'])
    model,_ = construct(s,arm,torch.zeros(a['vocab_size'],dtype=torch.int64))
    model.to(s['device'])
    docs=[Document((0,1,2,3,4,0),(0,1,1,1,1,0),'c'*64)]
    data=TrainingData(docs,context=4,stride=2,seed=42,tokenizer={'sha256':'a'*64},
                      dataset={'sha256':'b'*64,'split':'train'})
    validation=dict(sha256='d'*64,split='validation',protocol=s['evaluation'])
    cfg=dict(s['training']); cfg.update(context=4,stride=2,microbatch_size=1,accumulation=1,
        budget_limit=10**18,checkpoint_seconds=0,checkpoint_interval=0)
    cfg=TrainConfig(**cfg)
    payload=None
    load_seconds=None
    if stage=='resumed':
        started=time.perf_counter()
        model,payload=load_training_checkpoint(root/'last.pt',s['device'])
        torch.cuda.synchronize()
        load_seconds=time.perf_counter()-started
    # Same boundary-crossing rule as run(); resumed FLOPs determine previous.
    previous=payload['training_state']['counters']['estimated_flops'] if payload else 0
    thresholds={name:[scaling_training_flops(model,1,2)*f for f in factors]
                for name,factors in [('evaluation',(0.5,2.)),('diagnostics',(1.,3.))]}
    events={k:[] for k in thresholds}
    def boundary(state, model):
        nonlocal previous
        current=state['counters']['estimated_flops']
        for name,points in thresholds.items():
            events[name].extend(p for p in points if previous < p <= current)
        previous=current
    start=time.perf_counter()
    state=train(model,data,cfg,validation_identity=validation,
                resume=payload['training_state'] if payload else None,max_attempts=2 if stage=='full' else 1,
                on_boundary=boundary)
    elapsed=time.perf_counter()-start
    save_seconds=None
    if stage=='split':
        started=time.perf_counter()
        save_training_checkpoint(root/'last.pt',model,state,tokenizer=data.identity['tokenizer'],dataset=validation)
        save_seconds=time.perf_counter()-started
    def digest(tree):
        h=hashlib.sha256()
        def visit(x):
            if isinstance(x,torch.Tensor):
                h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(x,dict):
                for k in sorted(x,key=str): h.update(str(k).encode());visit(x[k])
            elif isinstance(x,(tuple,list)):
                for v in x: visit(v)
            else: h.update(repr(x).encode())
        visit(tree);return h.hexdigest()
    counters={k:v for k,v in state['counters'].items() if k not in ('wall_seconds','active_seconds')}
    result=dict(model=digest(model.state_dict()),optimizer=digest(state['optimizer']),
        scheduler=digest(state['scheduler']),rng=digest(state['rng']),data=digest(state['data']),
        counters=counters,seconds=elapsed,checkpoint_save_seconds=save_seconds,checkpoint_load_seconds=load_seconds,
        checkpoint_bytes=(root/'last.pt').stat().st_size if stage=='split' else None)
    result['events']=events
    write_json(root/(stage+'.json'),result)
    return result


def synthetic_resume(s, config_path):
    import subprocess
    root=Path(s['output'])
    results={}
    for arm in s['arms']:
        for stage in ('full','split','resumed'):
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'synthetic-resume',
                '--config',str(config_path),'--arm',arm,'--synthetic-stage',stage],check=True)
        folder=root/'synthetic_resume'/arm
        full=json.loads((folder/'full.json').read_text())
        split=json.loads((folder/'split.json').read_text())
        resumed=json.loads((folder/'resumed.json').read_text())
        keys=('model','optimizer','scheduler','rng','data','counters')
        results[arm]={k:full[k]==resumed[k] for k in keys}
        results[arm]['event_schedules']=all(full['events'][k]==split['events'][k]+resumed['events'][k]
                                                   for k in full['events'])
        write_json(root/'synthetic_resume.json',dict(identity=dict(source=source_identity(),
            config_sha256=sha256(json.dumps(s,sort_keys=True).encode())),results=results,
            passed=len(results)==len(s['arms']) and all(all(v.values()) for v in results.values())))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare", "smoke", "validate", "preflight", "synthetic-profile", "synthetic-resume", "status", "freeze", "profile", "train", "resume", "evaluate", "confirm", "diagnose", "report"])
    parser.add_argument("--config", default="configs/vocab_scaling_large.yaml")
    parser.add_argument("--arm", default="baseline16k")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--synthetic-stage',choices=['full','split','resumed'],help=argparse.SUPPRESS)
    parser.add_argument("--max-attempts", type=int, help="Graceful simulated job boundary; global budget remains frozen")
    args = parser.parse_args(argv)
    s = load_settings(args.config)
    torch.set_num_threads(s["threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.operation == 'prepare':
        if not large_study(s):
            parser.error('prepare requires the declared large generation')
        from vocab_lm.fineweb import prepare_scaling_data
        result=prepare_scaling_data(s['data']['root'],s['arms'][s['baseline']]['tokenizer'],s['protocol']['preparation'])
        frozen=Path('artifacts/vocab_scaling_large/protocol.json')
        if frozen.exists():
            expected=json.loads(frozen.read_text())
            if result['hashes']!=expected['dataset_hashes'] or file_hash(Path(s['data']['root'])/'preparation.json')!=expected['preparation_sha256']:
                raise ValueError('Fresh preparation differs from frozen corpus manifest')
        print(json.dumps(dict(coverage=result['coverage'],hashes=result['hashes']),indent=2))
    elif args.operation == "validate":
        print("Configuration valid; artifact availability is checked by preflight")
    elif args.operation == "preflight":
        result = preflight(s)
        Path(s["output"]).mkdir(parents=True, exist_ok=True)
        write_json(Path(s["output"])/"preflight.json", result)
        print(json.dumps(result, indent=2))
        if large_study(s) and not result['passed']:
            raise SystemExit(1)
    elif args.operation in {'synthetic-profile','profile'}:
        if not large_study(s):
            parser.error('Synthetic profiling requires large study config')
        synthetic_profile(s)
    elif args.operation == 'smoke':
        if not large_study(s): parser.error('smoke requires the large study')
        print(json.dumps(large_smoke(s),indent=2))
    elif args.operation == 'synthetic-resume':
        if not large_study(s):
            parser.error('Synthetic resume requires large study config')
        if args.synthetic_stage:
            synthetic_resume_worker(s,args.arm,args.synthetic_stage)
        else:
            print(json.dumps(synthetic_resume(s,args.config),indent=2))
    elif args.operation == 'status':
        path=Path(s['output'])/args.arm/'run'/'last.pt'
        if not path.exists():
            print(json.dumps(dict(arm=args.arm,status='not started')))
        else:
            _,payload=load_training_checkpoint(path)
            print(json.dumps(dict(arm=args.arm,counters=payload['training_state']['counters'],
                data=payload['training_state']['data']),indent=2))
    elif args.operation == "report":
        report(s)
    elif args.operation == 'freeze':
        print(json.dumps(freeze_large_inputs(s),indent=2))
    else:
        result = run(s, args.arm, resume=args.resume or args.operation == "resume", max_attempts=args.max_attempts,
                     operation="train" if args.operation in {"train", "resume", "profile"} else args.operation)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

