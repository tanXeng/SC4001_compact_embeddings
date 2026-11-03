# FP32 vocabulary scaling: continuation record

Dedicated branch: `research/vocab-scaling-fp32-isoflops`. Initial main was clean at
`0c6b54e`. Its regression collection failed because `scripts/experiment_001.py`
was absent. Restored that file from historical commit `ebde308`; baseline then
passed 93 tests. Existing Experiment 001 and overnight artifacts are untouched.

## Scientific scope

Compare the complete practical strategy (vocabulary size plus representation),
not the isolated causal effect of vocabulary size. Both primary arms train from
initialization with the same 4,865,536-parameter backbone, FP32 (including logits
and loss), TF32 disabled, AdamW recipe, shuffled raw-document order and seed 42.
No quality inference is justified by this short feasibility study.

## Frozen compute convention: vocab_scaling_v1

Primary papers reviewed before implementation:

- Tao et al., [Scaling Laws with Vocabulary](https://arxiv.org/html/2407.13623v3),
  section 3 equation 5 uses approximately `6(N_nv + V*d)*D`; tokenization changes D.
- Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/html/2203.15556v1#A6),
  Appendix F counts embedding, QKV, attention products, attention softmax,
  FFN and logits; multiply-add costs two and backward is twice forward.

We adopt the latter's explicit operation approach, adapted to the actual tied
graph. For B rows, T executed positions, n=B*T, d width, f FFN, L layers, H heads:

```
backbone matrix training = 6*n*L*(4*d*d + 2*d*f)
attention products      = 12*L*B*T*T*d
attention softmax       = 9*L*B*H*T*T
dense vocabulary        = 6*n*V*d
low-rank vocabulary     = 6*n*r*(V + 2*d)
```

Dense input is an indexed gather, not a one-hot dense multiplication; no fictitious
input GEMM is charged. Low-rank has three differentiable GEMMs (input A[i]B and
two output products), each charged forward plus both gradients. Tying counts
storage once but does not remove either path's work. Full square attention is
charged, including masked cells, regardless of fused causal implementation.
Padding and overlapping context count whenever executed. Short rows count their
actual padded batch shape. Bias, norms, GELU, dropout, indexing/scatter, loss/output
softmax, optimizer and clipping are excluded. Attention softmax is included using
Appendix F's forward-times-three approximation. Output CE is outside both papers'
explicit matrix convention; this omission matters more for very large V.

Hash support in the estimator follows the *current* reconstructed-table graph:
`6*n*V*d + (n+V)*d*(5*k-1)` (sum/reduction approximation). Adaptive uses the existing
full-band output estimate plus conservative maximum-band input bound. Those are
infrastructure support, not primary arms or validated scaling comparisons here.

This is an analytical convention, not exact hardware operations. PyTorch profiler
records only supported operators; SDPA and backward coverage can be incomplete.
`profiler.json` reports the observed count separately. Profile timing uses a second
session outside profiler, after warmed full accumulation/clipping/AdamW updates.

Complete groups are reserved before any forward/backward. Rejected groups restore
the cursor. Failed/skipped groups consume exposure and compute but never successful
updates. Counters use Python integers and checkpoint without rounding. No final
partial update or altered accumulation is allowed.

## Frozen storage convention: standalone_torch_save_v1

Use the existing `vocabulary_storage_report`: actual torch.save BytesIO archive of
`{specification,state_dict}`, CPU contiguous clones, all trainable tensors,
persistent buffers, metadata and ZIP/pickle overhead. Tied tensors count once.
The 8K dense archive is the ceiling (8,389,916 bytes in this environment).
Binary search actual archives for the largest rank <= min(V,d) before quality.
Rank 126 fits; rank 127 does not. Do not pad the archive. PyTorch-version archive
overhead can differ: manifests pin the environment and measurement convention.

Parameter, gradient and initialized AdamW tensor memory are recorded separately.
Model inference archive includes backbone and required identities. CUDA allocated,
reserved and peaks include activations/temporaries; a peak is not a decomposition.
Host peak RSS and isolated transient memory are currently null rather than guessed.

## Data and validation

Use only existing FineWeb 8K/16K artifacts. Preflight checks shard integrity and
tokenizer binding through the shared loader, then reconstructs document bytes and
verifies SHA256. It compares ordered raw hashes/bytes across arms. Seeded document
permutations are identical; documents have isolated causal windows. EOS is scored
with zero bytes. Screening is the first 8 validation documents; confirmation the
next 8, disjoint from training and each other. No sealed-test path exists.

Shared evaluation scores each target once with overlapping context and FP64 NLL
accumulation. BPB denominator is actual target UTF-8 bytes. Frequency strata are
three token-type groups ranked by training-only occurrence count, ties by ID;
strata boundaries differ by tokenizer and are explanatory, not matched token sets.

## Feasibility scope and commands

Data-only walk to 2,000,000 baseline bytes ended at 2,006,956 bytes, 277 groups,
50,892,756,872,976 FLOPs. This integer is frozen in toy YAML for both arms. Future
extended training must use a new output directory/configuration and authorization.

Run from repository root with the installed Python environment (on Windows use
`.venv/Scripts/python.exe`; on Linux use the environment's `python`):

```sh
python -B -m pytest -q --basetemp=.tmp/scaling-tests
python scripts/vocab_scaling.py validate --config configs/vocab_scaling_toy.yaml
python scripts/vocab_scaling.py preflight --config configs/vocab_scaling_toy.yaml
python scripts/vocab_scaling.py profile --arm baseline8k
python scripts/vocab_scaling.py profile --arm vocab16k
python scripts/vocab_scaling.py train --arm baseline8k
python scripts/vocab_scaling.py train --arm vocab16k
python scripts/vocab_scaling.py train --arm vocab16k --resume
python scripts/vocab_scaling.py evaluate --arm vocab16k
python scripts/vocab_scaling.py diagnose --arm vocab16k
python scripts/vocab_scaling.py report
```

`--max-attempts N` simulates a job boundary, retaining the original global budget.
SIGINT/SIGTERM request a stop after the current complete update; last.pt is saved
atomically with existing bounded Windows replacement retry. Periodic checkpoints
occur every 25 attempts or 300 seconds at completed boundaries. A hard kill during
an update loses work since the last checkpoint; resume recomputes that work from
the checkpoint trajectory. Source/environment/config/tokenizer changes fail closed.
Single writer per arm/output directory is required.

Validation and diagnostic fractions are frozen in configuration. Crossing a fraction
runs at the first complete update at/after that point; report actual FLOPs, never
interpolate. Resume uses saved cumulative FLOPs to preserve threshold progression.
The exact budget endpoint usually falls between groups: final evaluation is marked
separately and unused FLOPs are reported. Best screening is saved independently;
confirmation currently evaluates the final checkpoint, not the selected best.

## Outputs and future large study

Ignored `results/vocab_scaling_toy/` holds preflight, per-arm manifests, profiles,
training/validation/diagnostic JSONL, checkpoints, result.json, summary.csv,
report.md and curves.png. Logs record observed timestamps/exposure/compute and
storage. Sparse deterministic sampled spectra are bounded to 128 effective rows;
projection conditioning and factor norms supplement low-rank rank utilization.
Callbacks in `DIAGNOSTIC_EXTENSIONS` allow future representation diagnostics.
Hash includes occupancy/collisions/sharing/importance; optional bounded actual
LM-loss gradient interference takes a supplied probe batch and includes both tied
paths. Adaptive records per-band factor ranks/norms/conditioning/allocation.
Hash/Adaptive are not enabled as primary config arms and have not been profiled
in this task. Diagnostics run outside complete-update timings.

The large YAML is a template: 10 layers, d=512, f=2048, context=512 gives a
31,771,648-parameter backbone. It intentionally has null tokenizer locations and
a nonexistent corpus root; preflight refuses it. After separate authorization,
prepare compatible 16K/32K artifacts (64K optional), configure those paths, freeze
a production FLOP budget/output directory and screening/confirmation manifests,
then use the same CLI with `--arm baseline16k` and `--arm vocab32k`. Tokenizer-family
loading currently expects the shared NestedBPE base/expanded artifact contract;
optional 64K needs its own compatible family directory with common raw documents.
No new dependencies installed. Plotting requires the already available matplotlib.

## Limitations to address before production

Same-environment CPU and this CUDA setup pass bitwise continuation tests; CUDA
across hardware/software or nondeterministic kernels has no bitwise guarantee.
Time/peak allocator measurements naturally vary. Profiling uses real document
windows, so tokenizer-dependent shapes differ and must be interpreted as actual
batch characteristics. Host RAM and isolated reconstruction temporary bytes are
not instrumented. JSONL and checkpoint publication are separate: a hard kill can
leave log rows newer than the last checkpoint; recovery must trim these before
resuming (the runner performs this reconciliation). No concurrent writers.
