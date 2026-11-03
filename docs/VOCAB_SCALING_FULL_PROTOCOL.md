# Frozen full toy experiment, 2026-11-04

Authorized comparison: 8,192 Dense tied FP32 versus 16,384 Low-rank tied FP32,
rank 126, vocabulary archive ceiling 8,389,916 bytes. Shared backbone 4,865,536
parameters. Configuration: `configs/vocab_scaling_full_toy.yaml`, copied from the
feasibility toy recipe with only budget and output destination changed.

The pretraining machine-readable freeze is
`results/vocab_scaling_full_toy/manifest.json`. It records the implementation Git
revision/content identity, individual source hashes, exact configuration,
dataset/tokenizer identities, ordered raw document hashes/lengths, screening and
secondary manifests, environment, storage preflight, and reporting convention.
Historical feasibility and memory artifacts remain untouched.

## Budget and exposure

Using the unchanged `vocab_scaling_v1` estimator and a CPU data-only walk of the
seed-42 Dense stream, the first complete accumulation group reaching 60,000,000
raw bytes ends at 60,007,192 bytes, 15,883,125 scored tokens and 8,283 updates.
Its exact estimated cost is **1,521,851,151,555,792 FLOPs**. Both arms receive this
same configured ceiling and reject the next entire group if it would exceed it.
No equality of bytes, tokens or updates is imposed. Report unused compute.

The original toy configuration uses the smoke prefix: 574 documents containing
2,000,916 unique bytes. This full run preserves that corpus and repeats its seeded
epoch permutations. Approximately 60 MB is cumulative exposure, NOT unique text.
The Dense budget walk ends at epoch 29, document-position 569, first-target 641.

## Evaluation and diagnostics

Screening is raw validation documents [0,8); secondary validation is [8,16).
Hash-disjointness from training and each other, cross-tokenizer raw identity and
byte-exact tokenizer reconstruction are verified by preflight.

**Pretraining limitation:** secondary documents were already evaluated in the
feasibility study. Other historical validation usage is not sufficiently proven
absent to establish a genuinely untouched set from these artifacts. Therefore
the requested confirmation output is explicitly reused, document-disjoint
secondary validation, NOT independent confirmation. No sealed test is accessed.

Screening occurs at initialization and the first completed group reaching each
25%, 50%, 75% threshold, then at the final reachable boundary. An exact 100%
boundary is evaluated once. No additional job-boundary screening occurs.
Report exact counters without interpolation. After BOTH arms finish, run one
secondary evaluation per final `last.pt`; never select a checkpoint on it.

Existing deterministic 128-row diagnostics occur at 50% and endpoint; exact 100%
is not duplicated. Collect existing norms, variance, sampled spectrum, numerical
and entropy ranks, factor norms/conditioning/utilization, gradient statistics
and frequency metrics. No extra expensive diagnostics based on outcomes.
Post-run diagnose is unnecessary when endpoint diagnostics are already present.
Diagnostics/evaluation do not alter training RNG, optimizer, cursor or exposure.

## Identical training recipe and safety

Six layers, width 256, four heads, FFN 1024, model context 512, dropout 0.1;
training/evaluation context 256 and stride 128. Microbatch 2, accumulation 8.
AdamW lr 0.0003, betas (0.9,0.999), epsilon 1e-8, decay 0.01, clip 1.0,
constant schedule, no warmup, seed 42. FP32 throughout, TF32 disabled, one CPU
thread. No model/optimizer/estimator/evaluation mathematics changes were made.

Atomic last checkpoint every 25 attempted updates or 300 seconds and at exit.
Checkpoint includes global budget/config, model, optimizer/scheduler, RNG,
cursor, exposure/compute counters and source/environment identity. Evaluation
and diagnostic threshold progression is reconstructed from saved FLOPs.
Independent jobs continue with:

```sh
python scripts/vocab_scaling.py train --config configs/vocab_scaling_full_toy.yaml --arm baseline8k --resume
python scripts/vocab_scaling.py train --config configs/vocab_scaling_full_toy.yaml --arm vocab16k --resume
```

Run arms sequentially with no concurrent GPU tests. Preserve failures/checkpoints;
stop on nonfinite training. Minimal engineering fixes require a written deviation
before continuation; mathematics, optimizer, compute or evaluation changes
require user approval. No low-rank memory optimization is authorized or needed.

## Reporting fixed before observation

Use all five screening observations; final secondary BPB is separately identified.
Report BPB versus compute, raw exposure bytes, active time and updates; training
token NLL versus compute; byte and token throughput separately; allocated and
reserved GPU peaks; vocabulary/full-inference archives and persistent tensor
state; sampled representation diagnostics and tokenizer-specific frequency
strata. Throughput curves aggregate contiguous blocks (up to 100); no sample
selection. Host peak RSS remains unmeasured. Analytical FLOPs are not exact
hardware operations; formulas/exclusions remain those in VOCAB_SCALING.md.

Active timing covers synchronized updates, excluding CPU batch construction,
validation/diagnostics and checkpoint I/O. Full wall counters are also retained.
The approximately 68.4 MB Low-rank peak allocation overhead is a measured
larger-softmax trade-off; approximately 67.1 MB was attributed to overlapping
logits/CE tensors. Do not hide it or infer memory from parameter counts.

Machine-readable logs, manifests, atomic checkpoints and integrity evidence stay
in ignored results. Generate summary.csv, curves.png and report.md with
`scripts/vocab_scaling_full_report.py`. State single-seed and tiny/reused-validation
limitations. No causal vocabulary-only or statistical-significance claim. Do not
launch follow-up seeds, tuning, new tokenizers, other arms or the large study.
