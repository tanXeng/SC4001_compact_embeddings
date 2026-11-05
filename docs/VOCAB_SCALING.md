# Frozen large vocabulary-scaling study

Run one exact owner-approved commit from the repository root. Source content, configuration, project dependencies, tokenizers and data hashes are enforced by [the frozen input protocol](../artifacts/vocab_scaling_large/protocol.json). This package prepares and verifies the study; no substantive large-model training or push was performed during preparation. The actual 32 GB server preflight is still mandatory.

## Scientific protocol

Question: at fixed estimated compute, does 16K to 32K improve raw-byte BPB, and can established compressed representations preserve that benefit near the 16K Dense vocabulary-storage ceiling?

- **2.5e17 estimated FLOPs per arm**, unchanged `vocab_scaling_v1`, complete updates only; no wraparound or budget reset across jobs.
- Common FP32 Transformer: **10 layers, hidden 512, 8 heads, FFN 2048, context/position capacity 1024**, dropout .1; **32,033,792 backbone parameters** excluding vocabulary.
- Microbatch **2**, accumulation **4**, training stride **512**: 8,192 executed positions per full group; actual scored targets vary with document length/padding.
- AdamW LR .0003, betas .9/.999, epsilon 1e-8, weight decay .01, clip norm 1, constant LR, seed **42**, FP32 including logits/loss, TF32 disabled.
- Validation **fineweb_document_eos_c1024_s512_v2**: context1024/stride512, every intended text/closing-EOS target once, document-local context, exact raw UTF-8 byte BPB denominator. Same ordered raw validation documents for both tokenizers. Screen [0,256); secondary [256,512).
- Screening at initialization and 25/50/75/100% budget, with final fitting boundary when necessary; secondary once after all four arms finish. Diagnostics at 50% and endpoint, outside active update timing.
- Diagnostics: Dense norms/variance/sampled spectrum/frequency quality; Low-rank effective spectrum/rank, factors/gradients/conditioning and frequency quality; Adaptive native-band quality/mass, ranks, table/projection norms/gradients/conditioning and allocation. No Hash arm or Hash-specific diagnostics.

The small FP32 study motivated larger vocabularies and practical Low-rank/Adaptive baselines. Its corrected Adaptive allocation improved substantially over an accidentally underallocated configuration; the large pipeline therefore automatically enforces capacity utilization. Hash performed poorly and is excluded. Toy results used a different evaluation protocol, a small backbone, one seed and reused tiny validation subsets; no retroactive cross-protocol comparison or convergence claim is made. This budget is a comparative research choice around ten scored tokens/total parameter for the storage-matched arms, not a mandatory compute-optimal pretraining target.

| Arm | Vocabulary | Structure | Vocabulary parameters | Archive bytes | Ceiling use | Total parameters |
|---|---:|---|---:|---:|---:|---:|
| adaptive32k | 32768 | Cutoffs12544/20736; widths512/128/32 | 8,200,192 | 33,327,936 | 99.3211% | 40,233,984 |
| baseline16k | 16384 | Dense tied ceiling | 8,388,608 | 33,555,740 | 100.0000% | 40,422,400 |
| dense32k | 32768 | Dense tied; explicitly out of storage budget | 16,777,216 | 67,110,172 | 199.9961% | 48,811,008 |
| lowrank32k | 32768 | Rank252 exact tied AB | 8,386,560 | 33,547,800 | 99.9763% | 40,420,352 |

The 16K Dense standalone FP32 serialized archive is the exact ceiling. Low-rank rank252 fits; rank253 does not. Adaptive keeps three frequency bands, factor four, fixed middle V/4 and the largest fitting head on a 256-type grid. Nearby allocations, per-band parameters/buffers and actual archives are frozen in the protocol. Both compressed methods exceed 95%; no padding parameters or validation-selected capacity. Complete-training counts determine only the frequency ordering (EOS first, count descending, ID tie-break); count/order digests and band mass are frozen.

FLOPs include GEMM forward/backward, full-square attention products/softmax, executed padded/overlapping positions, Low-rank input projection and Adaptive maximum-band input bound. Excludes loss softmax, optimizer, LayerNorm/GELU, gathers/allocation, diagnostics and I/O. Partial profiler totals are coverage evidence, never substituted into the budget estimator.

## Frozen data and tokenizer provenance

FineWeb source: HuggingFaceFW/fineweb revision **9bb295ddab0e05d785b879661af7260fed5140fc**, sample-10BT, pinned 014 then 013 parquet files (2.723 GB download on a fresh clone). Seed42 domain-separated order/split hashes, modulus16 and global exact-text dedup; all reserved validation-bucket documents are excluded from training. The extended ordered raw train prefix and original validation selection are preserved. No sealed-test reader is provided.

The fixed tokenizers use the established byte-level nested BPE method: train maximum32K and derive its16K prefix from the exact 65,182-document, 200,000,346-byte training-only prefix; tokenizers0.22.1, min frequency2, complete byte alphabet, regex ByteLevel, no synthetic prefix space/normalization, EOS ID0. Literal EOS spelling stays ordinary text. Two independent fits matched IDs/merges. The 16K artifact matches the earlier 16K tokenizer. Data expansion did not retrain or modify either tokenizer.

| Vocabulary | Train documents | Unique train bytes | Scored corpus targets incl EOS | Validation documents / bytes |
|---|---:|---:|---:|---:|
| 16384 | 813,238 | 2,500,000,732 | 604,206,923 | 3,219 / 10,001,247 |
| 32768 | 813,238 | 2,500,000,732 | 566,201,245 | 3,219 / 10,001,247 |

`prepare` verifies every document reconstruction, EOS boundaries, complete ordered coverage and train/validation disjointness. Tokenizer construction-prefix provenance is verified against the expanded train selection. The final seeded-permutation shape walks verify every arm through **3e17 FLOPs**, providing at least **20% headroom** above the training budget. No quality metric selected any input. Full hashes and allocation details are in the small input protocol; generated receipts remain ignored.

| Tokenizer artifact | SHA-256 |
|---|---|
| base.json | `cf148983e4f5838d445c0eb0ccc6793997d03da9509910298b9094b29b1eb38c` |
| expanded.json | `1192955ffd00b6723043a20d599defbe863ed1a48ea1612d9f47c1dbc1e2323b` |
| nested.json | `c8a9dbbd0d78cd8ce5b6a1120acf3b4e170a907c2853238e335f6198f67bde5c` |
| protocol.json | `455ca357ef2808f061f7b602b57478edc863aa26cb687afc575b1aaad3b5feea` |

Total Git tokenizer/input-protocol artifacts: **3,523,556 bytes**. Manifest JSON explicitly preserves the established CRLF bytes on all OSes; Git attributes disable rewriting these artifacts. Python source identity normalizes CRLF/LF for Windows/Linux checkout compatibility.

## Environment and operator commands

Use the exact owner-approved commit, a CUDA-compatible nominal32GB GPU, at least16GB host RAM (32GB recommended) and preferably30GB free disk for source/data, temporary archives and checkpoints. Prepare data before reserving the training GPU. GPU model is currently unknown. If the pinned CUDA/PyTorch build cannot run it, stop for owner review; never change precision/batch/representations yourself.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[dev]"
python scripts/vocab_scaling.py prepare --config configs/vocab_scaling_large.yaml
python scripts/vocab_scaling.py preflight --config configs/vocab_scaling_large.yaml
```

On Windows use `.venv/Scripts/Activate.ps1`. Project dependencies pin the tested NumPy1.24.4, tokenizers0.22.1, PyYAML6.0.2, pyarrow18.1.0, matplotlib3.7.5 and pytest8.3.5. The system Python on the preparation machine has an older incompatible torch; use the project environment.

A fresh clone needs no external prepared corpus. `prepare` obtains pinned source files, reuses valid cache files, and generates identical local16K/32K train/validation shards under `data/vocab_scaling_large_final/`. Matching complete generations are verified without download/re-encoding. Corrupt/mismatched existing files fail without replacement. Completed stages are reusable; an interrupted incomplete shard stage requires owner recovery, preserving evidence.

On the actual server, `preflight` automatically runs short all-arm configured optimizer-update profiles, checkpoint I/O, process-boundary resume and tiny real-data integration checks when receipts are absent/stale. It records exact CUDA/PyTorch/GPU/environment identity and enforces device capacity >=30GiB and **allocated/reserved peaks <=24GiB**, including checkpoint load. Local6GB profiles cannot satisfy this server gate. Every mandatory scientific/resource failure remains FAIL and exits1. A passing server preflight creates the immutable local execution protocol.

After that preflight passes, run the assigned arm; repeat the identical command after an interruption:

```bash
python scripts/vocab_scaling.py train --config configs/vocab_scaling_large.yaml --arm baseline16k --resume
python scripts/vocab_scaling.py train --config configs/vocab_scaling_large.yaml --arm dense32k --resume
python scripts/vocab_scaling.py train --config configs/vocab_scaling_large.yaml --arm lowrank32k --resume
python scripts/vocab_scaling.py train --config configs/vocab_scaling_large.yaml --arm adaptive32k --resume
```

One command starts a fresh arm or resumes its last durable checkpoint. No scientific configuration is selected by the operator. Atomic checkpoints every 300 seconds. There is no wall-clock training limit: a normal run continues to the last complete update fitting the frozen 2.5e17 FLOP budget. SIGTERM/SIGINT request a stop at a complete-update boundary and save an atomic checkpoint where possible. Model, optimizer, scheduler, RNG, cursor, FLOPs/bytes/tokens/update counters and derived schedules restore; global budget never resets. The normal command runs until `result.json` reports `completion: budget_boundary`.

```bash
python scripts/vocab_scaling.py status --config configs/vocab_scaling_large.yaml --arm baseline16k
python scripts/vocab_scaling.py evaluate --config configs/vocab_scaling_large.yaml --arm baseline16k
python scripts/vocab_scaling.py confirm --config configs/vocab_scaling_large.yaml --arm baseline16k
python scripts/vocab_scaling.py report --config configs/vocab_scaling_large.yaml
python -m pytest -q
```

Repeat status/evaluate/confirm for other arms; confirm once after all four finish. Outputs: `results/vocab_scaling_large/{arm}/run/{manifest.json,last.pt,best.pt,training.jsonl,validation.jsonl,diagnostics.jsonl,result.json}`, plus root reports/plots/CSV and preflight receipts. All remain ignored. On termination preserve files and repeat the same train command. Stop/contact owner on missing/incompatible checkpoint, hash/source/environment change, failed preflight, OOM, nonfinite loss, stream exhaustion or absent scheduled outputs. Never delete counters or reset budget.

## Final local verification

Retained suite: **105 tests passed**; the suite is self-contained and does not require ignored research files. Mathematical Dense/Low-rank/Adaptive tying, shared checkpoint formats, storage/maximality/allocation, FLOPs/refusal, tokenizer/data coverage/disjointness, protocol/source enforcement, resume, graceful signal interruption and reporting remain covered. Historical Experiment001 diagnostic tests and old CLI/provenance-format tests were retired together with their dead implementations; shared production math/optimizer/data checks remain. No substantive large training occurred.

All-arm warmed local profiles, exact process-boundary trajectories and two-update real-data integration smokes pass. Smoke quality is not used for configuration selection. Local timing is short RTX2060 evidence; target GPU throughput cannot be inferred from32GB VRAM. Exact final data walks and runtime projections are in ignored `results/vocab_scaling_large/budget_projections.json`.

| Preflight item | Local status |
|---|---|
| backbone_equality | PASS |
| checkpoint_resume | PASS |
| configuration | PASS |
| context_frozen | PASS |
| dataset_identities | PASS |
| diagnostic_availability | PASS |
| expected_parameters | PASS |
| flop_estimator | PASS |
| immutable_protocol | PASS |
| no_sealed_test_access | PASS |
| no_unresolved_failure | FAIL |
| precision | PASS |
| real_data_smoke | PASS |
| representation_selection | PASS |
| runtime_profile_availability | PASS |
| server_profile | FAIL |
| interruption_readiness | PASS |
| source_identity | PASS |
| storage_ceiling | PASS |
| storage_utilization | PASS |
| tokenizer_identities | PASS |
| train_validation_disjointness | PASS |
| training_budget_frozen | PASS |
| unique_data_availability | PASS |
| validation_protocol | PASS |

Only the actual server resource/profile check and its aggregate summary may remain FAIL locally. The input protocol/config/source/project/tokenizer/data identities are frozen. Final commit SHA is assigned when this reviewed candidate is committed; source-content identity is independently enforced.

## Git handoff classification

**A: required Git tree** (includes unchanged tracked modules and new small frozen artifacts):

```text
.gitattributes
.gitignore
artifacts/vocab_scaling_large/protocol.json
artifacts/vocab_scaling_large/tokenizers/base.json
artifacts/vocab_scaling_large/tokenizers/expanded.json
artifacts/vocab_scaling_large/tokenizers/nested.json
artifacts/vocab_scaling_large/tokenizers/protocol.json
configs/testing.yaml
configs/vocab_scaling_large.yaml
docs/VOCAB_SCALING.md
pyproject.toml
README.md
scripts/__init__.py
scripts/vocab_scaling.py
tests/conftest.py
tests/fineweb/test_fineweb.py
tests/training/test_compressed.py
tests/training/test_standalone.py
tests/training/test_training.py
tests/training/test_vocab_scaling.py
vocab_lm/__init__.py
vocab_lm/accounting.py
vocab_lm/checkpoint.py
vocab_lm/config.py
vocab_lm/data.py
vocab_lm/evaluate.py
vocab_lm/fineweb.py
vocab_lm/model.py
vocab_lm/runtime.py
vocab_lm/tokenizer.py
vocab_lm/train.py
```

**B: local-only, ignored:** `.local_history/`, `data/`, `results/`, `checkpoints/`, `.tmp/`, `.venv/` and Python caches. Historical modified research reports/configs/scripts were copied into `.local_history/pre_final_large/` before removal from the active tree. Previous large preparation receipts/checkpoints remain in `results/vocab_scaling_large_previous/`. The previous complete source snapshot reproduces its archived input-protocol source hash. No local historical evidence was deleted or Git history rewritten.

**C: formerly tracked paths removed from the candidate Git tree after consolidation** (these are explicit tracked deletions, not merely unstaged files):

```text
configs/cfg.yaml
configs/fineweb.yaml
configs/vocab_scaling_full_toy.yaml
configs/vocab_scaling_toy.yaml
docs/FINEWEB_DATA_PROTOCOL.md
docs/VOCAB_SCALING_FULL_PROTOCOL.md
docs/VOCAB_SCALING_RESULTS.md
scripts/experiment_001.py
scripts/prepare_fineweb.py
scripts/profile_vocab_memory.py
scripts/train_dense.py
scripts/vocab_scaling_full_report.py
tests/training/test_hash_gate_e.py
```

`load_data` moved into existing `vocab_lm.data`; callers/tests now import it there. Experiment001-only diagnostics were not promoted into production; independent matrix/gradient/storage/restore tests were retained without that import. Old preparation/verifier code and competing CLIs were retired; large reporting stays in the one runner. README is only a pointer. There are no dedicated historical research files required in the current Git tree. Existing shared model/checkpoint/math modules retain generic representation-format support to avoid unrelated churn and weakened shared correctness coverage.

Commit/staging must contain only the A tree plus C deletions. Corpus downloads/shards, checkpoints, local receipts and generated reports must never be added. Frozen tokenizer/protocol bytes are deliberately small regular Git files; no LFS or external tokenizer download is needed.

**READY TO COMMIT/PUSH — SERVER PREFLIGHT REQUIRED BEFORE TRAINING**
