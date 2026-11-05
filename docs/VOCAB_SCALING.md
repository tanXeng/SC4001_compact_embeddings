# Final large vocabulary-scaling study

The owner-approved configuration is frozen in `configs/vocab_scaling_large.yaml`
and `artifacts/vocab_scaling_large/protocol.json`. Scientific training has not
begun. Use the final repository commit and require a passing A40 preflight before
launching any arm. Source/configuration/tokenizer/data/environment drift blocks
training; never edit a receipt to bypass that protection.

## Scientific policy

- **5e17 estimated FLOPs per arm**, `vocab_scaling_v1`, complete optimizer updates
  only. Match estimated compute, not bytes, tokens, updates or wall time.
- Exactly `baseline16k`, `dense32k`, `lowrank32k`, `adaptive32k`; no Hash or 64K.
- Common FP32 Transformer: 10 layers, hidden 512, 8 heads, FFN 2048,
  context/position capacity 1024, dropout .1; 32,033,792 backbone parameters.
- **Microbatch 16 x accumulation 2 = 32 sequences / 32,768 executed positions per
  update** at full context 1024. Training stride 512. Short document windows and
  padding determine actual scored targets and per-update FLOPs.
- AdamW LR **3e-4**, betas .9/.999, epsilon 1e-8, weight decay .01, clip norm 1,
  constant schedule, seed 42. FP32 including logits/loss; TF32 disabled.
- Dense tied 16K storage ceiling; Dense tied 32K is deliberately outside it.
  Low-rank rank **252**, exact tied AB. Adaptive cutoffs **[12544,20736]**,
  widths **[512,128,32]**, deterministic three-band factor-four allocation.
  Existing initialization, tying and standalone archive accounting are unchanged.
- Validation `fineweb_document_eos_c1024_s512_v2`: context1024/stride512,
  document-local context, each intended text/EOS target once, raw UTF-8 byte BPB.
  Screen documents [0,256); document-disjoint secondary [256,512). Initialization
  and 25/50/75/100% budget screening; diagnostics at 50% and endpoint, outside
  active update timing. Secondary once after all arms finish. No sealed-test reader.

FLOPs include GEMM forward/backward, full-square attention products/softmax,
executed padded/overlapping positions, Low-rank input projection and Adaptive
maximum-band input bound. Exclusions remain loss softmax, optimizer,
LayerNorm/GELU, gathers/allocation, diagnostics and I/O. Profiler partial totals do
not replace budget arithmetic.

## Data and identities

FineWeb `HuggingFaceFW/fineweb`, sample-10BT, revision
`9bb295ddab0e05d785b879661af7260fed5140fc`; pinned 014 then 013 parquet files.
The expanded generation is `data/vocab_scaling_large_5e17/`. Its rounded target
was derived from an exact complete-epoch capacity walk of the previous corpus;
final ordered-token walks must support **6e17 FLOPs for every arm**, providing at
least 20% compute headroom above 5e17. No extrapolated exposure authorizes a freeze.

The old generation remains intact. Frozen ordered-record digests verify its
813,238-document training prefix and complete validation selection, independently
of physical raw-file offsets. Fresh clones reproduce the same selection directly
from pinned sources and verify those digests. Seed42 domain-separated order/split
hashes, modulus16, global exact-text dedup and exclusion of all validation-bucket
texts from training are unchanged. Both tokenizers see identical ordered raw
training and validation documents. Preparation checks every reconstruction, token
range, byte attribution and EOS boundary.

Tokenizers are unchanged: nested byte-level BPE, maximum32K and its16K prefix,
trained only on the original 65,182-document / 200,000,346-byte construction prefix.
No retraining, normalization or synthetic prefix space; EOS0, literal EOS spelling
remains ordinary text. Frozen tokenizer files and their hashes remain in Git.

The input protocol records exact documents/raw bytes/token counts, tokenizer and
preparation hashes, all data-file hashes, representation allocations, schedules,
source/config identities and exact 5e17/6e17 walk counters. Stream identities hash
sorted mappings of shard-index and token-shard hashes; the index binds ordered
boundaries. Exact walk bytes sum actual token byte lengths for scored positions,
including zero-byte EOS; they are not proportional document-length estimates.

## Frozen data capacity and allocations

Prepared training: **1,137,468 documents / 3,500,002,716 unique raw bytes**. Validation remains **3,219 documents / 10,001,247 raw bytes**. Training target counts including EOS: 845,875,687 (16K), 792,710,829 (32K). Both final token streams passed complete reconstruction and disjointness checks. The measured maximum unique exposure at 5e17 is **2,657,873,921 bytes**; at the 6e17 headroom ceiling it is **3,188,972,045 bytes**. These actual-stream results justify rounding upward to a 3.5 GB preparation target; 6 GB was unnecessary.

| Arm | Complete updates at 5e17 | Exact unique bytes at 5e17 | Exact unique bytes at 6e17 | 20% headroom |
|---|---:|---:|---:|---|
| adaptive32k | 51,081 | 2,657,873,921 | 3,188,972,045 | PASS |
| baseline16k | 50,962 | 2,538,583,404 | 3,045,307,010 | PASS |
| dense32k | 43,848 | 2,282,591,868 | 2,737,684,115 | PASS |
| lowrank32k | 51,027 | 2,655,115,827 | 3,185,626,202 | PASS |

| Arm | Vocabulary parameters | Standalone archive bytes | Ceiling use |
|---|---:|---:|---:|
| adaptive32k | 8,200,192 | 33,327,936 | 99.3211% |
| baseline16k | 8,388,608 | 33,555,740 | 100.0000% |
| dense32k | 16,777,216 | 67,110,172 | 199.9961% |
| lowrank32k | 8,386,560 | 33,547,800 | 99.9763% |

The ceiling is 33,555,740 bytes. Low-rank rank253 does not fit; rank252 does. Adaptive uses the largest fitting 256-type head grid point and exceeds 95% utilization. Dense32K remains the explicitly out-of-storage-budget control.

## Environment and mandatory preflight

The server has two NVIDIA A40 GPUs (driver reports 46,068 MiB each). Use one
independent arm per GPU. No DDP or cross-arm gradient synchronization. All mutable
model/optimizer/RNG/cursor/counter/log/checkpoint/schedule state lives under
`results/vocab_scaling_large/{arm}/run/`. Prepared shard maps are read-only;
preflight verifies independent concurrent readers for both tokenizers.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[dev]"
python scripts/vocab_scaling.py prepare --config configs/vocab_scaling_large.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/vocab_scaling.py preflight --config configs/vocab_scaling_large.yaml
python -m pytest -q
git diff --check
```

On Windows activate `.venv/Scripts/Activate.ps1`. Dependencies remain pinned in
`pyproject.toml`. Preparation reuses matching pinned caches and complete stages;
mismatched/corrupt/incomplete stages fail while preserving evidence. Do not delete
or overwrite a generation to recover silently.

Final preflight checks the selected batch only: two warmup updates initializing
AdamW, at least three synchronized complete-update timings, finite loss/gradients,
checkpoint save/load, exact process-boundary resume, bounded two-update real-data
integration, concurrent readers and exact data headroom. Allocated **and reserved**
CUDA peaks, including checkpoint load, must stay **<=24 GiB**. Every mandatory
check must PASS; a passing preflight creates the immutable execution protocol.
There is no need to rerun the broad batch sweep.

## Training commands: execute only after final preflight passes

First pair, in separate processes:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vocab_scaling.py train --config configs/vocab_scaling_large.yaml --arm baseline16k --resume
CUDA_VISIBLE_DEVICES=1 python scripts/vocab_scaling.py train --config configs/vocab_scaling_large.yaml --arm dense32k --resume
```

When the corresponding A40 becomes free:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vocab_scaling.py train --config configs/vocab_scaling_large.yaml --arm lowrank32k --resume
CUDA_VISIBLE_DEVICES=1 python scripts/vocab_scaling.py train --config configs/vocab_scaling_large.yaml --arm adaptive32k --resume
```

Each command starts fresh or resumes exactly from its own durable checkpoint.
Execution continues to the last complete update fitting within the global 5e17
ceiling. No artificial wall-clock stop. Atomic checkpoints every 300 seconds;
SIGTERM/SIGINT request a complete-update boundary and checkpoint. Resume restores
optimizer, scheduler, RNG, cursor, counters and evaluation/diagnostic schedules;
the budget never resets. Normal `result.json` completion is `budget_boundary`.

```bash
python scripts/vocab_scaling.py status --config configs/vocab_scaling_large.yaml --arm baseline16k
python scripts/vocab_scaling.py evaluate --config configs/vocab_scaling_large.yaml --arm baseline16k
python scripts/vocab_scaling.py confirm --config configs/vocab_scaling_large.yaml --arm baseline16k
python scripts/vocab_scaling.py report --config configs/vocab_scaling_large.yaml
```

Repeat status/evaluate/confirm per arm; confirmation runs after all four finish.
Preserve checkpoints and receipts on failure. Stop on source/data/environment
mismatch, OOM, nonfinite values, stream exhaustion or missing scheduled outputs.

## Handoff scope

Git contains necessary source/tests, the final YAML, frozen tokenizer/input-protocol
artifacts and concise operator documentation. FineWeb sources, prepared shards,
checkpoints, logs, profiles and generated receipts/results remain ignored under
`data/`, `results/`, `.tmp/` and `.local_history/`. Previous protocols and preparation
evidence were archived locally before regeneration. Do not push without owner
approval. This preparation pass does not launch scientific training.
