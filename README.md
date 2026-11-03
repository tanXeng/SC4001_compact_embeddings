# Vocabulary and cost in small language models

## FP32 vocabulary scaling research

Use `scripts/vocab_scaling.py` for the storage-matched 8K Dense versus 16K
Low-rank IsoFLOPs feasibility experiment. Shared model/training/evaluation and
atomic checkpoints remain in `vocab_lm/`. Toy and future 16K/32K configurations
are `configs/vocab_scaling_toy.yaml` and `configs/vocab_scaling_large.yaml`.
The large file deliberately requires tokenizer/data preparation and is not runnable.

See [the continuation record](docs/VOCAB_SCALING.md) for literature, frozen FLOP
and archive conventions, verification, run/resume commands and limitations.
Generated evidence lives under ignored `results/vocab_scaling_toy/`.
The checked-in [feasibility results](docs/VOCAB_SCALING_RESULTS.md) summarize
the completed short runs and their measurement limitations.
Do not launch extended toy or large training without separate authorization.

## Experiment 001 Gate A

`model.py` also provides `LowRankVocabulary`, `HashVocabulary` (learned
importance weights and deterministic shared components), and `AdaptiveVocabulary`
(training-frequency bands). Their lookup and full-softmax heads share the same
effective vectors. Checkpoint loading and storage accounting support these kinds;
compressed FLOP estimates include representation operations and report omissions.
The existing dense initializer, model forward, shared optimizer loop, evaluator,
and dense CLI retain their behavior.

`configs/experiment001.yaml` is a separate proposed 8K pilot recipe with a
4,865,536-parameter backbone; it does not replace `configs/cfg.yaml`.
The experiment entry point below runs synthetic forward/backward checks only,
at 8K and 16K, and cannot launch corpus training. Diagnostics are optional.

```powershell
& .venv/Scripts/python.exe -B -m pytest -q --basetemp=.tmp/experiment001-tests
& .venv/Scripts/python.exe -B -m scripts.experiment_001 --diagnostics
# Optional synthetic CUDA measurements:
& .venv/Scripts/python.exe -B -m scripts.experiment_001 --device cuda --diagnostics --output results/experiment_001/cuda.json
```

Measured storage, synthetic diagnostics, literature adaptations and limitations
are recorded under ignored `results/experiment_001/`. FineWeb comparisons require
separate authorization; Gate A results do not establish model quality.

This research project investigates vocabulary size, training cost and embedding
compression in small autoregressive language models. The initial standalone
workflow trains **dense decoder-only models with tied embeddings** from
initialization. It preserves the verified Transformer, AdamW update arithmetic,
document-local scoring and exact-resume behavior. Toy checks establish correctness,
not quality or scalability; no vocabulary benchmark or corpus pretraining was run
while preparing this repository.

```text
vocab_lm/                  model, data, training, evaluation and accounting
scripts/
  train_dense.py           training, resume and validation CLI
  prepare_fineweb.py       explicit preparation and read-only verification stages
configs/
  cfg.yaml                 explicit dense training recipe
  fineweb.yaml             pinned FineWeb preparation protocol
  testing.yaml             small synthetic correctness fixtures
tests/
  training/                mathematics, budgets, CLI and exact resume
  fineweb/                 selection, UTF-8, tokenizers, shards and BPB coverage
docs/                      protocol
pyproject.toml             editable-install metadata and dependencies
```

## File-by-file guide

| Module in `vocab_lm/` | Responsibility |
| --- | --- |
| `__init__.py` | Package marker. |
| `config.py` | Validated model, training, data, checkpoint and evaluation configuration. |
| `model.py` | Transformer and dense/PQ vocabulary representations; tied lookup and projection. |
| `train.py` | AdamW training loop, accumulation, budget policies, exposure counters and compute estimates. |
| `data.py` | Training windows, exact cursor, shard serialization, mmap loading and prefix views. |
| `fineweb.py` | Pinned source acquisition, deterministic selection, raw-text/tokenizer/shard preparation, statistics and verification. |
| `tokenizer.py` | Reversible byte-level BPE, nested vocabulary family and tokenizer identities. |
| `evaluate.py` | Document-local validation windows and byte-weighted BPB. |
| `checkpoint.py` | Model and complete training checkpoints with exact resume. |
| `runtime.py` | RNG state, learning-rate schedule, execution provenance and shared hash/JSON utilities. |
| `accounting.py` | Parameter counts, serialized storage, optimizer state and runtime memory accounting. |

Suggested reading order: `config.py` ? `model.py` ? `tokenizer.py` ?
`data.py` ? `fineweb.py` ? `evaluate.py` ? `train.py` ? `checkpoint.py` ?
`runtime.py` ? `accounting.py`.

`scripts/` contains orchestration only; updates remain in `train.py`. `tests/`
uses synthetic data and never opens a sealed test split. `configs/` holds explicit
recipes and fixtures; `docs/` records the corpus contract and limitations.

## Execution flow

Configuration → identified tokenizer and prepared shards → dense model → shared
training loop → validation BPB → complete checkpoint. Resume loads model weights,
then the same loop validates and restores optimizer/schedule/scaler, exact data
cursor, RNG and counters before the next update. Evaluation performs no updates.

YAML keys match dataclass fields: `hidden`, `accumulation`, `fp16`,
`budget_policy` and `budget_limit`. Unknown fields fail. Model/data/evaluation
contexts, vocabulary identities, device availability and FP16/CUDA compatibility
are checked. Paths are relative to the working directory unless absolute.
`--validate-only` checks schema and device without loading data or constructing an
optimizer; full data/tokenizer checks happen before model construction.

The supplied recipe retains 11 layers, width 512, 8 heads, model context 512,
microbatch 2, accumulation 32, FP16 and 1,000 optimizer steps. Formerly absent
feedforward 2048/dropout 0.1 follow the verified larger-backbone settings; remaining
optimizer values are explicit existing `TrainConfig` defaults. Training context
512/stride 128 is explicit; FineWeb validation remains **256/128**. Validation and
checkpoint intervals are 100. This recipe is runnable after data setup; it is not
a tuned baseline. CPU use requires `device: cpu` and `fp16: false`.

## Setup and commands

From the standalone project root, use the project-local virtual environment:

```powershell
$py = (Resolve-Path .venv\Scripts\python.exe).Path
New-Item -ItemType Directory -Force .tmp | Out-Null
& $py -B -m pytest -q --basetemp=.tmp/tests
& $py -B -m scripts.prepare_fineweb --help
& $py -B -m scripts.train_dense --help
& $py -B -m scripts.train_dense --config configs/cfg.yaml --validate-only
```

For a future independent checkout, create a local environment and install the
editable package with development dependencies. Choose a PyTorch wheel suitable
for your CPU/CUDA system when installing.

```powershell
py -3.10 -m venv .venv
$py = (Resolve-Path .venv\Scripts\python.exe).Path
& $py -m pip install -e ".[dev]"
```

FineWeb uses the pinned source revision, selection, byte-BPE family and shard
format in `configs/fineweb.yaml`. For an approved **new destination only**, the
preparation sequence is:

```powershell
& $py -B -m scripts.prepare_fineweb --stage source
& $py -B -m scripts.prepare_fineweb --stage download
& $py -B -m scripts.prepare_fineweb --stage raw
& $py -B -m scripts.prepare_fineweb --stage tokenizers
& $py -B -m scripts.prepare_fineweb --stage shards
& $py -B -m scripts.prepare_fineweb --stage verify --publish
```

Source/download require network approval; other stages create artifacts. None
were executed on real data in this task. `--root PATH` selects another destination.
For an existing generation, `--stage verify` is offline and read-only. If its
recorded producing code differs from the current implementation, review that
provenance before using `--allow-code-provenance-mismatch`. Artifact integrity
checks remain strict, and existing manifests must never be rewritten. Set
`data.root`/`data.tokenizer` to your prepared artifacts. Verification does not
compute model BPB.

With prepared data configured, the actual CLI commands are:

```powershell
& $py -B -m scripts.train_dense --config configs/cfg.yaml
& $py -B -m scripts.train_dense --config configs/cfg.yaml --resume checkpoints/dense/last.pt
& $py -B -m scripts.train_dense --config configs/cfg.yaml --evaluate checkpoints/dense/last.pt
```

The same CLI paths were exercised with tiny synthetic shards; the supplied
corpus recipe was validated only. Fresh training requires an empty output
directory. `last.pt` is complete and resumable; `best.pt` tracks periodic minimum
validation BPB. Resume keeps all scientific settings and source/environment
identity fixed. `checkpoint.resume` is the YAML equivalent of `--resume`.

## Research notes

Tied embeddings share one vocabulary matrix between lookup and output logits,
so its parameter/storage count appears once. Backbone parameters are attention,
feedforward, normalization and positional parameters; vocabulary parameters scale
with vocabulary size × hidden width.

Validation BPB is negative log likelihood divided by `ln(2) × raw UTF-8 bytes`.
Documents retain boundaries and bytes; prefix EOS is context only, closing EOS is
scored with zero bytes. Overlap supplies context, with every target scored once.
FineWeb is the only supported dataset protocol. Token perplexities are not
comparable across tokenizers.

An `optimizer_steps` budget stops after successful updates; a `raw_bytes` budget
stops after executed bytes, including overflow attempts. Accumulation groups
finish atomically, so byte overshoot is reported. Exposure, attempts, successful
updates and estimated compute remain distinct. Storage accounting separates
trainable/total parameters, persistent tensors, serialized model/tokenizer bytes,
optimizer state and transient runtime/GPU memory. Retained PQ branches are not a
standalone compression experiment or evidence of compression quality.
