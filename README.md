# Vocabulary storage and IsoFLOPs experiments

Frozen FP32 Transformer experiments at16K/32K/64K vocabulary sizes.
[Methodology and complete fresh-clone workflow](docs/REPRODUCTION.txt).
[Published analysis data](analysis/data/SCHEMA.txt) needs no datasets/checkpoints.

```sh
git clone <repository-url> vocab-study
cd vocab-study
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/reproduce.py verify --study 64k --run replication1
python scripts/reproduce.py prepare --study 64k --run replication1
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py preflight --study 64k --run replication1
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py authorize --study 64k --run replication1 --accept-frozen-protocol
```

Use `--study large` for16K/32K. The document gives all training/resume, evaluation,
once-only confirmation and diagnostic/report commands. GPU preflight and explicit
local authorization are required before training. No private historical receipt
is needed. New runs live under `results/reproductions/`; committed CSV/JSON data
are the frozen outputs of our completed runs and are never overwritten.
