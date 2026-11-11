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

## Build the LaTeX report

The entry point is `report/main.tex`. Use XeLaTeX (required by `fontspec`),
BibTeX (`natbib` with `unsrtnat`), and `latexmk` with Perl. On Windows, install
basic MiKTeX and native Strawberry Perl, and let MiKTeX's package manager supply
missing packages. Ensure MiKTeX and Perl are on `PATH`, then reopen your
terminal/editor after installation. Use native Perl rather than Git's MSYS Perl,
which can convert Windows paths incorrectly.
Arial is used when available; otherwise the source requires Liberation Sans.

From the repository root in PowerShell:

```powershell
latexmk -cd -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build report/main.tex
```

The PDF is `report/build/main.pdf`; `latexmk` runs BibTeX and repeats XeLaTeX
until references settle. `build/` is ignored. For a full rebuild, add `-gg`
(removes generated outputs in that build directory before compiling).
The tracked `report/main.pdf` and `report/figures/*.pdf` are preserved.
The committed figures suffice for compilation; no Python, datasets, or GPU are
needed. Optional figure regeneration uses `report/make_figures.py` and the
committed `analysis/data/` inputs (see the script for Python dependencies).
