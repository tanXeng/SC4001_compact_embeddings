# FP32 vocabulary-scaling study

Use the frozen large-study recipe and [operator/methodology instructions](docs/VOCAB_SCALING.md).

```bash
python scripts/vocab_scaling.py prepare --config configs/vocab_scaling_large.yaml
python scripts/vocab_scaling.py preflight --config configs/vocab_scaling_large.yaml
```

Train/resume the assigned arm only after the actual 32 GB server preflight passes.
Downloaded data, checkpoints, logs and local historical records are excluded from Git.
