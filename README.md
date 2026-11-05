# FP32 vocabulary-scaling study

Use the frozen large-study recipe and [operator/methodology instructions](docs/VOCAB_SCALING.md).

```bash
python scripts/vocab_scaling.py prepare --config configs/vocab_scaling_large.yaml
python scripts/vocab_scaling.py preflight --config configs/vocab_scaling_large.yaml
```

Train/resume only after the final A40 preflight passes: 5e17 FLOPs per arm,
microbatch16 x accumulation2, one independent arm per GPU, no DDP.
Downloaded data, checkpoints, logs and local historical records are excluded from Git.
