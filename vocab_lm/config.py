"""Explicit architecture, training and FineWeb scoring configuration."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ModelConfig:
    layers: int
    hidden: int
    heads: int
    feedforward: int
    context: int
    dropout: float

    def __post_init__(self) -> None:
        dimensions = (self.layers, self.hidden, self.heads, self.feedforward, self.context)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("Model dimensions must be positive integers")
        if self.hidden % self.heads:
            raise ValueError("Hidden dimension must be divisible by attention heads")
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float))
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError("Dropout must be finite and in [0, 1)")
        if self.backbone_parameters >= 50_000_000:
            raise ValueError("This research framework requires a backbone below 50M parameters")

    @property
    def backbone_parameters(self) -> int:
        # QKV/output + two FF matrices + their biases + two affine LayerNorms.
        per_block = 4*self.hidden**2 + 2*self.hidden*self.feedforward
        per_block += 6*self.hidden + self.feedforward
        return self.layers*per_block + (self.context + 2)*self.hidden


@dataclass(frozen=True)
class TrainConfig:
    microbatch_size: int = 2
    context: int = 128
    stride: int = 64
    accumulation: int = 1
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    clip_norm: float = 1.0
    schedule: str = "constant"
    warmup_updates: int = 0
    decay_updates: int = 1000
    budget_policy: str = "optimizer_steps"
    budget_limit: int = 10
    seed: int = 42
    fp16: bool = False
    validation_interval: int = 0
    checkpoint_interval: int = 0
    flop_estimator: str = "legacy"
    checkpoint_seconds: float = 0

    def __post_init__(self):
        for name in ("microbatch_size", "context", "stride", "accumulation",
                     "decay_updates", "budget_limit"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_updates", "validation_interval", "checkpoint_interval", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.stride > self.context or self.schedule not in {"constant", "linear", "cosine"}:
            raise ValueError("Invalid training stride/schedule")
        if self.budget_policy not in {"optimizer_steps", "raw_bytes", "flops"}:
            raise ValueError("Unsupported budget policy")
        if self.flop_estimator not in {"legacy", "vocab_scaling_v1"} or self.checkpoint_seconds < 0:
            raise ValueError("Invalid estimator/checkpoint interval")
        for name in ("learning_rate", "epsilon", "clip_norm", "weight_decay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}")
        if min(self.learning_rate, self.epsilon, self.clip_norm) <= 0:
            raise ValueError("Learning rate, epsilon and clipping must be positive")
        if len(self.betas) != 2 or any(not math.isfinite(b) or not 0 <= b < 1 for b in self.betas):
            raise ValueError("Invalid AdamW betas")
        if type(self.fp16) is not bool:
            raise ValueError("fp16 must be boolean")


def evaluation_config(settings: dict):
    """Validate the fixed FineWeb validation protocol."""
    from .tokenizer import FineWebEvaluationConfig
    return FineWebEvaluationConfig(**settings)


@dataclass(frozen=True)
class DataConfig:
    root: str
    tokenizer: str
    tier: str = "smoke"

    def __post_init__(self):
        if (not isinstance(self.root, str) or not self.root
                or not isinstance(self.tokenizer, str) or not self.tokenizer
                or self.tier not in {"smoke", "pilot"}):
            raise ValueError("Require dataset/tokenizer paths and smoke or pilot tier")


@dataclass(frozen=True)
class CheckpointConfig:
    output: str
    resume: str | None = None

    def __post_init__(self):
        if not isinstance(self.output, str) or not self.output:
            raise ValueError("Require checkpoint output directory")
        if self.resume is not None and (not isinstance(self.resume, str) or not self.resume):
            raise ValueError("Resume must be a checkpoint path or null")


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    training: TrainConfig
    data: DataConfig
    evaluation: object
    checkpoint: CheckpointConfig
    vocab_size: int
    device: str


def load_config(path) -> RunConfig:
    """Strict YAML schema; paths are relative to the standalone working directory."""
    from pathlib import Path
    import yaml
    import torch
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"model", "training", "data", "evaluation", "checkpoint", "vocab_size", "device"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(f"Configuration requires exactly {sorted(required)}")
    training = dict(settings["training"])
    if "betas" in training:
        training["betas"] = tuple(training["betas"])
    result = RunConfig(ModelConfig(**settings["model"]), TrainConfig(**training),
                       DataConfig(**settings["data"]), evaluation_config(settings["evaluation"]),
                       CheckpointConfig(**settings["checkpoint"]), settings["vocab_size"], settings["device"])
    if type(result.vocab_size) is not int or result.vocab_size < 1:
        raise ValueError("Vocabulary size must be a positive integer matching prepared shards")
    if result.training.context > result.model.context or result.evaluation.context > result.model.context:
        raise ValueError("Model context must cover training and validation")
    device = torch.device(result.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU/CUDA execution is supported")
    if device.type == "cuda" and (not torch.cuda.is_available() or
            (device.index is not None and device.index >= torch.cuda.device_count())):
        raise ValueError("Configured CUDA device is unavailable")
    if result.training.fp16 and device.type != "cuda":
        raise ValueError("FP16 training requires CUDA")
    return result
