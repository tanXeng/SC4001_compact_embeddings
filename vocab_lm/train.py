"""One AdamW update loop, independent of vocabulary representation.

Order: collect CPU microbatches, count targets, zero grads, backward summed CE
divided by total scored targets, unscale, clip, attempt update, update scaler,
advance scheduler iff successful, validate/checkpoint at completed boundaries.
Overflow consumes data/exposure but never advances successful steps or schedule.
"""
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .checkpoint import _identity, save_training_checkpoint
from .config import evaluation_config, TrainConfig
from .evaluate import evaluate
from .runtime import Schedule, environment, restore_rng, rng_state, seed_all


def accumulated_backward(model, batches, *, device, scaler, fp16):
    """One normalization across all valid targets, including unequal short rows."""
    total = sum(batch["tokens"] for batch in batches)
    if total <= 0:
        raise ValueError("Update has no scored targets")
    for batch in batches:
        with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
            logits = model(batch["inputs"].to(device))
            loss = F.cross_entropy(logits.flatten(0, 1), batch["targets"].to(device).flatten(),
                                   ignore_index=-100, reduction="sum") / total
        scaler.scale(loss).backward()


def optimizer_update(model, optimizer, scaler, clip_norm):
    """Public GradScaler scale reduction identifies skips (FP16 scalar sync only)."""
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip_norm,
                                  error_if_nonfinite=not scaler.is_enabled())
    previous = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() >= previous


def train(model, data, config: TrainConfig, *, validation_documents=None,
          validation_identity: dict, output: Path | None = None, resume: dict | None = None,
          max_attempts: int | None = None, on_update=None) -> dict:
    """Train at safe update boundaries. max_attempts provides a resumable interruption.

    Pass load_training_checkpoint's training_state as resume, with its returned
    model. To continue an inference/pretrained model, explicitly choose trainable
    parameters and omit resume. Caller seeds before model initialization.
    """
    device = next(model.parameters()).device
    _identity(validation_identity, "dataset")
    validation_config = evaluation_config(validation_identity["protocol"])
    if validation_documents is not None and validation_config.context > model.config.context:
        raise ValueError("Model context cannot cover validation protocol")
    if config.fp16 and device.type != "cuda":
        raise ValueError("FP16 training requires CUDA")
    if (data.context != config.context or data.stride != config.stride
            or data.seed != config.seed or config.context > model.config.context):
        raise ValueError("Training/data context, stride or ordering seed mismatch")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Require FP32 master parameters")
    if config.validation_interval and validation_documents is None:
        raise ValueError("Validation interval requires validation documents")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, betas=tuple(config.betas),
                                 eps=config.epsilon, weight_decay=config.weight_decay,
                                 foreach=False, fused=False)
    scheduler = Schedule(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=config.fp16)
    counters, best = Counters(), None
    execution = environment(device)
    free_cuda_bytes = torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else None
    if resume is not None:
        if resume["config"] != asdict(config) or resume["validation_identity"] != validation_identity:
            raise ValueError("Resume configuration or validation identity changed")
        if resume["environment"] != execution:
            raise ValueError("Exact resume requires the same pinned execution environment")
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        data.load_state_dict(resume["data"])
        counters = Counters(**resume["counters"])
        if scheduler.completed != counters.successful_updates:
            raise ValueError("Scheduler/update counters disagree")
        best = resume["best_bpb"]
        restore_rng(resume["rng"])
    else:
        seed_all(config.seed)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wall_start, wall_base = time.perf_counter(), counters.wall_seconds
    start_attempts = counters.attempted_updates

    def state():
        counters.wall_seconds = wall_base + time.perf_counter()-wall_start
        return {"config": asdict(config), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "rng": rng_state(), "data": data.state_dict(), "counters": asdict(counters),
                "best_bpb": best, "environment": execution,
                "cuda_free_bytes_at_session_start": free_cuda_bytes,
                "validation_identity": validation_identity,
                "budget_overshoot": overshoot(counters, config.budget_policy, config.budget_limit),
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda" else None}

    def save(name):
        if output is not None:
            save_training_checkpoint(Path(output)/name, model, state(),
                                     tokenizer=data.identity["tokenizer"], dataset=validation_identity)

    while budget_value(counters, config.budget_policy) < config.budget_limit:
        if max_attempts is not None and counters.attempted_updates-start_attempts >= max_attempts:
            break
        active_start = time.perf_counter()
        batches = [data.batch(config.microbatch_size) for _ in range(config.accumulation)]
        optimizer.zero_grad(set_to_none=True)
        lr = optimizer.param_groups[0]["lr"]
        accumulated_backward(model, batches, device=device, scaler=scaler, fp16=config.fp16)
        successful = optimizer_update(model, optimizer, scaler, config.clip_norm)
        counters.attempted_updates += 1
        counters.successful_updates += int(successful)
        counters.skipped_updates += int(not successful)
        counters.microsteps += len(batches)
        for batch in batches:
            counters.scored_tokens += batch["tokens"]
            counters.raw_bytes += batch["bytes"]
            counters.input_positions += batch["inputs"].numel()
            counters.estimated_flops += training_flops(model, *batch["inputs"].shape)
        counters.active_seconds += time.perf_counter()-active_start
        if successful:
            scheduler.step()
            if config.validation_interval and counters.successful_updates % config.validation_interval == 0:
                result = evaluate(model, validation_documents, validation_config,
                                  expected_bytes=validation_identity.get("bytes"))
                if best is None or result["bpb"] < best:
                    best = result["bpb"]
                    save("best.pt")
        if on_update is not None:
            on_update({"successful": successful, "lr": lr,
                       "examples": [e for b in batches for e in b["examples"]]})
        if config.checkpoint_interval and counters.attempted_updates % config.checkpoint_interval == 0:
            save("last.pt")
    save("last.pt")
    return state()



"""Exposure and update counters; times are observations, not deterministic state."""
@dataclass
class Counters:
    attempted_updates: int = 0
    successful_updates: int = 0
    skipped_updates: int = 0
    microsteps: int = 0
    scored_tokens: int = 0
    raw_bytes: int = 0
    input_positions: int = 0
    estimated_flops: int = 0
    active_seconds: float = 0.0
    wall_seconds: float = 0.0


def budget_value(counters: Counters, policy: str) -> int:
    if policy == "optimizer_steps":
        return counters.successful_updates
    if policy == "raw_bytes":
        return counters.raw_bytes
    raise ValueError("Unsupported budget policy")


def overshoot(counters: Counters, policy: str, limit: int) -> int:
    return max(0, budget_value(counters, policy) - limit)


def training_flops(model, batch: int, time: int) -> int:
    """Dense forward+backward estimate: 6NP + 12LB T²d.

    P includes vocabulary projection and backbone matrices/affines. Attention
    products are added separately. Ignores optimizer, softmax, recomputation,
    gather/scatter work and hardware utilization; includes padded executed positions.
    Compressed representations still pay for an effective dense output projection.
    """
    c = model.config
    # Both representations execute a dense V*d output projection. Counting only
    # compressed parameters would falsely imply a reduction in projection work.
    parameters = (sum(p.numel() for p in model.parameters())
                  - sum(p.numel() for p in model.vocabulary.parameters())
                  + model.vocabulary.vocab_size * c.hidden)
    return 6*batch*time*parameters + 12*c.layers*batch*time*time*c.hidden
