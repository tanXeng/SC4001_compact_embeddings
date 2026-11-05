"""Pinned-environment RNG handling and successful-update learning-rate policy."""
import math
from importlib.metadata import version
import random
import subprocess
import sys
from pathlib import Path
import hashlib
import json

import numpy as np
import torch


# Shared file, hash and JSON helpers (also used by data and tokenizers).
def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    # Preserve the existing Windows-produced manifest bytes on every host.
    with Path(path).open('w',encoding='utf-8',newline='\r\n') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n')


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    numpy = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["cpu"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count changed")
        torch.cuda.set_rng_state_all(state["cuda"])


def source_identity() -> dict:
    """A missing Git revision is explicit; content identity remains mandatory."""
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    digest = hashlib.sha256()
    paths = list((root / "vocab_lm").glob("*.py")) + list((root / "scripts").glob("*.py"))
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes().replace(b'\r\n',b'\n') + b"\0")
    return {"git_commit": commit, "source_sha256": digest.hexdigest()}


def environment(device) -> dict:
    return {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
            "packages": {name: version(name) for name in ("torch", "numpy", "tokenizers", "pyyaml")},
            **source_identity(),
            "device": str(device), "threads": torch.get_num_threads(),
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "sdpa": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                     "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                     "math": torch.backends.cuda.math_sdp_enabled()},
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory
            if device.type == "cuda" else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


class Schedule:
    """Set LR for update (completed+1); advance only after a successful update."""
    def __init__(self, optimizer, config):
        self.optimizer, self.config, self.completed = optimizer, config, 0
        self.apply()

    def apply(self):
        c, next_update = self.config, self.completed + 1
        if next_update <= c.warmup_updates:
            factor = next_update / c.warmup_updates
        elif c.schedule == "constant":
            factor = 1.0
        else:
            progress = min(1.0, max(0.0, (next_update-c.warmup_updates-1) /
                                     max(1, c.decay_updates-c.warmup_updates)))
            factor = 1-progress if c.schedule == "linear" else (1+math.cos(math.pi*progress))/2
        for group in self.optimizer.param_groups:
            group["lr"] = c.learning_rate*factor

    def step(self):
        self.completed += 1
        self.apply()

    def state_dict(self):
        return {"completed": self.completed}

    def load_state_dict(self, state):
        self.completed = state["completed"]
        self.apply()
