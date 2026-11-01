"""Gate 2 synthetic training only; Gate 1's optimizer prohibition stays intact."""
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import yaml

from vocab_lm.train import Counters, overshoot
from vocab_lm.checkpoint import load_training_checkpoint, load_checkpoint, save_checkpoint
from vocab_lm.config import ModelConfig, TrainConfig
from vocab_lm.tokenizer import FineWebEvaluationConfig
from vocab_lm.data import TrainingData
from vocab_lm.evaluate import Document
from vocab_lm.model import CausalTransformer
from vocab_lm.runtime import Schedule, restore_rng, rng_state
from vocab_lm.train import accumulated_backward, optimizer_update, train


@pytest.fixture
def settings():
    return yaml.safe_load(Path("configs/testing.yaml").read_text())


@pytest.fixture
def config(settings):
    return TrainConfig(**settings["training"])


@pytest.fixture
def model(settings):
    return CausalTransformer(ModelConfig(**settings["model"]), settings["vocab_size"])


@pytest.fixture
def documents():
    return [Document((0, *ids, 0), (0, *sizes, 0), hashlib.sha256(bytes(ids)).hexdigest())
            for ids, sizes in [([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 1, 3, 1, 1, 2, 1, 1]),
                               ([10], [4]), ([11, 12, 13], [1, 2, 1])]]


def make_data(documents, config):
    return TrainingData(documents, context=config.context, stride=config.stride, seed=config.seed,
                        tokenizer={"sha256": "a"*64}, dataset={"sha256": "b"*64, "split": "train"})


@pytest.mark.parametrize('permanent', [False, True])
def test_checkpoint_access_denied_retry(model, validation, tmp_path, monkeypatch, permanent):
    import vocab_lm.checkpoint as checkpoint
    path = tmp_path/'last.pt'
    metadata = dict(tokenizer={'sha256': 'a'*64}, dataset=validation)
    checkpoint.save_training_checkpoint(path, model, {'marker': 'old'}, **metadata)
    previous = path.read_bytes()
    replace = checkpoint.os.replace
    errors, delays = [], []
    def denied(source, destination):
        assert source.parent.resolve() == path.parent.resolve()
        assert path.read_bytes() == previous
        if permanent or len(errors) < 2:
            error = PermissionError('simulated Windows access denial')
            errors.append(error)
            raise error
        return replace(source, destination)
    monkeypatch.setattr(checkpoint.os, 'replace', denied)
    monkeypatch.setattr(checkpoint.time, 'sleep', delays.append)
    if permanent:
        with pytest.raises(PermissionError) as caught:
            checkpoint.save_training_checkpoint(path, model, {'marker': 'new'}, **metadata)
        assert caught.value is errors[0]
        assert path.read_bytes() == previous
        assert len(errors) == 5
        assert delays == [.05, .1, .2, .4]
    else:
        checkpoint.save_training_checkpoint(path, model, {'marker': 'new'}, **metadata)
        restored, payload = checkpoint.load_training_checkpoint(path)
        assert payload['training_state']['marker'] == 'new'
        assert_tree(model.state_dict(), restored.state_dict())
        assert delays == [.05, .1]
    assert not list(tmp_path.glob('*.tmp'))


@pytest.fixture
def validation():
    return {"sha256": "c"*64, "split": "validation", "protocol": asdict(FineWebEvaluationConfig()), "bytes": 22}


def assert_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_tree(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree(x, y)
    else:
        assert a == b


def test_order_cursor_bytes_and_boundaries(documents, config):
    data = make_data(documents, config)
    rows = []
    while data.cursor.epoch == 0:
        rows.append(data.next_example())
    assert sum(r[2] for r in rows) == sum(sum(d.byte_lengths) for d in documents)
    assert sum(sum(t != -100 for t in r[1]) for r in rows) == sum(len(d.ids)-1 for d in documents)
    for x, y, _, (source, first) in rows:
        d = documents[source]
        targets = [t for t in y if t != -100]
        assert targets == list(d.ids[first:first+len(targets)])
        assert tuple(x) in [d.ids[i:i+len(x)] for i in range(len(d.ids))]
    snapshot = data.state_dict()
    expected = [data.next_example() for _ in range(12)]
    restored = make_data(documents, config)
    restored.load_state_dict(snapshot)
    assert expected == [restored.next_example() for _ in range(12)]
    repeat = make_data(documents, config)
    assert rows == [repeat.next_example() for _ in rows]
    changed = make_data(documents, replace(config, seed=config.seed+1))
    with pytest.raises(ValueError, match="identity"):
        changed.load_state_dict(snapshot)


@pytest.mark.parametrize("sizes", [(1, 1, 1, 1), (3, 1)])
def test_accumulation_unequal_targets(model, documents, config, sizes):
    model.eval()  # dropout disabled for mathematical equivalence
    data = make_data(documents, config)
    batch = data.batch(4)
    # Keep equal tensor extents, but variable valid-target counts and padding.
    pieces, start = [], 0
    for size in sizes:
        end = start+size
        pieces.append({"inputs": batch["inputs"][start:end], "targets": batch["targets"][start:end],
                       "tokens": int((batch["targets"][start:end] != -100).sum())})
        start = end
    assert len({p["tokens"] for p in pieces}) > 1
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    accumulated_backward(model, [batch], device=torch.device("cpu"), scaler=scaler, fp16=False)
    expected = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    accumulated_backward(model, pieces, device=torch.device("cpu"), scaler=scaler, fp16=False)
    for p, g in zip(model.parameters(), expected):
        torch.testing.assert_close(p.grad, g, atol=2e-7, rtol=2e-5)


def test_causal_and_padding(model):
    model.eval()
    a = torch.tensor([[0, 1, 2, 3]])
    b = torch.tensor([[0, 1, 8, 9]])
    torch.testing.assert_close(model(a)[:, :2], model(b)[:, :2], atol=0, rtol=0)
    padded = torch.tensor([[0, 1, 2, 3], [0, 10, 0, 0]])
    torch.testing.assert_close(model(padded)[1, :2], model(padded[1:2, :2])[0], atol=1e-7, rtol=1e-5)


def test_rng():
    state = rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng(state)
    assert_tree(expected, (random.random(), np.random.rand(), torch.rand(3)))


@pytest.mark.parametrize("device,fp16", [("cpu", False), ("cuda", False), ("cuda", True)])
def test_exact_resume(model, documents, config, validation, tmp_path, device, fp16):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = replace(config, fp16=fp16)
    model.to(device)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    full_trace, split_trace = [], []
    full = train(model, make_data(documents, config), config, validation_identity=validation,
                 on_update=full_trace.append)
    final = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(initial)
    interrupted = train(model, make_data(documents, config), config, validation_identity=validation,
                        output=tmp_path, max_attempts=2, on_update=split_trace.append)
    restored, payload = load_training_checkpoint(tmp_path/"last.pt", device=device)
    assert_tree(payload["training_state"]["optimizer"], interrupted["optimizer"])
    resumed = train(restored, make_data(documents, config), config, validation_identity=validation,
                    resume=payload["training_state"], output=tmp_path, on_update=split_trace.append)
    assert_tree(final, restored.state_dict())
    assert full_trace == split_trace
    for key in ("optimizer", "scheduler", "scaler", "data", "rng"):
        assert_tree(full[key], resumed[key])
    for key in full["counters"]:
        if key not in {"active_seconds", "wall_seconds"}:
            assert full["counters"][key] == resumed["counters"][key]
    assert resumed["counters"]["successful_updates"] == config.budget_limit
    assert len(list(restored.vocabulary.parameters())) == 1
    assert not torch.equal(initial["vocabulary.weight"], final["vocabulary.weight"])


def test_skip_schedule_and_step_budget(model, documents, config, validation, monkeypatch):
    import vocab_lm.train as module
    original = module.optimizer_update
    calls = []
    def update(*args):
        calls.append(1)
        return False if len(calls) == 1 else original(*args)
    monkeypatch.setattr(module, "optimizer_update", update)
    trace = []
    result = train(model, make_data(documents, config), config,
                   validation_identity=validation, on_update=trace.append)
    assert result["counters"]["attempted_updates"] == config.budget_limit+1
    assert result["counters"]["skipped_updates"] == 1
    assert result["scheduler"]["completed"] == config.budget_limit
    assert trace[0]["lr"] == trace[1]["lr"] == config.learning_rate/2
    assert trace[2]["lr"] == config.learning_rate


def test_raw_byte_budget(model, documents, config, validation):
    config = replace(config, budget_policy="raw_bytes", budget_limit=1)
    result = train(model, make_data(documents, config), config, validation_identity=validation)
    c = Counters(**result["counters"])
    assert c.attempted_updates == 1
    assert c.raw_bytes > 1
    assert result["budget_overshoot"] == c.raw_bytes-1 == overshoot(c, "raw_bytes", 1)
    assert c.input_positions >= c.scored_tokens > 0
    assert c.estimated_flops > 0


def test_validation_selection(model, documents, config, validation, tmp_path):
    validation = {**validation, "bytes": sum(sum(d.byte_lengths) for d in documents)}
    config = replace(config, budget_limit=2, validation_interval=1)
    result = train(model, make_data(documents, config), config, validation_identity=validation,
                   validation_documents=documents, output=tmp_path)
    _, payload = load_training_checkpoint(tmp_path/"best.pt")
    assert payload["training_state"]["best_bpb"] == result["best_bpb"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_fp16_overflow(model):
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", init_scale=8)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    scaler.scale(model(torch.tensor([[0, 1]], device="cuda")).sum()).backward()
    next(model.parameters()).grad.fill_(float("inf"))
    assert not optimizer_update(model, optimizer, scaler, 1.0)
    assert scaler.get_scale() == 4
    assert not optimizer.state
    assert_tree(before, model.state_dict())


@pytest.mark.parametrize("schedule", ["constant", "linear", "cosine"])
def test_schedule(config, schedule):
    config = replace(config, schedule=schedule)
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    sched = Schedule(opt, config)
    values = []
    for _ in range(10):
        values.append(opt.param_groups[0]["lr"])
        sched.step()
    assert values[:3] == [config.learning_rate/2, config.learning_rate, config.learning_rate]
    assert values[-1] == (config.learning_rate if schedule == "constant" else 0)


def test_reject_changed_resume(model, documents, config, validation, tmp_path):
    train(model, make_data(documents, config), config, validation_identity=validation,
          output=tmp_path, max_attempts=1)
    restored, payload = load_training_checkpoint(tmp_path/"last.pt")
    with pytest.raises(ValueError, match="configuration"):
        train(restored, make_data(documents, config), replace(config, learning_rate=0.02),
              validation_identity=validation, resume=payload["training_state"])


@pytest.mark.parametrize("vocab_size", [17, 31])
def test_continue_pretrained(model, documents, config, validation, tmp_path, vocab_size):
    model = CausalTransformer(model.config, vocab_size)
    model.requires_grad_(False)
    save_checkpoint(tmp_path/"pretrained.pt", model, tokenizer={"sha256": "a"*64}, dataset=validation)
    restored, _ = load_checkpoint(tmp_path/"pretrained.pt")
    restored.requires_grad_(True)  # explicit policy for continued training
    before = restored.vocabulary.weight.detach().clone()
    result = train(restored, make_data(documents, config), replace(config, budget_limit=1),
                   validation_identity=validation)
    assert result["counters"]["successful_updates"] == 1
    assert not torch.equal(before, restored.vocabulary.weight)


def test_data_rejects_test_split(documents, config):
    with pytest.raises(ValueError, match="train split"):
        TrainingData(documents, context=8, stride=4, seed=42,
                     tokenizer={"sha256": "a"*64}, dataset={"sha256": "b"*64, "split": "test"})


@pytest.mark.parametrize("device,fp16", [("cpu", False), ("cuda", False), ("cuda", True)])
def test_fineweb_validation_and_exact_resume(model, documents, config, tmp_path, device, fp16):
    from vocab_lm.tokenizer import FineWebEvaluationConfig
    from vocab_lm.evaluate import evaluate
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = CausalTransformer(replace(model.config, context=256), 17).to(device)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    config = replace(config, budget_limit=3, fp16=fp16, validation_interval=1)
    protocol = FineWebEvaluationConfig()
    validation = {"sha256": "c"*64, "split": "validation", "protocol": asdict(protocol),
                  "bytes": sum(sum(d.byte_lengths) for d in documents)}
    full = train(model, make_data(documents, config), config, validation_identity=validation,
                 validation_documents=documents)
    final = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(initial)
    train(model, make_data(documents, config), config, validation_identity=validation,
          validation_documents=documents, output=tmp_path, max_attempts=1)
    restored, payload = load_training_checkpoint(tmp_path/"last.pt", device)
    assert payload["dataset"] == validation
    resumed = train(restored, make_data(documents, config), config, validation_identity=validation,
                    validation_documents=documents, output=tmp_path, resume=payload["training_state"])
    assert_tree(final, restored.state_dict())
    for key in ("optimizer", "scheduler", "scaler", "data", "rng", "best_bpb"):
        assert_tree(full[key], resumed[key])
    result = evaluate(restored, documents, protocol, expected_bytes=validation["bytes"])
    assert result["tokens"] == sum(len(d.ids)-1 for d in documents)
    save_checkpoint(tmp_path/"inference.pt", restored, tokenizer={"sha256": "a"*64}, dataset=validation)
    inference, saved = load_checkpoint(tmp_path/"inference.pt", device)
    assert saved["dataset"]["protocol"] == asdict(protocol)
    assert_tree(final, inference.state_dict())


@pytest.mark.parametrize("field,value", [("git_commit", "unavailable"), ("source_sha256", "f"*64)])
def test_resume_rejects_changed_source(model, documents, config, validation, field, value):
    state = train(model, make_data(documents, config), config, validation_identity=validation, max_attempts=1)
    state["environment"][field] = value if state["environment"][field] != value else "changed"
    with pytest.raises(ValueError, match="execution environment"):
        train(model, make_data(documents, config), config, validation_identity=validation, resume=state)


@pytest.mark.parametrize("kind", ["inference", "training"])
@pytest.mark.parametrize("change", [dict(protocol="unsupported"), dict(context=128),
                                    dict(stride=64), dict(eos_id=1)])
def test_checkpoint_rejects_unsupported_protocol(model, validation, tmp_path, kind, change):
    from vocab_lm.checkpoint import save_training_checkpoint
    invalid = {**validation, "protocol": {**validation["protocol"], **change}}
    path = tmp_path / "checkpoint.pt"
    if kind == "inference":
        save = lambda dataset: save_checkpoint(path, model, tokenizer={"sha256": "a"*64}, dataset=dataset)
        load = load_checkpoint
    else:
        save = lambda dataset: save_training_checkpoint(path, model, {}, tokenizer={"sha256": "a"*64}, dataset=dataset)
        load = load_training_checkpoint
    with pytest.raises(ValueError, match="FineWeb"):
        save(invalid)
    assert not path.exists()
    save(validation)
    payload = torch.load(path, weights_only=True)
    payload["dataset"] = invalid
    torch.save(payload, path)
    with pytest.raises(ValueError, match="FineWeb"):
        load(path)
