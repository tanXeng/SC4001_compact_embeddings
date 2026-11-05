"""Independent effective-matrix references, synthetic only."""
from dataclasses import asdict
import json

import pytest
import torch
from torch.nn import functional as F

from vocab_lm.model import (CausalTransformer, LowRankVocabulary, HashVocabulary,
                            AdaptiveVocabulary)
from vocab_lm.config import ModelConfig
from vocab_lm.checkpoint import (archive_bytes, save_checkpoint, load_checkpoint,
    save_training_checkpoint, load_training_checkpoint)
from vocab_lm.accounting import storage_report, vocabulary_storage_report
from vocab_lm.train import training_flops, compressed_vocabulary_flops
from vocab_lm.tokenizer import FineWebEvaluationConfig



def make(kind):
    if kind == "low_rank":
        return LowRankVocabulary(11, 8, 3)
    if kind == "hash":
        return HashVocabulary(11, 8, 4, (3, 7))
    return AdaptiveVocabulary(torch.tensor([0, 6, 9, 6, 3, 2, 1, 8, 0, 0, 0]), 8, [3, 7], factor=2)


def reference(v):
    if isinstance(v, LowRankVocabulary):
        return v.a @ v.b
    if isinstance(v, HashVocabulary):
        return torch.stack([sum(v.importance[i, j]*v.components[int(v.indices[i, j])]
                                for j in range(len(v.seeds))) for i in range(v.vocab_size)])
    rows = [None]*v.vocab_size
    start = 0
    for end, a, b in zip(v.cutoffs, v.tables, v.projections):
        for offset in range(end-start):
            rows[int(v.order[start+offset])] = a[offset] @ b
        start = end
    return torch.stack(rows)


@pytest.mark.parametrize("kind", ["low_rank", "hash", "adaptive"])
def test_lookup_logits_and_gradients(kind):
    v = make(kind).double()
    ids = torch.tensor([[0, 3, 3, 10], [2, 6, 1, 8]])
    h = torch.randn(2, 4, 8, dtype=torch.double, requires_grad=True)
    matrix = reference(v)
    assert v.lookup(ids).shape == (2, 4, 8)
    torch.testing.assert_close(v.lookup(ids), matrix[ids])
    torch.testing.assert_close(v.project(h), h @ matrix.T)
    variables = [h, *v.parameters()]
    actual = v.lookup(ids).square().sum() + v.project(h).square().sum()
    expected = matrix[ids].square().sum() + (h @ matrix.T).square().sum()
    ga = torch.autograd.grad(actual, variables)
    ge = torch.autograd.grad(expected, variables)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e)
        assert torch.isfinite(a).all() and a.norm() > 0
    # Finite-difference check of both paths with respect to hidden states.
    assert torch.autograd.gradcheck(v.project, (h,), fast_mode=True)


@pytest.mark.parametrize("kind", ["low_rank", "hash", "adaptive"])
def test_checkpoint_storage_updates_and_backbone(kind, tmp_path):
    config = ModelConfig(1, 8, 2, 16, 16, 0.)
    torch.manual_seed(42)
    dense = CausalTransformer(config, 11)
    torch.manual_seed(42)
    model = CausalTransformer(config, 11)
    model.vocabulary = make(kind)
    for name, tensor in dense.state_dict().items():
        if not name.startswith("vocabulary."):
            assert torch.equal(model.state_dict()[name], tensor)
    assert not any("head" in n for n, _ in model.named_parameters())
    assert sum(p.numel() for p in dense.vocabulary.parameters()) == 11*8
    expected_parameters = {"low_rank": 3*(11+8), "hash": 4*8+11*2,
                           "adaptive": 3*8+8*8+4*4+4*8+4*2+2*8}
    assert sum(p.numel() for p in model.vocabulary.parameters()) == expected_parameters[kind]
    ids = torch.tensor([[0, 1, 3, 10]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    before = {n: p.clone() for n, p in model.vocabulary.named_parameters()}
    for _ in range(2):
        optimizer.zero_grad()
        F.cross_entropy(model(ids).flatten(0, 1), torch.tensor([1, 3, 10, 0])).backward()
        optimizer.step()
    assert all(not torch.equal(p, before[n]) for n, p in model.vocabulary.named_parameters())
    metadata = dict(tokenizer={"sha256": "a"*64}, dataset={"sha256": "b"*64,
        "split": "validation", "protocol": asdict(FineWebEvaluationConfig())})
    path = tmp_path / "model.pt"
    size = save_checkpoint(path, model, **metadata)
    restored, _ = load_checkpoint(path)
    assert restored.vocabulary.specification() == model.vocabulary.specification()
    for n, tensor in model.state_dict().items():
        assert torch.equal(restored.state_dict()[n], tensor)
    assert torch.equal(model.eval()(ids), restored(ids))
    training = tmp_path / "training.pt"
    save_training_checkpoint(training, model, {"sentinel": 123}, **metadata)
    reloaded, payload = load_training_checkpoint(training)
    assert payload["training_state"]["sentinel"] == 123
    assert torch.equal(reloaded.eval()(ids), restored(ids))
    report = storage_report(model, optimizer_state=optimizer.state_dict(), **metadata)
    assert report["serialized_inference_checkpoint_bytes"] == size == path.stat().st_size
    assert report["vocabulary_parameter_bytes"] == 4*expected_parameters[kind]
    assert report["vocabulary_fixed_buffer_bytes"] == (176 if kind in {"hash", "adaptive"} else 0)
    assert report["optimizer_state_tensor_bytes"] > 0
    v = model.vocabulary
    expected_archive = archive_bytes(dict(specification=v.specification(),
                         state_dict={n: t.detach().cpu().clone() for n, t in v.state_dict().items()}))
    assert report["vocabulary_serialized_inference_bytes"] == len(expected_archive)
    assert report["vocabulary_metadata_json_bytes"] == len(json.dumps(v.specification(), sort_keys=True).encode())




def test_adaptive_mapping_and_invalid_state():
    v = make("adaptive")
    assert v.order.tolist() == [0, 2, 7, 1, 3, 4, 5, 6, 8, 9, 10]
    assert torch.equal(v.order[v.inverse], torch.arange(11))
    state = v.state_dict()
    state["inverse"] = state["inverse"].roll(1)
    with pytest.raises(ValueError, match="permutation"):
        make("adaptive").load_state_dict(state)
    with pytest.raises(ValueError):
        AdaptiveVocabulary(torch.ones(10, dtype=torch.long), 8, [5, 4])


@pytest.mark.parametrize("kind", ["low_rank", "hash", "adaptive"])
def test_flops_and_optional_diagnostics(kind):
    v = make(kind)
    n = 4
    estimate = compressed_vocabulary_flops(v, n)
    if kind == "low_rank":
        expected = 6*n*3*(11+2*8)
    elif kind == "hash":
        expected = 6*n*4*8+n*(8+11)*(5*2-1)
    else:
        expected = 6*n*(8*(3+8)+4*(4+8)+2*(4+8))+6*n*8*8
    assert estimate["estimated_flops"] == expected


def test_dense_legacy_initialization_forward_and_flops():
    config = ModelConfig(1, 8, 2, 16, 16, 0.)
    torch.manual_seed(42)
    model = CausalTransformer(config, 11)
    # Independent explicit forward through unchanged backbone and single dense table.
    ids = torch.tensor([[0, 2, 5]])
    x = F.embedding(ids, model.vocabulary.weight)+model.position(torch.arange(3))
    for block in model.blocks:
        x = block(x)
    assert torch.equal(model(ids), F.linear(model.norm(x), model.vocabulary.weight))
    p = sum(t.numel() for t in model.parameters())
    assert training_flops(model, 1, 3) == 6*3*p+12*config.layers*3*3*8


@pytest.mark.parametrize("kind", ["low_rank", "hash", "adaptive"])
def test_shared_loop_exact_resume(kind, tmp_path):
    from vocab_lm.config import TrainConfig
    from vocab_lm.data import TrainingData
    from vocab_lm.evaluate import Document
    from vocab_lm.train import train
    from vocab_lm.runtime import rng_state, restore_rng
    config = TrainConfig(context=4, stride=2, budget_limit=3, seed=42)
    model = CausalTransformer(ModelConfig(1, 8, 2, 16, 256, .1), 11)
    model.vocabulary = make(kind)
    documents = [Document((0, 1, 2, 3, 4, 0), (0, 1, 1, 1, 1, 0), "c"*64)]
    def data():
        return TrainingData(documents, context=4, stride=2, seed=42,
            tokenizer={"sha256": "a"*64}, dataset={"sha256": "b"*64, "split": "train"})
    validation = {"sha256": "d"*64, "split": "validation",
                  "protocol": asdict(FineWebEvaluationConfig())}
    initial = {n: t.clone() for n, t in model.state_dict().items()}
    rng = rng_state()
    full = train(model, data(), config, validation_identity=validation)
    final = {n: t.clone() for n, t in model.state_dict().items()}
    model.load_state_dict(initial)
    restore_rng(rng)
    train(model, data(), config, validation_identity=validation, output=tmp_path, max_attempts=1)
    restored, payload = load_training_checkpoint(tmp_path/"last.pt")
    resumed = train(restored, data(), config, validation_identity=validation,
                    resume=payload["training_state"], output=tmp_path)
    assert all(torch.equal(t, restored.state_dict()[n]) for n, t in final.items())
    assert full["counters"]["estimated_flops"] == resumed["counters"]["estimated_flops"]
