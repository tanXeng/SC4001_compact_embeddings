"""Hard budget and continuation contracts for the shared FP32 loop."""
from dataclasses import asdict
import copy

import pytest
import torch

from scripts.vocab_scaling import construct, diagnose, load_settings
from vocab_lm.checkpoint import load_training_checkpoint
from vocab_lm.config import ModelConfig, TrainConfig
from vocab_lm.data import TrainingData
from vocab_lm.evaluate import Document
from vocab_lm.model import CausalTransformer, LowRankVocabulary
from vocab_lm.runtime import rng_state
from vocab_lm.tokenizer import FineWebEvaluationConfig
from vocab_lm.train import train, scaling_training_flops


def fixture(kind='dense', device='cpu'):
    model = CausalTransformer(ModelConfig(1, 8, 2, 16, 256, .1), 11)
    if kind == 'low_rank':
        model.vocabulary = LowRankVocabulary(11, 8, 3)
    model.to(device)
    docs = [Document((0, 1, 2, 3, 4, 0), (0, 1, 1, 1, 1, 0), 'c'*64)]
    def data():
        return TrainingData(docs, context=4, stride=2, seed=42,
                            tokenizer={'sha256':'a'*64}, dataset={'sha256':'b'*64,'split':'train'})
    validation = dict(sha256='d'*64, split='validation', protocol=asdict(FineWebEvaluationConfig()))
    return model, data, validation


def test_hand_computable_flops():
    model, _, _ = fixture()
    # B=2,T=3,d=8,F=16,L=1,H=2,V=11. No learned position/LN charged as GEMMs.
    body = 6*6*(4*64+2*8*16)
    attention = 12*2*9*8 + 9*2*2*9
    assert scaling_training_flops(model, 2, 3) == body+attention+6*6*11*8
    model.vocabulary = LowRankVocabulary(11,8,3)
    assert scaling_training_flops(model, 2, 3) == body+attention+6*6*3*(11+16)


@pytest.mark.parametrize('kind', ['dense', 'low_rank'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_budget_exact_resume(kind, device, tmp_path):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    model, data, validation = fixture(kind, device)
    initial = copy.deepcopy(model.state_dict())
    cfg = TrainConfig(context=4,stride=2, accumulation=2,budget_policy='flops',
                      budget_limit=2000000,flop_estimator='vocab_scaling_v1',checkpoint_seconds=1e-9)
    full = train(model,data(),cfg,validation_identity=validation)
    final = copy.deepcopy(model.state_dict())
    model.load_state_dict(initial)
    train(model,data(),cfg,validation_identity=validation,output=tmp_path,max_attempts=2)
    restored, payload = load_training_checkpoint(tmp_path/'last.pt',device)
    resumed = train(restored,data(),cfg,validation_identity=validation,resume=payload['training_state'])
    for name, value in final.items():
        assert torch.equal(value,restored.state_dict()[name])
    for key in ('estimated_flops','raw_bytes','scored_tokens','input_positions','successful_updates','attempted_updates'):
        assert full['counters'][key] == resumed['counters'][key]
    assert full['data'] == resumed['data']
    assert full['counters']['estimated_flops'] <= cfg.budget_limit
    # Rejected group must leave cursor untouched and perform no update.
    tiny = TrainConfig(context=4,stride=2,accumulation=2,budget_policy='flops',budget_limit=1,flop_estimator='vocab_scaling_v1')
    d = data(); cursor = d.state_dict()
    state = train(model,d,tiny,validation_identity=validation)
    assert d.state_dict() == cursor
    assert state['counters']['attempted_updates'] == state['counters']['estimated_flops'] == 0


def test_diagnostics_immutable():
    model, _, _ = fixture('low_rank')
    before = copy.deepcopy(model.state_dict())
    rng = rng_state()['cpu'].clone()
    values = diagnose(model)
    assert values['numerical_rank'] <= 3
    assert torch.equal(rng,rng_state()['cpu'])
    for n,v in before.items():
        assert torch.equal(v, model.state_dict()[n])
    assert all(p.grad is None for p in model.parameters())


def test_storage_maximal_rank():
    from vocab_lm.accounting import vocabulary_storage_report
    s=load_settings('configs/vocab_scaling_toy.yaml')
    m, ceiling=construct(s,'vocab16k')
    assert m.vocabulary.rank == 126
    assert vocabulary_storage_report(m.vocabulary)['vocabulary_serialized_inference_bytes'] <= ceiling
    next_rank=LowRankVocabulary(16384,256,127)
    assert vocabulary_storage_report(next_rank)['vocabulary_serialized_inference_bytes'] > ceiling


def test_graceful_stop_and_schedule_boundary(tmp_path):
    model,data,validation=fixture()
    cfg=TrainConfig(context=4,stride=2,budget_policy='flops',budget_limit=2000000,flop_estimator='vocab_scaling_v1')
    seen=[]
    state=train(model,data(),cfg,validation_identity=validation,output=tmp_path,
                on_boundary=lambda state,m: seen.append(state['counters']['estimated_flops']),
                should_stop=lambda: len(seen)==2)
    assert state['counters']['attempted_updates']==2
    _,p=load_training_checkpoint(tmp_path/'last.pt')
    assert p['training_state']['counters']['estimated_flops']==seen[-1]


def test_report_observations_only(tmp_path):
    import json
    from scripts.vocab_scaling import report
    s=load_settings('configs/vocab_scaling_toy.yaml'); s['output']=str(tmp_path)
    p=tmp_path/'baseline8k'/'run'; p.mkdir(parents=True)
    counters=dict(estimated_flops=100,raw_bytes=20,active_seconds=1.)
    (p/'result.json').write_text(json.dumps(dict(counters=counters,storage={'vocabulary_serialized_inference_bytes':1000},peak_memory={},median_update_seconds=.1,unused_flops=2)))
    (p/'validation.jsonl').write_text(json.dumps(dict(**counters,bpb=3.,subset='final_screen'))+'\n')
    (p/'training.jsonl').write_text(json.dumps(dict(estimated_flops=100,loss=2.))+'\n')
    report(s)
    assert (tmp_path/'curves.png').stat().st_size>1000
    assert '3.000000' in (tmp_path/'report.md').read_text()
    assert 'baseline8k' in (tmp_path/'summary.csv').read_text()


def test_deferred_confirmation_and_initial_screen(tmp_path, monkeypatch):
    import json
    import scripts.vocab_scaling as runner
    _, make_data, identity = fixture()
    s = load_settings('configs/vocab_scaling_toy.yaml')
    s.update(output=str(tmp_path), device='cpu', model=asdict(ModelConfig(1,8,2,16,256,.1)))
    s['training'] = asdict(TrainConfig(context=4,stride=2,budget_policy='flops',
        budget_limit=20000,flop_estimator='vocab_scaling_v1'))
    identity['protocol'] = dict(s['evaluation'])
    s['arms'] = {a:dict(kind='dense',vocab_size=11) for a in ('baseline8k','vocab16k')}
    def prepared(*args):
        d = make_data()
        return d, d.documents, d.documents, identity, torch.zeros(11,dtype=torch.long)
    monkeypatch.setattr(runner,'prepared',prepared)
    runner.run(s,'baseline8k')
    p = tmp_path/'baseline8k'/'run'/'validation.jsonl'
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert rows[0]['subset'] == 'initial_screen'
    assert rows[0]['estimated_flops'] == 0
    assert not any(r['subset']=='confirmation_final' for r in rows)
    with pytest.raises(FileNotFoundError):
        runner.run(s,'baseline8k',operation='confirm')
    runner.run(s,'vocab16k')
    runner.run(s,'baseline8k',operation='confirm')
    with pytest.raises(ValueError,match='already recorded'):
        runner.run(s,'baseline8k',operation='confirm')
