"""Independent execution gates do not depend on historical machine receipts."""
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
from scripts import reproduce as r
from scripts import vocab_scaling as historical
from scripts import vocab_scaling_64k as historical64


@pytest.fixture
def clone(tmp_path,monkeypatch):
    origin=Path.cwd()
    files=list(json.loads(r.LOCK.read_text())['files'])+[str(r.LOCK),'scripts/reproduce.py']
    for name in files:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(origin/name,target)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize('study',['large','64k'])
def test_clean_configuration_without_private_receipts(clone,study):
    assert not Path('results').exists() and not Path('data').exists()
    s=r.configuration(study,'researcher')
    assert s['output']==f'results/reproductions/researcher/{study}'
    assert r.main(['verify','--study',study,'--run','researcher'])['private_historical_receipts_required'] is False
    assert not Path('results').exists()


@pytest.mark.parametrize('field,value',[('budget_limit',1),('microbatch_size',8),('accumulation',4),('learning_rate',.001),('seed',43),('fp16',True)])
def test_changed_scientific_setting_rejected(clone,field,value):
    import yaml
    path=Path('configs/vocab_scaling_64k.yaml'); s=yaml.safe_load(path.read_text());s['training'][field]=value
    path.write_text(yaml.safe_dump(s))
    with pytest.raises(ValueError,match='scientific input changed'):r.configuration('64k','researcher')


@pytest.mark.parametrize('name',['artifacts/vocab_scaling_64k/tokenizer.json','vocab_lm/train.py'])
def test_changed_tokenizer_and_implementation_rejected(clone,name):
    path=Path(name);path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='scientific input changed'):r.verify_science()


@pytest.mark.parametrize('name',['../vocab_scaling_large','/tmp/run','x/y','..',''])
def test_historical_output_paths_cannot_be_selected(clone,name):
    with pytest.raises(ValueError,match='Run name'):r.configuration('large',name)


def test_adapters_do_not_modify_historical_gates(clone):
    import pickle
    a,b=historical.require_large_receipt,historical64.require_authorization
    s=r.configuration('large','researcher');large,extension=r.adapters('large',s)
    assert historical.require_large_receipt is a and historical64.require_authorization is b
    assert large.require_large_receipt is not a and extension.require_authorization is not b
    assert large.concurrent_readers is historical.concurrent_readers
    pickle.dumps(large.concurrent_readers)  # multiprocessing spawn can import it
    with pytest.raises(FileNotFoundError):large.require_large_receipt(s)


@pytest.mark.parametrize('study',['large','64k'])
def test_independent_prepare_uses_pinned_public_inputs(clone,monkeypatch,study):
    s=r.configuration(study,'researcher'); calls=[]
    def public_prepare(root,tokenizers,config):
        assert root.as_posix()=='data/reproductions/common'
        assert config['source']['revision']=='9bb295ddab0e05d785b879661af7260fed5140fc'
        assert config['preserve_selection']['train_documents']==813238
        calls.append('public_pinned_fineweb')
    monkeypatch.setattr(r,'prepare_scaling_data',public_prepare)
    monkeypatch.setattr(r,'verify_data',lambda *args:{})
    monkeypatch.setattr(r.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*2**30))
    extension=SimpleNamespace(prepare_data=lambda config:calls.append('encode_frozen_64k'))
    r.prepare(study,s,extension)
    assert calls==(['public_pinned_fineweb'] if study=='large' else ['public_pinned_fineweb','encode_frozen_64k'])


@pytest.mark.parametrize('study',['large','64k'])
def test_own_preflight_authorization_and_environment_binding(clone,monkeypatch,study):
    s=r.configuration(study,'researcher'); root=Path(s['output'])
    data=Path(s['data']['root']);data.mkdir(parents=True);(data/'preparation.json').write_text('{}')
    allocations=json.loads(Path('artifacts/vocab_scaling_large/protocol.json' if study=='large' else 'artifacts/vocab_scaling_64k/experiment.json').read_text())['allocations']
    seal={'training_counts':{'32768':[1]}} if study=='large' else {'training_counts':[1]}
    monkeypatch.setattr(r,'verify_data',lambda *args:seal)
    env={'gpu_total_bytes':40*2**30,'device':'cuda','machine':'local-test'}
    monkeypatch.setattr(r,'environment',lambda device:dict(env))
    monkeypatch.setattr(r.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(r.torch.cuda,'get_device_properties',lambda i:SimpleNamespace(total_memory=40*2**30))
    monkeypatch.setattr(r,'resume_checks',lambda *args:{a:True for a in s['arms']})
    walk=[dict(budget=5*10**17,supported_with_margin=True)]
    rows=[dict(status='PASS',checkpoint_roundtrip=True,finite_loss=True,finite_gradients=True,initialized_adamw=True,
        microbatch=16,accumulation=2,context=1024,peak_allocated=10,peak_reserved=10,checkpoint_peak_allocated=10,checkpoint_peak_reserved=10) for _ in range(4)]
    large=SimpleNamespace(allocation_table=lambda *args:allocations,budget_projections=lambda *args:walk,
        synthetic_profile=lambda *args:{'rows':rows},large_smoke=lambda *args:{'passed':True},concurrent_readers=lambda *args:{'passed':True})
    extension=SimpleNamespace(allocations=lambda *args:allocations,stream_walks=lambda *args:walk,
        real_smoke=lambda *args:{a:{'finite':True} for a in s['arms']},immutable_json=historical64.immutable_json)
    monkeypatch.setattr(r,'adapters',lambda *args:(large,extension))
    receipt=r.preflight(study,'researcher',s,large,extension)
    assert receipt['status']=='PASS' and not Path('results/vocab_scaling_large').exists()
    assert not Path('results/vocab_scaling_64k').exists()
    with pytest.raises(FileNotFoundError):r.require_local(study,s)
    with pytest.raises(SystemExit):r.main(['authorize','--study',study,'--run','researcher'])
    r.main(['authorize','--study',study,'--run','researcher','--accept-frozen-protocol'])
    assert r.require_local(study,s)['status']=='PASS'
    env['machine']='different'
    with pytest.raises(ValueError,match='environment changed'):r.require_local(study,s)


def test_cpu_preflight_cannot_authorize_training(clone,monkeypatch):
    s=r.configuration('large','cpu');data=Path(s['data']['root']);data.mkdir(parents=True);(data/'preparation.json').write_text('{}')
    monkeypatch.setattr(r,'verify_data',lambda *args:{})
    monkeypatch.setattr(r.torch.cuda,'is_available',lambda:False)
    with pytest.raises(ValueError,match='at least30GiB'):r.preflight('large','cpu',s,None,None)
    assert json.loads((Path(s['output'])/'reproduction_preflight.json').read_text())['status']=='FAIL'


def test_confirmation_refused_before_entering_runner(clone):
    s=r.configuration('64k','researcher');folder=Path(s['output'])/'hash64k/run';folder.mkdir(parents=True)
    path=folder/'validation.jsonl';path.write_text(json.dumps({'subset':'confirmation_final'})+'\n');before=path.read_bytes()
    with pytest.raises(ValueError,match='once-only confirmation'):
        r.main(['confirm','--study','64k','--run','researcher','--arm','hash64k'])
    assert path.read_bytes()==before


def test_prepared_data_corruption_rejected_without_historical_results(clone,monkeypatch):
    s=r.configuration('large','researcher');folder=Path(s['data']['root']);folder.mkdir(parents=True)
    shard=folder/'tokens.bin';shard.write_bytes(b'exact scientific tokens')
    seal=folder/'preparation.json';seal.write_text('{}')
    expected=dict(dataset_hashes={'tokens.bin':r.file_hash(shard)},preparation_sha256=r.file_hash(seal))
    original=r.read_json
    monkeypatch.setattr(r,'read_json',lambda p:expected if str(p)=='artifacts/vocab_scaling_large/protocol.json' else original(p))
    assert r.verify_data('large',s)=={}
    shard.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='common data differ'):r.verify_data('large',s)
