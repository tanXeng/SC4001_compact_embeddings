"""Additive-extension boundaries, accounting and diagnostic correctness."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import vocab_scaling_64k as extension
from scripts.vocab_scaling_64k_diagnostics import effective_path_gradients, interference, shared_pairs
from vocab_lm.config import ModelConfig, TrainConfig
from vocab_lm.data import TrainingData, token_dtype
from vocab_lm.evaluate import Document
from vocab_lm.model import AdaptiveVocabulary, CausalTransformer, HashVocabulary, LowRankVocabulary
from vocab_lm.train import scaling_training_flops


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_frozen_recipe_and_exactly_four_arms():
    s=extension.settings('configs/vocab_scaling_64k.yaml')
    assert s['training']['budget_limit']==500000000000000000
    assert ModelConfig(**s['model']).backbone_parameters==32033792
    assert s['reference']['ceiling_bytes']==33555740
    assert s['protocol']['phase']=='preparation_only'
    assert len(s['arms'])==4 and 'baseline16k' not in s['arms']
    assert token_dtype(65536)==np.dtype('<u2')


@pytest.mark.parametrize('kind',['dense','low_rank','adaptive','hash'])
def test_hand_computable_v1_formulas(kind):
    model=CausalTransformer(ModelConfig(1,8,2,16,1024,0.),11)
    if kind=='low_rank': model.vocabulary=LowRankVocabulary(11,8,3)
    elif kind=='adaptive': model.vocabulary=AdaptiveVocabulary(torch.arange(11),8,[3,6],factor=2.)
    elif kind=='hash': model.vocabulary=HashVocabulary(11,8,5,seeds=[17,29])
    body=6*6*(4*64+2*8*16)
    attention=12*2*9*8+9*2*2*9
    expected={'dense':6*6*11*8,'low_rank':6*6*3*(11+16),
              'hash':6*6*11*8+(6+11)*8*9,
              'adaptive':6*6*(8*(3+8)+4*(3+8)+2*(5+8))+6*6*8*8}[kind]
    assert scaling_training_flops(model,2,3)==body+attention+expected


def test_real_serialized_maximality_and_adaptive_neighbours():
    s=extension.settings('configs/vocab_scaling_64k.yaml')
    counts=torch.arange(65536,dtype=torch.int64)
    table=extension.allocations(s,counts)
    ceiling=s['reference']['ceiling_bytes']
    for a in ('lowrank64k','hash64k'):
        fit,fail=table[a]['nearby_allocations']
        assert fit['vocabulary_serialized_inference_bytes']<=ceiling<fail['vocabulary_serialized_inference_bytes']
        assert fail['allocation']==fit['allocation']+1
    assert table['lowrank64k']['specification']['rank']==127
    assert table['hash64k']['specification']['buckets']==15615
    assert table['hash64k']['coefficient_bytes']==65536*2*4
    assert table['hash64k']['vocabulary_integer_buffer_bytes']==65536*2*8
    adaptive=table['adaptive64k']
    assert .95<=adaptive['utilization']<=1
    below,selected,above=adaptive['nearby_allocations']
    assert below['archive_bytes']<selected['archive_bytes']<=ceiling<above['archive_bytes']
    assert selected['dimensions']==[512,128,32]
    assert selected['cutoffs'][1]-selected['cutoffs'][0]==65536//4
    assert sum(adaptive['band_training_mass'])==int(counts.sum())
    # Changing counts changes ordering/mass, never the storage-based allocation.
    other,_=extension.construct(s,'adaptive64k',counts.flip(0))
    assert list(other.vocabulary.cutoffs[:-1])==adaptive['specification']['cutoffs']


def test_evidence_drift_stops_without_overwrite(tmp_path):
    evidence=tmp_path/'completed.json'; protected=tmp_path/'old_result.json'
    protected.write_text('scientific evidence')
    extension.immutable_json(evidence,dict(files={str(protected):dict(bytes=protected.stat().st_size,sha256=extension.file_hash(protected))}))
    assert extension.verify_evidence(evidence)['status']=='PASS'
    with pytest.raises(FileExistsError): extension.immutable_json(evidence,{})
    protected.write_text('changed')
    with pytest.raises(RuntimeError,match='STOP'): extension.verify_evidence(evidence)


def test_tokenizer_identity_and_literal_eos():
    s=extension.settings('configs/vocab_scaling_64k.yaml'); pair=extension.tokenizer_pair(s)
    p=extension.read_json(extension.ARTIFACTS/'protocol.json')
    assert extension.file_hash(Path(s['tokenizer']['artifact']))==p['artifact_sha256']
    assert pair.expanded.get_vocab_size()==65536
    old=json.loads(Path('artifacts/vocab_scaling_large/tokenizers/expanded.json').read_text(encoding='utf-8'))
    new=json.loads(pair.expanded.to_str())
    assert {k:v for k,v in new['model']['vocab'].items() if v<32768}==old['model']['vocab']
    assert new['model']['merges'][:len(old['model']['merges'])]==old['model']['merges']
    for raw in [b'',b'<|endoftext|>',b'a\x00\r\n','é 中文 🐈'.encode()]:
        ids=pair.encode_document(raw,expanded=True)
        assert ids[-1]==0 and 0 not in ids[:-1]
        assert pair.reconstruct(ids,expanded=True)==raw


def test_stream_walk_matches_shared_loader_exactly(monkeypatch,tmp_path):
    # Variable document lengths, padded shapes, overlap, EOS zero bytes and
    # incomplete final updates exercise the independent metadata-only walk.
    lengths=[17,699,2000,1025,2049,512,514]*11
    arrays=[]; docs=[]; byte_table=[0,1,3,4]
    for i,n in enumerate(lengths):
        ids=np.asarray([0]+[1+j%3 for j in range(n)]+[0],dtype=np.uint16)
        arrays.append(ids)
        docs.append(Document(tuple(map(int,ids)),tuple(byte_table[int(x)] for x in ids),f'{i:064x}'))
    fake=SimpleNamespace(locations=[(i,0,len(x)) for i,x in enumerate(arrays)],maps=arrays,metadata=dict(byte_lengths=byte_table))
    class Stream:
        def __len__(self): return len(arrays)
        def __getattr__(self,name): return getattr(fake,name)
    model=CausalTransformer(ModelConfig(1,8,2,16,1024,0.),4)
    monkeypatch.setattr(extension,'ShardedDocuments',lambda p:Stream())
    monkeypatch.setattr(extension,'construct',lambda *args:(model,[]))
    s=dict(data=dict(root=str(tmp_path)),output=str(tmp_path),arms={'dense64k':{}},training={})
    rows=extension.stream_walks(s,dict(training_counts=[1,2,3,4]),{})
    data=TrainingData(docs,context=1024,stride=512,seed=42,tokenizer=dict(sha256='a'*64),dataset=dict(sha256='b'*64,split='train'),non_repeating=True)
    counters=dict(updates=0,estimated_flops=0,scored_tokens=0,input_positions=0,raw_bytes=0)
    while True:
        try: batches=[data.batch(16) for _ in range(2)]
        except RuntimeError: break
        counters['updates']+=1
        for b in batches:
            counters['estimated_flops']+=scaling_training_flops(model,*b['inputs'].shape)
            counters['scored_tokens']+=b['tokens']; counters['input_positions']+=b['inputs'].numel(); counters['raw_bytes']+=b['bytes']
    for row in rows:
        assert not row['supported']
        assert all(row[k]==v for k,v in counters.items())


def test_hash_diagnostic_paths_preserve_training_state():
    model=CausalTransformer(ModelConfig(1,8,2,16,32,.1),17)
    model.vocabulary=HashVocabulary(17,8,3)
    doc=Document(tuple([0]+list(range(1,17))+[0]),tuple([0]+[1]*16+[0]),'c'*64)
    original=copy.deepcopy(model.state_dict())
    for p in model.parameters(): p.grad=torch.ones_like(p)
    policy=dict(documents=1,targets=32,shared_buckets=3,tokens_per_bucket=4)
    result=interference(model,[doc],policy)
    assert set(result['paths'])=={'input','output','combined'}
    assert result['selected_buckets']==3
    for row in result['paths'].values():
        assert row['mean_cancellation'] is not None
        assert row['pairwise_cosine'] is not None
        assert -.00001<=row['mean_cancellation']<=1.00001
    assert model.training
    assert all(torch.equal(t,model.state_dict()[k]) for k,t in original.items())
    assert all(torch.equal(p.grad,torch.ones_like(p)) for p in model.parameters())
    # Known example: three token pairs share a component, one is a full code.
    assert shared_pairs(torch.tensor([[0,1],[1,0],[1,2]]))==dict(any_shared_component_token_pairs=3,partial_only_token_pairs=2)


def test_training_is_not_authorized(tmp_path,monkeypatch):
    s=extension.settings('configs/vocab_scaling_64k.yaml'); s['output']=str(tmp_path)
    with pytest.raises(FileNotFoundError): extension.require_authorization(s)
    assert not (tmp_path/'training_authorization.json').exists()


def test_hash_attribution_matches_native_actual_lm_gradient():
    model=CausalTransformer(ModelConfig(1,8,2,16,32,0.),17)
    model.vocabulary=HashVocabulary(17,8,3); model.eval()
    doc=Document(tuple([0]+list(range(1,17))+[0]),tuple([0]+[1]*16+[0]),'c'*64)
    policy=dict(documents=1,targets=32)
    gin,gout,_,_=effective_path_gradients(model,[doc],policy)
    v=model.vocabulary
    chain=torch.zeros_like(v.components)
    for token in range(v.vocab_size):
        for j in range(2): chain[v.indices[token,j]]+=(gin[token]+gout[token])*v.importance[token,j].detach()
    logits=model(torch.tensor([doc.ids[:-1]]))[0]
    loss=torch.nn.functional.cross_entropy(logits,torch.tensor(doc.ids[1:]))
    native,=torch.autograd.grad(loss,v.components)
    torch.testing.assert_close(chain,native,atol=1e-6,rtol=1e-5)
    assert all(p.grad is None for p in model.parameters())


def test_extension_summary_never_rewrites_historical_results(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    old=Path('results/vocab_scaling_large/baseline16k/run'); old.mkdir(parents=True)
    result=old/'result.json'; result.write_text(json.dumps(dict(completion='budget_boundary',storage={},counters={})))
    (old/'validation.jsonl').write_text(json.dumps(dict(subset='final_screen',bpb=1.234,nll=123.,token_nll=2.,tokens=61,bytes=72))+'\n')
    historical_hash=extension.file_hash(result)
    root=Path('results/vocab_scaling_64k'); root.mkdir(parents=True)
    extension.immutable_json(root/'allocations.json',{a:{} for a in extension.ARM_KINDS})
    rows=extension.report(dict(output=str(root),arms=extension.ARM_KINDS))
    assert extension.file_hash(result)==historical_hash
    assert rows[0]['final_bpb']==1.234
    assert all(r['status']=='not_started' and r['final_bpb'] is None for r in rows[4:])
    assert rows[0]['provenance']!=rows[-1]['provenance']
    assert (root/'summary.csv').exists()
