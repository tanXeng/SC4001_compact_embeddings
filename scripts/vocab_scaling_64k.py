"""Additive 64K orchestration. The e579a46 runner and vocab_lm remain untouched.

All optimization, evaluation, checkpointing and FLOP/storage arithmetic use the
existing package. This file owns only extension inputs, receipts and scheduling.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from tokenizers import Tokenizer
from scripts.vocab_scaling import adaptive_allocation, diagnose, gpu_memory
from vocab_lm.accounting import parameter_counts, storage_report, tensor_bytes, vocabulary_storage_report
from vocab_lm.checkpoint import load_training_checkpoint, save_training_checkpoint
from vocab_lm.config import ModelConfig, TrainConfig, evaluation_config
from vocab_lm.data import ShardedDocuments, TrainingData, write_shards
from vocab_lm.evaluate import Document, evaluate
from vocab_lm.fineweb import RawDocuments, dataset_identity, selection_provenance
from vocab_lm.model import CausalTransformer, DenseVocabulary, HashVocabulary, LowRankVocabulary
from vocab_lm.runtime import environment, file_hash, read_json, restore_rng, rng_state, seed_all, sha256, source_identity, write_json
from vocab_lm.tokenizer import NestedBPE, train_nested
from vocab_lm.train import accumulated_backward, optimizer_update, scaling_training_flops, train

ARTIFACTS = Path('artifacts/vocab_scaling_64k')
ARM_KINDS = dict(dense64k='dense', lowrank64k='low_rank', adaptive64k='adaptive', hash64k='hash')


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def verify_evidence(path):
    manifest = read_json(path)
    changed = [name for name, r in manifest['files'].items()
               if not Path(name).is_file() or Path(name).stat().st_size != r['bytes']
               or file_hash(Path(name)) != r['sha256']]
    if changed:
        raise RuntimeError('STOP: completed evidence changed: ' + ', '.join(changed))
    return dict(status='PASS', manifest=str(path), manifest_sha256=file_hash(Path(path)),
                files=len(manifest['files']), bytes=sum(r['bytes'] for r in manifest['files'].values()))


def settings(path):
    s = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    old = yaml.safe_load(Path('configs/vocab_scaling_large.yaml').read_text(encoding='utf-8'))
    for key in ('model', 'training', 'evaluation', 'evaluation_fractions', 'diagnostic_fractions',
                'screen_documents', 'confirmation_documents', 'threads'):
        if s[key] != old[key]:
            raise ValueError('64K must preserve completed recipe: ' + key)
    if {k: v['kind'] for k, v in s['arms'].items()} != ARM_KINDS or any(
            a['vocab_size'] != 65536 for a in s['arms'].values()):
        raise ValueError('Exactly four declared 65536-token arms required')
    if s['arms']['hash64k'] != dict(kind='hash', vocab_size=65536, k=2, seeds=[17, 29], importance_init='normal'):
        raise ValueError('Hash baseline must retain k=2 and historical seeds/initialization')
    if s['arms']['adaptive64k'] != dict(kind='adaptive',vocab_size=65536,allocation_policy='three_band_factor4_middle_quarter_grid256_v1'):
        raise ValueError('Adaptive must retain the established deterministic allocation policy')
    if s['protocol']['phase']!='preparation_only':
        raise ValueError('This extension config must remain preparation_only; later owner authorization is a separate bound receipt')
    if s['output'] != 'results/vocab_scaling_64k' or s['data']['root'] != 'data/vocab_scaling_64k':
        raise ValueError('Extension outputs must be separate')
    ceiling = read_json('artifacts/vocab_scaling_large/protocol.json')['allocations']['baseline16k']['vocabulary_serialized_inference_bytes']
    if s['reference']['ceiling_bytes'] != ceiling or ceiling != 33555740:
        raise ValueError('Frozen baseline ceiling mismatch')
    ModelConfig(**s['model']); TrainConfig(**s['training']); evaluation_config(s['evaluation'])
    return s


def tokenizer_pair(s):
    expanded = Tokenizer.from_file(s['tokenizer']['artifact'])
    payload = json.loads(expanded.to_str())
    # A transient byte-alphabet prefix adapts the existing generic shard writer;
    # no historical tokenizer or second on-disk tokenizer is created.
    payload['model']['vocab'] = {t: i for t, i in payload['model']['vocab'].items() if i < 257}
    payload['model']['merges'] = []
    pair = NestedBPE(Tokenizer.from_str(json.dumps(payload)), expanded, '<|endoftext|>')
    if expanded.get_vocab_size() != 65536:
        raise ValueError('64K tokenizer has incorrect size')
    return pair


def prepare_tokenizer(s, sample_root):
    old = read_json('artifacts/vocab_scaling_large/tokenizers/protocol.json')
    root = Path(sample_root); selection = read_json(root/'selection.json')
    records = selection['train'][:old['selection']['documents']]
    sample = [{k: d[k] for k in ('source_id', 'raw_sha256', 'raw_bytes', 'selection_order')} for d in records]
    digest = sha256((json.dumps(sample, ensure_ascii=False, sort_keys=True, indent=2)+'\n').replace('\n', '\r\n').encode())
    if digest != old['selection']['ordered_sample_sha256'] or sum(r['raw_bytes'] for r in records) != 200000346:
        raise ValueError('Not the frozen tokenizer construction prefix')
    if {r['raw_sha256'] for r in records} & {r['raw_sha256'] for r in selection['validation']}:
        raise ValueError('Tokenizer train/validation overlap')
    destination = Path(s['tokenizer']['artifact']); protocol = destination.parent/'protocol.json'
    if destination.exists() or protocol.exists():
        p = read_json(protocol)
        if file_hash(destination) != p['artifact_sha256'] or p['ordered_sample_sha256'] != digest:
            raise ValueError('Frozen 64K tokenizer changed/incomplete')
        return p
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    cfg = dict(old['config']); cfg.pop('parallelism'); cfg['expanded_vocab_size'] = 65536
    raws = RawDocuments(root, records)
    first = train_nested(iter(raws), **cfg)
    replay = train_nested(iter(raws), **cfg)
    if json.loads(first.expanded.to_str()) != json.loads(replay.expanded.to_str()):
        raise ValueError('Independent tokenizer replay differs')
    for raw in (b'', b'<|endoftext|>', 'a\x00\r\n\t café 中文 🐈'.encode()):
        ids = first.encode_document(raw, expanded=True)
        if ids[-1] != 0 or 0 in ids[:-1] or first.reconstruct(ids, expanded=True) != raw:
            raise ValueError('64K EOS/byte reconstruction failed')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8', newline='\n') as f:
        f.write(first.expanded.to_str(pretty=True))
    p = dict(protocol='fineweb_train_bpe64k_v1', artifact_sha256=file_hash(destination),
             vocab_size=65536, eos_id=0, literal_eos_is_text=True, config=cfg,
             ordered_sample_sha256=digest, documents=len(records), raw_bytes=200000346,
             provenance=old, construction='Two independent parsed-identical fits; no LM validation; training-only original prefix',
             tokenizers_version=__import__('tokenizers').__version__)
    immutable_json(protocol, p)
    return p


def construct(s, arm, counts):
    seed_all(42)
    c = ModelConfig(**s['model']); a = s['arms'][arm]; ceiling = s['reference']['ceiling_bytes']
    model = CausalTransformer(c, 65536 if a['kind'] == 'dense' else 1)
    nearby = []
    if a['kind'] in ('low_rank', 'hash'):
        factory = (lambda n: LowRankVocabulary(65536, c.hidden, n)) if a['kind'] == 'low_rank' else (
            lambda n: HashVocabulary(65536, c.hidden, n, seeds=a['seeds'], importance_init=a['importance_init']))
        lo, hi = 0, c.hidden if a['kind'] == 'low_rank' else ceiling//(4*c.hidden)
        while lo < hi:
            mid = (lo+hi+1)//2
            if vocabulary_storage_report(factory(mid))['vocabulary_serialized_inference_bytes'] <= ceiling:
                lo = mid
            else:
                hi = mid-1
        if lo < 1:
            raise ValueError('No compressed allocation fits')
        for n in (lo, lo+1):
            nearby.append(dict(allocation=n, **vocabulary_storage_report(factory(n))))
        seed_all(42); model.vocabulary = factory(lo)
    elif a['kind'] == 'adaptive':
        if counts is None or len(counts) != 65536:
            raise ValueError('Adaptive requires complete training-only counts')
        model.vocabulary, nearby = adaptive_allocation(counts, c.hidden, ceiling)
    r = vocabulary_storage_report(model.vocabulary)
    if a['kind'] != 'dense' and r['vocabulary_serialized_inference_bytes'] > ceiling:
        raise ValueError('Compressed archive exceeds frozen ceiling')
    if a['kind'] == 'adaptive' and r['vocabulary_serialized_inference_bytes']/ceiling < .95:
        raise ValueError('Adaptive utilization below 95%')
    return model, nearby


def allocations(s, counts=None):
    rows = {}
    for arm in s['arms']:
        if arm == 'adaptive64k' and counts is None:
            rows[arm] = dict(status='PENDING', reason='Requires complete 64K training frequencies')
            continue
        model, nearby = construct(s, arm, counts)
        v = model.vocabulary; r = vocabulary_storage_report(v); p = parameter_counts(model)
        r.update(**p, specification=v.specification(), nearby_allocations=nearby,
                 ceiling_bytes=s['reference']['ceiling_bytes'],
                 utilization=r['vocabulary_serialized_inference_bytes']/s['reference']['ceiling_bytes'],
                 archive_overhead_bytes=r['vocabulary_serialized_inference_bytes']-r['vocabulary_parameter_bytes']-r['vocabulary_fixed_buffer_bytes'],
                 full_group_flops=2*scaling_training_flops(model,16,1024),
                 status='PASS', outside_storage_budget=arm=='dense64k')
        r['full_context_updates_lower_bound'] = 500000000000000000//r['full_group_flops']
        if arm == 'hash64k':
            r.update(component_bytes=tensor_bytes(v.components), coefficient_bytes=tensor_bytes(v.importance),
                     indices_sha256=sha256(v.indices.numpy().tobytes()))
        if arm == 'adaptive64k':
            r.update(counts_sha256=sha256(counts.numpy().tobytes()), order_sha256=sha256(v.order.numpy().tobytes()),
                     band_training_mass=[int(counts[v.order[l:h]].sum()) for l,h in zip((0,)+v.cutoffs[:-1],v.cutoffs)])
        rows[arm] = r
    return rows


def prepare_data(s):
    source = Path(s['data']['raw_root']); frozen = read_json('artifacts/vocab_scaling_large/protocol.json')
    for name in ('raw.bin','selection.json','source_manifest.json'):
        if file_hash(source/name) != frozen['dataset_hashes'][name]:
            raise ValueError('Completed raw-document identity mismatch: '+name)
    selection = read_json(source/'selection.json'); root = Path(s['data']['root'])
    for key in ('raw_sha256','source_id'):
        if {d[key] for d in selection['train']} & {d[key] for d in selection['validation']}:
            raise ValueError('Train/validation overlap')
    pair = tokenizer_pair(s)
    tok = dict(sha256=file_hash(Path(s['tokenizer']['artifact'])), vocab_size=65536, artifact='tokenizer.json')
    coverage = {}; counts = None; hashes = {}
    for split in ('train','validation'):
        folder = root/'shards/65536'/split
        identity = dataset_identity(source,split)
        if not folder.exists():
            write_shards(folder, RawDocuments(source,selection[split]), selection[split], pair, expanded=True,
                         tokenizer=tok, dataset=identity, max_tokens=2000000, encode_batch_size=128)
        docs = ShardedDocuments(folder, tokenizer_sha256=tok['sha256'], dataset_sha256=identity['sha256'])
        if len(docs) != len(selection[split]):
            raise ValueError('Incomplete 64K shard generation')
        pieces = [pair.token_bytes(i) for i in range(65536)]
        freq = np.zeros(65536,dtype=np.int64); raw_bytes = 0
        for d,r in zip(docs,selection[split]):
            if d.raw_sha256 != r['raw_sha256'] or sha256(b''.join(pieces[i] for i in d.ids)) != r['raw_sha256']:
                raise ValueError('64K reconstruction/order mismatch')
            raw_bytes += sum(d.byte_lengths); freq += np.bincount(d.ids[1:],minlength=65536)
        coverage[split] = dict(documents=len(docs),raw_bytes=raw_bytes,scored_tokens=int(freq.sum()))
        if split == 'train': counts = freq.tolist()
        hashes[(folder/'index.json').relative_to(root).as_posix()] = file_hash(folder/'index.json')
        for shard in docs.metadata['shards']:
            hashes[(folder/shard['file']).relative_to(root).as_posix()] = shard['sha256']
    seal = dict(protocol='fineweb_64k_extension_v1',source_root=str(source), tokenizer=tok,
                raw_identities={n:frozen['dataset_hashes'][n] for n in ('raw.bin','selection.json','source_manifest.json')},
                ordered_selection=selection_provenance(selection,len(selection['train'])),coverage=coverage,
                training_counts=counts, hashes=hashes,screen_hashes=[r['raw_sha256'] for r in selection['validation'][:256]],
                confirmation_hashes=[r['raw_sha256'] for r in selection['validation'][256:512]])
    target=root/'preparation.json'
    if target.exists():
        if read_json(target) != seal: raise ValueError('64K preparation drift')
    else: immutable_json(target,seal)
    return seal


def inputs(s):
    seal=read_json(Path(s['data']['root'])/'preparation.json')
    if file_hash(Path(s['tokenizer']['artifact'])) != seal['tokenizer']['sha256']:
        raise ValueError('64K tokenizer drift')
    for n,h in seal['raw_identities'].items():
        if file_hash(Path(s['data']['raw_root'])/n) != h:
            raise ValueError('Completed raw source changed: '+n)
    for n,h in seal['hashes'].items():
        path=(Path(s['data']['root'])/n).resolve()
        if not path.is_relative_to(Path(s['data']['root']).resolve()) or file_hash(path) != h:
            raise ValueError('64K data changed: '+n)
    return seal


def stream_walks(s, seal, table):
    """Exact ordered token-byte walk of complete updates; no model execution."""
    stream=ShardedDocuments(Path(s['data']['root'])/'shards/65536/train')
    order=list(range(len(stream))); random.Random(42).shuffle(order)
    byte_table=np.asarray(stream.metadata['byte_lengths'],dtype=np.int64)
    totals={a:dict(updates=0,estimated_flops=0,scored_tokens=0,input_positions=0,raw_bytes=0) for a in s['arms']}
    boundaries={a:{} for a in s['arms']}; budgets=(500000000000000000,600000000000000000)
    # Shape formulas depend only on the already-measured fixed representation.
    models={a:construct(s,a,torch.tensor(seal['training_counts'],dtype=torch.int64))[0] for a in s['arms']}
    costs={}; batch=[]; group=[]
    for index in order:
        shard,left,right=stream.locations[index]; ids=stream.maps[shard][left:right]
        prefix=np.concatenate((np.zeros(1,dtype=np.int64),np.cumsum(byte_table[ids])))
        for first in range(1,len(ids),512):
            end=min(first+511,len(ids)-1)
            batch.append((min(1024,end),end-first+1,int(prefix[end+1]-prefix[first])))
            if len(batch)<16: continue
            group.append((max(x[0] for x in batch),sum(x[1] for x in batch),sum(x[2] for x in batch))); batch=[]
            if len(group)<2: continue
            for a,model in models.items():
                cost=0
                for length,_,_ in group:
                    key=(a,length)
                    if key not in costs: costs[key]=scaling_training_flops(model,16,length)
                    cost+=costs[key]
                c=totals[a]
                for budget in budgets:
                    if budget not in boundaries[a] and c['estimated_flops']+cost>budget:
                        boundaries[a][budget]=dict(c,supported=True,next_update_flops=cost)
                c['updates']+=1; c['estimated_flops']+=cost
                c['scored_tokens']+=sum(x[1] for x in group); c['raw_bytes']+=sum(x[2] for x in group)
                c['input_positions']+=sum(16*x[0] for x in group)
            group=[]
        if all(len(v)==2 for v in boundaries.values()): break
    rows=[]
    for a in s['arms']:
        for budget in budgets:
            row=boundaries[a].get(budget,dict(totals[a],supported=False,next_update_flops=None))
            rows.append(dict(arm=a,budget=budget,**row,unused_flops=budget-row['estimated_flops'],
                             supported_with_margin=(600000000000000000 in boundaries[a]) if budget==500000000000000000 else None,
                             observed_stream_flops_lower_bound=totals[a]['estimated_flops'],raw_bytes_exact=True))
    write_json(Path(s['output'])/'data_walks.json',rows)
    return rows


def diagnostic(s, model, counts, documents, endpoint=False):
    from scripts.vocab_scaling_64k_diagnostics import hash_structure, shared_pairs, interference
    saved=rng_state(); previous=model.training
    try:
        result=diagnose(model)
        if hasattr(model.vocabulary,'order'):
            v=model.vocabulary
            result['band_training_mass']=[int(counts[v.order.detach().cpu()[l:h]].sum())
                for l,h in zip((0,)+v.cutoffs[:-1],v.cutoffs)]
        if isinstance(model.vocabulary,HashVocabulary):
            result['hash']=hash_structure(model.vocabulary,counts)
            result['hash'].update(shared_pairs(model.vocabulary.indices.cpu()))
            if endpoint:
                with torch.enable_grad():
                    result['hash']['interference']=interference(model,documents,s['protocol']['hash_interference'])
        return result
    finally:
        restore_rng(saved); model.train(previous)


def quality_groups(model,counts):
    groups=torch.empty_like(counts)
    order=sorted(range(len(counts)),key=lambda i:(-int(counts[i]),i))
    groups[torch.tensor(order)]=torch.arange(len(counts))*3//len(counts)
    if hasattr(model.vocabulary,'inverse'):
        v=model.vocabulary
        groups=torch.bucketize(v.inverse.cpu(),torch.tensor(v.cutoffs[:-1]),right=True)
    return groups


def profile(s, seal, arms=None):
    """Two warmups + three timed complete updates; no LM validation selection."""
    device=torch.device(s['device'])
    if device.type!='cuda' or 'A40' not in torch.cuda.get_device_name(device):
        raise ValueError('Resource profiling requires the actual A40')
    root=Path(s['output'])/'profile'; root.mkdir(parents=True,exist_ok=True)
    receipt=dict(identity=identity(s),environment=environment(device),rows=[])
    for arm in arms or s['arms']:
        gc.collect(); torch.cuda.empty_cache()
        counts=torch.tensor(seal['training_counts'],dtype=torch.int64)
        model,_=construct(s,arm,counts); model.to(device)
        t=TrainConfig(**s['training'])
        opt=torch.optim.AdamW(model.parameters(),lr=t.learning_rate,betas=tuple(t.betas),eps=t.epsilon,
                             weight_decay=t.weight_decay,foreach=False,fused=False)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        seed_all(42)
        batches=[dict(inputs=torch.randint(65536,(16,1024)),targets=torch.randint(65536,(16,1024)),tokens=16384) for _ in range(2)]
        samples=[]; restored=payload=None
        row=dict(arm=arm,status='FAIL',context=1024,microbatch=16,accumulation=2)
        try:
            torch.cuda.reset_peak_memory_stats(device)
            for step in range(5):
                torch.cuda.synchronize(device); started=time.perf_counter()
                opt.zero_grad(set_to_none=True)
                nll=accumulated_backward(model,batches,device=device,scaler=scaler,fp16=False)
                optimizer_update(model,opt,scaler,t.clip_norm)
                torch.cuda.synchronize(device)
                if step>=2: samples.append(time.perf_counter()-started)
                if not np.isfinite(nll) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite profile loss/gradients')
            seconds=statistics.median(samples); flops=2*scaling_training_flops(model,16,1024)
            row.update(status='PASS',finite_loss=True,finite_gradients=True,initialized_adamw=bool(opt.state),
                       warmup_updates=2,timed_updates=3,samples_seconds=samples,update_seconds=seconds,
                       tokens_per_second=32768/seconds,estimated_tflops_per_second=flops/seconds/1e12,
                       flops_per_update=flops,**gpu_memory(device),optimizer_state_tensor_bytes=tensor_bytes(opt.state_dict()))
            probe=root/(arm+'.pt'); started=time.perf_counter()
            save_training_checkpoint(probe,model,dict(optimizer=opt.state_dict()),tokenizer=seal['tokenizer'],
                                     dataset=dict(sha256='b'*64,split='validation',protocol=s['evaluation']))
            row.update(checkpoint_save_seconds=time.perf_counter()-started,checkpoint_bytes=probe.stat().st_size)
            started=time.perf_counter(); restored,payload=load_training_checkpoint(probe,str(device))
            torch.cuda.synchronize(device)
            row.update(checkpoint_load_seconds=time.perf_counter()-started,
                       checkpoint_peak_allocated=torch.cuda.max_memory_allocated(device),
                       checkpoint_peak_reserved=torch.cuda.max_memory_reserved(device),
                       checkpoint_roundtrip=all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items()))
            if not row['checkpoint_roundtrip']: raise ValueError('Checkpoint roundtrip differs')
            row['storage']=storage_report(model,optimizer_state=opt.state_dict(),tokenizer_paths=[s['tokenizer']['artifact']],
                                         tokenizer=seal['tokenizer'],dataset=dict(sha256='b'*64,split='validation',protocol=s['evaluation']))
            row['over_24gib_target']=max(row['checkpoint_peak_allocated'],row['checkpoint_peak_reserved'])>24*2**30
            comfortable=min(38*2**30,int(torch.cuda.get_device_properties(device).total_memory*.9))
            if max(row['checkpoint_peak_allocated'],row['checkpoint_peak_reserved'])>comfortable:
                raise ValueError('Common batch does not fit with comfortable A40 headroom; owner review required')
        except (torch.cuda.OutOfMemoryError,FloatingPointError,ValueError) as error:
            row.update(status='FAIL',reason=str(error))
        receipt['rows'].append(row)
        write_json(Path(s['output'])/'profiles.json',receipt)
        print(json.dumps({k:v for k,v in row.items() if k!='storage'}),flush=True)
        del model,opt,scaler,batches,restored,payload
    return receipt


def identity(s):
    return dict(source=source_identity(),config_sha256=sha256(json.dumps(s,sort_keys=True).encode()),
                tokenizer_sha256=file_hash(Path(s['tokenizer']['artifact'])),
                preparation_sha256=file_hash(Path(s['data']['root'])/'preparation.json'))


def tree_digest(tree):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            h.update(str((x.dtype,tuple(x.shape))).encode()); h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str): h.update(str(k).encode()); visit(x[k])
        elif isinstance(x,(tuple,list)):
            for v in x: visit(v)
        else: h.update(repr(x).encode())
    visit(tree); return h.hexdigest()


def resume_worker(s,arm,stage):
    seal=read_json(Path(s['data']['root'])/'preparation.json')
    root=Path(s['output'])/'resume'/arm; root.mkdir(parents=True,exist_ok=True)
    model,_=construct(s,arm,torch.tensor(seal['training_counts'],dtype=torch.int64)); model.to(s['device'])
    docs=[Document((0,1,2,3,4,0),(0,1,1,1,1,0),'c'*64)]
    data=TrainingData(docs,context=4,stride=2,seed=42,tokenizer=seal['tokenizer'],dataset=dict(sha256='b'*64,split='train'))
    validation=dict(sha256='d'*64,split='validation',protocol=s['evaluation'])
    cfg=dict(s['training']); cfg.update(context=4,stride=2,microbatch_size=1,accumulation=1,
                                      budget_limit=10**18,checkpoint_seconds=0)
    payload=None
    if stage=='resumed': model,payload=load_training_checkpoint(root/'last.pt',s['device'])
    previous=payload['training_state']['counters']['estimated_flops'] if payload else 0
    events=dict(evaluation=[],diagnostics=[])
    unit=scaling_training_flops(model,1,2)
    points=dict(evaluation=[unit//2,2*unit],diagnostics=[unit,3*unit])
    def boundary(state,model):
        nonlocal previous
        current=state['counters']['estimated_flops']
        for k,thresholds in points.items(): events[k].extend(p for p in thresholds if previous<p<=current)
        previous=current
    state=train(model,data,TrainConfig(**cfg),validation_identity=validation,
                resume=payload['training_state'] if payload else None,max_attempts=2 if stage=='full' else 1,
                on_boundary=boundary,experiment_identity=identity(s))
    if stage=='split': save_training_checkpoint(root/'last.pt',model,state,tokenizer=seal['tokenizer'],dataset=validation)
    result={k:tree_digest(v) for k,v in dict(model=model.state_dict(),optimizer=state['optimizer'],scheduler=state['scheduler'],
                                           rng=state['rng'],data=state['data']).items()}
    result.update(counters={k:v for k,v in state['counters'].items() if k not in ('wall_seconds','active_seconds')},events=events)
    write_json(root/(stage+'.json'),result)
    return result


def resume_check(s,config_path):
    results={}
    for arm in s['arms']:
        for stage in ('full','split','resumed'):
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'resume-worker','--config',str(config_path),
                            '--arm',arm,'--stage',stage],check=True,stdout=subprocess.DEVNULL)
        root=Path(s['output'])/'resume'/arm
        full,split,resumed=[read_json(root/(x+'.json')) for x in ('full','split','resumed')]
        results[arm]={k:full[k]==resumed[k] for k in ('model','optimizer','scheduler','rng','data','counters')}
        results[arm]['schedules']=all(full['events'][k]==split['events'][k]+resumed['events'][k] for k in full['events'])
    receipt=dict(identity=identity(s),results=results,passed=all(all(r.values()) for r in results.values()))
    write_json(Path(s['output'])/'resume.json',receipt)
    return receipt


def real_smoke(s,seal):
    """Two common-geometry training-only updates per arm, then bounded checks."""
    stream=ShardedDocuments(Path(s['data']['root'])/'shards/65536/train')
    docs=[stream[i] for i in range(64)]
    counts=torch.tensor(seal['training_counts'],dtype=torch.int64); results={}
    validation=dict(sha256='d'*64,split='validation',protocol=s['evaluation'])
    for arm in s['arms']:
        gc.collect(); torch.cuda.empty_cache()
        model,_=construct(s,arm,counts); model.to(s['device'])
        data=TrainingData(docs,context=1024,stride=512,seed=42,tokenizer=seal['tokenizer'],dataset=dict(sha256='b'*64,split='train'))
        state=train(model,data,TrainConfig(**s['training']),validation_identity=validation,max_attempts=2)
        metrics=evaluate(model,docs[:2],evaluation_config(s['evaluation']),frequency_groups=quality_groups(model,counts))
        if sum(g['targets'] for g in metrics['frequency_metrics'].values())!=metrics['tokens']:
            raise ValueError('Frequency/band metrics do not cover every target')
        values=diagnostic(s,model,counts,docs[:2],endpoint=True)
        results[arm]=dict(counters=state['counters'],training_only_metrics=metrics,diagnostics=values,
                          finite=bool(np.isfinite(metrics['bpb'])),peak_memory=gpu_memory(s['device']))
        write_json(Path(s['output'])/'smoke.json',dict(identity=identity(s),results=results))
        del model,state
    return results


def freeze_protocol(s,seal,table,walks):
    protocol=dict(schema='64k_stress_test_v1',phase='prepared_for_review_not_authorized',
                  reference=s['reference'],identity=identity(s),config=s,
                  tokenizer_provenance=read_json(ARTIFACTS/'protocol.json'),
                  methodology_sha256=file_hash(ARTIFACTS/'methodology.json'),
                  data={k:v for k,v in seal.items() if k!='training_counts'},
                  counts_sha256=sha256(np.asarray(seal['training_counts'],dtype=np.int64).tobytes()),
                  allocations=table,data_walks=walks,
                  evaluation='fineweb_document_eos_c1024_s512_v2; screen [0,256), confirmation [256,512)',
                  completed_evidence_server_sha256=file_hash(ARTIFACTS/'completed_evidence_server.json'))
    target=ARTIFACTS/'experiment.json'
    if target.exists():
        if read_json(target)!=protocol: raise ValueError('Frozen 64K protocol drift; do not overwrite')
    else: immutable_json(target,protocol)
    return protocol


def preflight(s,args):
    root=Path(s['output']); checks={}
    # Evidence drift is fatal before any preparation/profile operation.
    evidence=verify_evidence(args.evidence); checks['completed_evidence_before']=evidence
    seal=inputs(s); table=allocations(s,torch.tensor(seal['training_counts'],dtype=torch.int64))
    write_json(root/'allocations.json',table)
    walks=stream_walks(s,seal,table)
    checks['headroom']=all(r['supported_with_margin'] for r in walks if r['budget']==500000000000000000)
    for name,callback,valid in (
        ('profiles',lambda:profile(s,seal),lambda r:len(r['rows'])==4 and all(x['status']=='PASS' for x in r['rows'])),
        ('resume',lambda:resume_check(s,args.config),lambda r:r['passed']),
        ('smoke',lambda:real_smoke(s,seal),lambda r:len(r)==4 and all(x['finite'] for x in r.values()))):
        try:
            result=callback(); checks[name]=dict(passed=bool(valid(result)))
        except Exception as error:
            checks[name]=dict(passed=False,reason=str(error))
    with (root/'tests.log').open('w') as log:
        tested=subprocess.run([sys.executable,'-m','pytest','-q','tests/training/test_vocab_scaling_64k.py',
                               '--basetemp',str(root/'test_tmp')],stdout=log,stderr=subprocess.STDOUT)
    checks['tests']=dict(passed=tested.returncode==0,log=str(root/'tests.log'))
    checks['completed_evidence_after']=verify_evidence(args.evidence)
    passed=checks['headroom'] and all(checks[k]['passed'] for k in ('profiles','resume','smoke','tests'))
    if passed: freeze_protocol(s,seal,table,walks)
    receipt=dict(status='PASS' if passed else 'FAIL',identity=identity(s),checks=checks,
                 environment=environment(torch.device(s['device'])),training_authorized=False,
                 experiment_sha256=file_hash(ARTIFACTS/'experiment.json') if (ARTIFACTS/'experiment.json').exists() else None)
    write_json(root/'preflight.json',receipt)
    return receipt


def report(s):
    """New comparison table only. Historical JSON/logs are opened read-only."""
    root=Path(s['output']); table=read_json(root/'allocations.json'); rows=[]
    profile_storage={r['arm']:r.get('storage',{}) for r in read_json(root/'profiles.json')['rows']} if (root/'profiles.json').exists() else {}
    for historical,arms,folder in (
        (True,['baseline16k','dense32k','lowrank32k','adaptive32k'],Path('results/vocab_scaling_large')),
        (False,list(s['arms']),root)):
        for arm in arms:
            path=folder/arm/'run'; result=read_json(path/'result.json') if (path/'result.json').exists() else None
            validation=[json.loads(line) for line in (path/'validation.jsonl').read_text().splitlines()] if (path/'validation.jsonl').exists() else []
            screens=[v for v in validation if v.get('subset') in ('screen','final_screen') and v.get('event')!='initial_screen']
            confirmation=[v for v in validation if v.get('subset')=='confirmation_final']
            final=screens[-1] if screens else {}; conf=confirmation[-1] if confirmation else {}
            storage=result['storage'] if result else ({**table[arm],**profile_storage.get(arm,{})} if not historical else {})
            counters=result['counters'] if result else {}
            spec=read_json(path/'manifest.json')['representation'] if historical and (path/'manifest.json').exists() else table.get(arm,{}).get('specification',{})
            row=dict(arm=arm,provenance='frozen completed e579a46' if historical else 'new 64K extension',
                vocab_size=spec.get('vocab_size'),representation=spec.get('kind'),
                status=result.get('completion','completed') if result else ('unavailable' if historical else 'not_started'),
                final_bpb=final.get('bpb'),confirmation_bpb=conf.get('bpb'),total_nll=final.get('nll'),
                token_nll=final.get('token_nll'),accuracy=final.get('accuracy'),target_count=final.get('tokens'),
                raw_validation_bytes=final.get('bytes'),confirmation_total_nll=conf.get('nll'),
                confirmation_token_nll=conf.get('token_nll'),confirmation_accuracy=conf.get('accuracy'),
                confirmation_target_count=conf.get('tokens'),confirmation_raw_bytes=conf.get('bytes'))
            row.update({k:counters.get(k) for k in ('estimated_flops','raw_bytes','scored_tokens','input_positions','successful_updates','active_seconds')})
            row.update({k:storage.get(k) for k in ('vocabulary_parameters','vocabulary_trainable_parameters','vocabulary_parameter_bytes',
                'vocabulary_fixed_buffer_bytes','vocabulary_integer_buffer_bytes','vocabulary_metadata_json_bytes',
                'vocabulary_serialized_inference_bytes','total_parameters','optimizer_state_tensor_bytes','deployment_bytes')})
            row['unused_flops']=result.get('unused_flops') if result else None
            rows.append(row)
    with (root/'summary.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(root/'summary.json',rows)
    return rows


def require_authorization(s):
    receipt=read_json(Path(s['output'])/'preflight.json')
    approval=read_json(Path(s['output'])/'training_authorization.json')
    current=identity(s); protocol_hash=file_hash(ARTIFACTS/'experiment.json')
    if receipt['status']!='PASS' or receipt['identity']!=current or receipt['experiment_sha256']!=protocol_hash:
        raise ValueError('Current passing 64K preflight and frozen protocol required')
    if receipt['environment']!=environment(torch.device(s['device'])):
        raise ValueError('Execution environment differs from A40 preflight')
    if approval.get('owner_approved') is not True or approval.get('identity')!=current or approval.get('experiment_sha256')!=protocol_hash:
        raise ValueError('Owner review and explicit authorization bound to this exact extension are required')
    # No command in this preparation tool creates authorization.
    verify_evidence(ARTIFACTS/'completed_evidence_server.json')


def append(path,row):
    with Path(path).open('a',encoding='utf-8') as f: f.write(json.dumps(row,allow_nan=False)+'\n')


def run(s,arm,resume=False,operation='train'):
    require_authorization(s)
    seal=inputs(s); counts=torch.tensor(seal['training_counts'],dtype=torch.int64)
    root=Path(s['output'])/arm/'run'
    if not resume and operation=='train' and root.exists() and any(root.iterdir()):
        raise FileExistsError('Fresh arm requires empty output')
    root.mkdir(parents=True,exist_ok=True)
    model,_=construct(s,arm,counts); model.to(s['device']); payload=None
    if (resume or operation!='train') and (root/'last.pt').exists():
        model,payload=load_training_checkpoint(root/'last.pt',s['device'])
    elif operation!='train': raise ValueError('No scientific 64K checkpoint')
    manifest=dict(identity=identity(s),arm=arm,representation=model.vocabulary.specification())
    if (root/'manifest.json').exists():
        if read_json(root/'manifest.json')!=manifest: raise ValueError('Run manifest drift')
    else: immutable_json(root/'manifest.json',manifest)
    limit=payload['training_state']['counters']['attempted_updates'] if payload else 0
    seen=set()
    for name in ('training','validation','diagnostics'):
        path=root/(name+'.jsonl'); kept=[]
        if path.exists():
            for line in path.read_text().splitlines():
                try: row=json.loads(line)
                except json.JSONDecodeError: break
                if row['attempted_updates']<=limit:
                    kept.append(line)
                    if 'event' in row: seen.add(row['event'])
            path.write_text('\n'.join(kept)+('\n' if kept else ''),encoding='utf-8')
    train_docs=ShardedDocuments(Path(s['data']['root'])/'shards/65536/train')
    valid_docs=ShardedDocuments(Path(s['data']['root'])/'shards/65536/validation')
    source=Path(s['data']['raw_root'])
    validation=dict(dataset_identity(source,'validation'),protocol=s['evaluation'])
    data=TrainingData(train_docs,context=1024,stride=512,seed=42,tokenizer=seal['tokenizer'],
                      dataset=dataset_identity(source,'train'),non_repeating=True)
    screen=[valid_docs[i] for i in range(256)]; confirm=[valid_docs[i] for i in range(256,512)]
    groups=quality_groups(model,counts)
    def observe(state,event,subset=screen):
        if event in seen: return None
        value=evaluate(model,subset,evaluation_config(s['evaluation']),frequency_groups=groups)
        for r in value['frequency_metrics'].values():
            r.update(token_nll=r['nll']/r['targets'],accuracy=r['correct']/r['targets'],mass=r['targets']/value['tokens'])
        append(root/'validation.jsonl',dict(**value,**state['counters'],event=event,
               subset='confirmation_final' if event=='confirmation_final' else 'screen'))
        seen.add(event); return value
    def diag(state,event,endpoint=False):
        if event in seen: return
        values=diagnostic(s,model,counts,[train_docs[i] for i in range(2)],endpoint=endpoint)
        append(root/'diagnostics.jsonl',dict(**state['counters'],event=event,values=values))
        seen.add(event)
    if operation=='confirm':
        if any(read_json(Path(s['output'])/a/'run/result.json')['completion']!='budget_boundary' for a in s['arms']):
            raise ValueError('All four new arms must finish before confirmation')
        if 'confirmation_final' in seen: raise ValueError('Confirmation already recorded')
        return observe(payload['training_state'],'confirmation_final',confirm)
    if operation=='evaluate': return observe(payload['training_state'],'manual_screen')
    previous=payload['training_state']['counters']['estimated_flops'] if payload else 0
    t=TrainConfig(**s['training'])
    def boundary(state,model):
        nonlocal previous
        current=state['counters']['estimated_flops']
        for fraction in s['evaluation_fractions']:
            if previous<int(fraction*t.budget_limit)<=current:
                value=observe(state,'screen_'+str(fraction))
                if value is not None and (state['best_bpb'] is None or value['bpb']<state['best_bpb']):
                    state['best_bpb']=value['bpb']
                    save_training_checkpoint(root/'best.pt',model,state,tokenizer=seal['tokenizer'],dataset=validation)
        for fraction in s['diagnostic_fractions']:
            if previous<int(fraction*t.budget_limit)<=current: diag(state,'diag_'+str(fraction),fraction==1.)
        previous=current
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
    def metrics(row):
        row.pop('examples'); append(root/'training.jsonl',dict(**row,**gpu_memory(s['device'])))
    try:
        if payload is None:
            from vocab_lm.train import Counters
            observe(dict(counters=asdict(Counters())),'initial_screen')
        state=train(model,data,t,validation_identity=validation,output=root,
                    resume=payload['training_state'] if payload else None,on_metrics=metrics,on_boundary=boundary,
                    should_stop=lambda:stopped,experiment_identity=manifest)
        if not stopped:
            observe(state,'screen_1.0'); diag(state,'diag_1.0',endpoint=True)
    finally:
        for sig,handler in handlers.items(): signal.signal(sig,handler)
    result=dict(completion='job_boundary' if stopped else 'budget_boundary',counters=state['counters'],
                unused_flops=t.budget_limit-state['counters']['estimated_flops'],
                storage=storage_report(model,optimizer_state=state['optimizer'],tokenizer_paths=[s['tokenizer']['artifact']],
                    tokenizer=seal['tokenizer'],dataset=validation),provenance='64K additive extension; historical experiment read-only')
    write_json(root/'result.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['tokenizer','prepare','storage','walk','verify-evidence',
                                            'profile','resume-check','resume-worker','smoke','preflight',
                                            'train','evaluate','confirm','report'])
    parser.add_argument('--config',default='configs/vocab_scaling_64k.yaml')
    parser.add_argument('--sample-root',default='data/fineweb_v1')
    parser.add_argument('--evidence',default='artifacts/vocab_scaling_64k/completed_evidence.json')
    parser.add_argument('--arm',choices=list(ARM_KINDS))
    parser.add_argument('--stage',choices=['full','split','resumed'])
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv); s=settings(args.config)
    torch.set_num_threads(s['threads']); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    Path(s['output']).mkdir(parents=True,exist_ok=True)
    if args.operation=='verify-evidence': result=verify_evidence(args.evidence)
    elif args.operation=='tokenizer': result=prepare_tokenizer(s,args.sample_root)
    elif args.operation=='prepare': result=prepare_data(s); result={k:v for k,v in result.items() if k not in ('training_counts','hashes')}
    elif args.operation=='resume-worker': result=resume_worker(s,args.arm,args.stage)
    elif args.operation=='resume-check': result=resume_check(s,args.config)
    elif args.operation=='profile': result=profile(s,inputs(s),[args.arm] if args.arm else None)
    elif args.operation=='smoke': result=real_smoke(s,inputs(s))
    elif args.operation=='preflight': result=preflight(s,args)
    elif args.operation=='report': result=report(s)
    elif args.operation in ('train','evaluate','confirm'):
        if args.arm is None: parser.error('--arm required')
        result=run(s,args.arm,args.resume,args.operation)
    else:
        seal=inputs(s) if (Path(s['data']['root'])/'preparation.json').exists() else None
        result=allocations(s,torch.tensor(seal['training_counts'],dtype=torch.int64) if seal else None)
        write_json(Path(s['output'])/'allocations.json',result)
        if args.operation=='walk':
            if seal is None: raise ValueError('Final 64K prepared data required')
            result=stream_walks(s,seal,result)
    print(json.dumps(result,indent=2,allow_nan=False))
    if args.operation=='preflight' and result['status']!='PASS': return 1
    return 0


if __name__=='__main__': sys.exit(main())
