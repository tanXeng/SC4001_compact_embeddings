"""Independent runs of the frozen protocols; historical runners remain unchanged.

Adapters are private module instances, never patches to the historical imports.
Only execution identity, local receipt gates and output/data locations differ.
Optimization, representations, scoring, diagnostics and checkpoints are reused.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
from importlib.metadata import version
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ''}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from vocab_lm.runtime import file_hash, read_json, sha256, source_identity, environment, write_json
from vocab_lm.fineweb import prepare_scaling_data

LOCK=Path('artifacts/reproduction/scientific_inputs.json')


def verify_science():
    for package,expected in dict(torch='2.5.1',numpy='1.24.4',tokenizers='0.22.1',PyYAML='6.0.2',pyarrow='18.1.0',matplotlib='3.7.5').items():
        if version(package).split('+')[0]!=expected:
            raise ValueError('Pinned dependency differs: '+package)
    lock=read_json(LOCK)
    for name,r in lock['files'].items():
        payload=Path(name).read_bytes()
        if r['mode']=='lf': payload=payload.replace(b'\r\n',b'\n')
        if sha256(payload)!=r['sha256']:
            raise ValueError('Frozen scientific input changed: '+name)
    return file_hash(LOCK)


def private_runner(filename):
    spec=importlib.util.spec_from_file_location('scripts._independent_'+Path(filename).stem,filename)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def configuration(study,run):
    verify_science()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',run):
        raise ValueError('Run name must be a simple independent identifier')
    large=read_json('artifacts/vocab_scaling_large/protocol.json')
    extension=read_json('artifacts/vocab_scaling_64k/experiment.json')
    expected=large['config'] if study=='large' else extension['config']
    filename='configs/vocab_scaling_large.yaml' if study=='large' else 'configs/vocab_scaling_64k.yaml'
    frozen=yaml.safe_load(Path(filename).read_text(encoding='utf-8'))
    if frozen!=expected: raise ValueError('YAML disagrees with frozen scientific protocol')
    s=copy.deepcopy(frozen)
    s['output']=(Path('results/reproductions')/run/study).as_posix()
    s['data']['root']='data/reproductions/common' if study=='large' else 'data/reproductions/64k'
    if study=='64k': s['data']['raw_root']='data/reproductions/common'
    for name in [s['output'],s['data']['root']]:
        p=Path(name)
        if any(x.is_symlink() for x in [p,*p.parents]):
            raise ValueError('Independent output/data paths must not be symlinks')
    return s


def verify_data(study,s):
    frozen=read_json('artifacts/vocab_scaling_large/protocol.json')
    common=Path('data/reproductions/common')
    seal=read_json(common/'preparation.json')
    for name,digest in frozen['dataset_hashes'].items():
        path=(common/name).resolve()
        if not path.is_relative_to(common.resolve()) or file_hash(path)!=digest:
            raise ValueError('Prepared common data differ: '+name)
    if file_hash(common/'preparation.json')!=frozen['preparation_sha256']:
        raise ValueError('Common preparation differs from frozen protocol')
    if study=='large': return seal
    expected=read_json('artifacts/vocab_scaling_64k/experiment.json')
    root=Path(s['data']['root']); seal=read_json(root/'preparation.json')
    for key,value in expected['data'].items():
        if key!='source_root' and seal.get(key)!=value:
            raise ValueError('64K data metadata differ: '+key)
    counts=torch.tensor(seal['training_counts'],dtype=torch.int64)
    if sha256(counts.numpy().tobytes())!=expected['counts_sha256']:
        raise ValueError('64K complete training frequencies differ')
    for name,digest in seal['hashes'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path)!=digest:
            raise ValueError('64K prepared shard differs: '+name)
    return seal


def binding(study,s):
    return dict(role='independent_reproduction',study=study,scientific_inputs_sha256=verify_science(),
                implementation=source_identity(),config_sha256=sha256(json.dumps(s,sort_keys=True).encode()),
                data_preparation_sha256=file_hash(Path(s['data']['root'])/'preparation.json'))


def require_local(study,s,authorized=True):
    verify_data(study,s)
    receipt=read_json(Path(s['output'])/'reproduction_preflight.json')
    if receipt['status']!='PASS' or receipt['binding']!=binding(study,s):
        raise ValueError('Current independent preflight required')
    if receipt['environment']!=environment(torch.device(s['device'])):
        raise ValueError('Independent machine/environment changed; repeat preflight')
    if authorized:
        approval=read_json(Path(s['output'])/'reproduction_authorization.json')
        if approval!=dict(accepted_frozen_protocol=True,binding=receipt['binding'],
                          preflight_sha256=file_hash(Path(s['output'])/'reproduction_preflight.json')):
            raise ValueError('Explicit authorization for this reproduction required')
    return receipt


def adapters(study,s):
    from scripts.vocab_scaling import concurrent_readers
    large=private_runner('scripts/vocab_scaling.py')
    extension=private_runner('scripts/vocab_scaling_64k.py')
    # These hooks only replace private historical-execution gates. They require
    # new local scientific/data/resource evidence and cannot authorize old paths.
    large.require_large_receipt=lambda config:require_local('large',config)
    extension.require_authorization=lambda config:require_local('64k',config)
    # Spawn targets must retain their importable original module, rather than
    # the private orchestration instance. This helper has no historical gate.
    large.concurrent_readers=concurrent_readers
    return large,extension


def prepare(study,s,extension):
    common=Path('data/reproductions/common')
    if not (common/'preparation.json').exists() and shutil.disk_usage('.').free<16_000_000_000:
        raise ValueError('At least16GB free disk required for pinned data preparation')
    config=read_json('artifacts/vocab_scaling_large/protocol.json')['config']['protocol']['preparation']
    prepare_scaling_data(common,'artifacts/vocab_scaling_large/tokenizers',config)
    if study=='64k': extension.prepare_data(s)
    verify_data(study,s)


def resume_checks(study,run,s,large,extension):
    results={}
    for arm in s['arms']:
        for stage in ('full','split','resumed'):
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'_resume-worker',
                '--study',study,'--run',run,'--arm',arm,'--stage',stage],check=True,stdout=subprocess.DEVNULL)
        folder=Path(s['output'])/('synthetic_resume' if study=='large' else 'resume')/arm
        full,split,resumed=[read_json(folder/(name+'.json')) for name in ('full','split','resumed')]
        results[arm]=all(full[k]==resumed[k] for k in ('model','optimizer','scheduler','rng','data','counters')) and all(
            full['events'][k]==split['events'][k]+resumed['events'][k] for k in full['events'])
    if not all(results.values()): raise ValueError('Exact process-boundary resume failed')
    return results


def preflight(study,run,s,large,extension):
    seal=verify_data(study,s); root=Path(s['output']); root.mkdir(parents=True,exist_ok=True)
    if list(root.glob('*/run/*.pt')): raise ValueError('Do not regenerate preflight after scientific training; use a new run name')
    receipt=dict(status='FAIL',binding=binding(study,s),environment=None)
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<30*2**30:
            raise ValueError('CUDA GPU with at least30GiB capacity required; CPU verification cannot authorize training')
        receipt['environment']=environment(torch.device(s['device']))
        counts=torch.tensor(seal['training_counts'] if study=='64k' else seal['training_counts']['32768'],dtype=torch.int64)
        allocations=large.allocation_table(s,seal) if study=='large' else extension.allocations(s,counts)
        expected=read_json('artifacts/vocab_scaling_large/protocol.json' if study=='large' else 'artifacts/vocab_scaling_64k/experiment.json')['allocations']
        if allocations!=expected: raise ValueError('Representation/storage allocation differs from frozen protocol')
        if study=='large':
            walks=large.budget_projections(s,allocations)
        else:
            walks=extension.stream_walks(s,seal,allocations)
        if not all(r['supported_with_margin'] for r in walks if r['budget']==s['training']['budget_limit']):
            raise ValueError('Exact stream walk lacks20% compute headroom')
        # Reuse the existing generic profiling orchestration for both studies.
        # Its input convention is a mapping keyed by vocabulary size.
        prof_settings=copy.deepcopy(s)
        if study=='64k':
            inputs=root/'profile_inputs'; inputs.mkdir(exist_ok=True)
            write_json(inputs/'preparation.json',dict(training_counts={'65536':seal['training_counts']}))
            prof_settings['data']['root']=str(inputs)
            large.construct=lambda settings,arm,counts:(extension.construct(s,arm,counts)[0],s['reference']['ceiling_bytes'])
        profile=large.synthetic_profile(prof_settings)
        limit=min((24 if study=='large' else 38)*2**30,int(receipt['environment']['gpu_total_bytes']*.9))
        if len(profile['rows'])!=4 or not all(r.get('status')=='PASS' and r.get('checkpoint_roundtrip') and
                r.get('finite_loss') and r.get('finite_gradients') and r.get('initialized_adamw') and
                (r.get('microbatch'),r.get('accumulation'),r.get('context'))==(16,2,1024) and
                max(r[k] for k in ['peak_allocated','peak_reserved','checkpoint_peak_allocated','checkpoint_peak_reserved'])<=limit
                for r in profile['rows']):
            raise ValueError('Common16x2 FP32 batch/profile exceeds safe GPU resources or failed')
        receipt['resume']=resume_checks(study,run,s,large,extension)
        if study=='large':
            smoke=large.large_smoke(s)
            if not smoke['passed']: raise ValueError('Bounded real-data integration failed')
            readers=large.concurrent_readers(s)
            if not readers['passed']: raise ValueError('Independent read-only shard readers failed')
        else:
            smoke=extension.real_smoke(s,seal)
            if len(smoke)!=4 or not all(r['finite'] for r in smoke.values()): raise ValueError('Bounded real-data integration failed')
        receipt.update(status='PASS',allocations=allocations,data_walks=walks)
    except Exception as error:
        receipt['error']=str(error)
        write_json(root/'reproduction_preflight.json',receipt)
        raise
    write_json(root/'reproduction_preflight.json',receipt)
    return receipt


def report(s):
    rows=[]
    for arm in s['arms']:
        folder=Path(s['output'])/arm/'run'; result=read_json(folder/'result.json')
        observations=[json.loads(line) for line in (folder/'validation.jsonl').read_text().splitlines()]
        rows.append(dict(arm=arm,role='independent_reproduction',result=result,observations=observations,
                         diagnostics=[json.loads(line) for line in (folder/'diagnostics.jsonl').read_text().splitlines()]))
    write_json(Path(s['output'])/'reproduction_report.json',rows)
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['verify','prepare','preflight','authorize','train','evaluate','confirm','diagnostics','report','_resume-worker'])
    p.add_argument('--study',choices=['large','64k'],required=True)
    p.add_argument('--run',required=True,help='Independent local run name; never an historical output path')
    p.add_argument('--arm');p.add_argument('--resume',action='store_true')
    p.add_argument('--accept-frozen-protocol',action='store_true')
    p.add_argument('--stage',choices=['full','split','resumed'],help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    s=configuration(a.study,a.run);large,extension=adapters(a.study,s)
    if a.arm is not None and a.arm not in s['arms']: p.error('Unknown frozen arm')
    if a.operation=='verify': return dict(status='PASS',scientific_inputs_sha256=verify_science(),private_historical_receipts_required=False)
    if a.operation=='prepare': prepare(a.study,s,extension);return dict(status='PASS')
    if a.operation=='preflight': return preflight(a.study,a.run,s,large,extension)
    if a.operation=='authorize':
        if not a.accept_frozen_protocol: p.error('Explicit --accept-frozen-protocol required')
        receipt=require_local(a.study,s,False)
        target=Path(s['output'])/'reproduction_authorization.json'
        value=dict(accepted_frozen_protocol=True,binding=receipt['binding'],preflight_sha256=file_hash(Path(s['output'])/'reproduction_preflight.json'))
        if target.exists():
            if read_json(target)!=value: raise ValueError('Existing authorization differs')
        else: extension.immutable_json(target,value)
        return dict(status='PASS',role='independent_reproduction')
    if a.operation=='_resume-worker':
        if a.arm is None or a.stage is None:p.error('Worker requires arm and stage')
        return large.synthetic_resume_worker(s,a.arm,a.stage) if a.study=='large' else extension.resume_worker(s,a.arm,a.stage)
    if a.operation=='report': return report(s)
    if a.operation=='diagnostics':
        # Endpoint computations are part of the frozen training completion path.
        # Export their existing records rather than duplicating gradient attribution.
        return report(s)
    if a.arm is None:p.error('--arm required')
    if a.operation=='confirm':
        path=Path(s['output'])/a.arm/'run/validation.jsonl'
        if path.exists() and any(json.loads(line)['subset']=='confirmation_final' for line in path.read_text().splitlines()):
            raise ValueError('This independent arm already has its once-only confirmation')
    if a.study=='large': return large.run(s,a.arm,resume=a.resume,operation=a.operation)
    return extension.run(s,a.arm,resume=a.resume,operation=a.operation)


if __name__=='__main__':
    print(json.dumps(main(),allow_nan=False,default=str))
