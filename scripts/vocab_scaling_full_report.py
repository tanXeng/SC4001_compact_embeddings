"""Read-only audits and reporting for the frozen full toy experiment."""
import csv
import json
from pathlib import Path
import sys

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scripts.vocab_scaling import load_settings, prepared, construct
from vocab_lm.checkpoint import load_training_checkpoint
from vocab_lm.train import scaling_training_flops
from vocab_lm.runtime import write_json, source_identity


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit(s):
    root = Path(s['output'])
    protocol = json.loads((root/'manifest.json').read_text())
    assert protocol['source'] == source_identity()
    assert protocol['config'] == s
    evidence = {}
    for arm in s['arms']:
        p = root/arm/'run'
        model, payload = load_training_checkpoint(p/'last.pt')
        state = payload['training_state']
        result = json.loads((p/'result.json').read_text())
        rows = jsonl(p/'training.jsonl')
        vals = jsonl(p/'validation.jsonl')
        c = state['counters']
        assert c == result['counters']
        assert c['estimated_flops'] == sum(r['update_flops'] for r in rows)
        assert c['raw_bytes'] == sum(r['update_bytes'] for r in rows)
        assert c['scored_tokens'] == sum(r['update_tokens'] for r in rows)
        assert c['successful_updates'] == c['attempted_updates'] == len(rows)
        assert all(r['successful'] and np.isfinite(r['loss']) for r in rows)
        assert [r['attempted_updates'] for r in rows] == list(range(1,len(rows)+1))
        assert state['scheduler']['completed'] == len(rows)
        assert c['estimated_flops'] <= s['training']['budget_limit']
        data, *_ = prepared(s,arm)
        data.load_state_dict(state['data'])
        batches = [data.batch(s['training']['microbatch_size']) for _ in range(s['training']['accumulation'])]
        next_flops = sum(scaling_training_flops(model,*b['inputs'].shape) for b in batches)
        unused = s['training']['budget_limit']-c['estimated_flops']
        assert 0 <= unused < next_flops
        screening = [v for v in vals if v['subset'] != 'confirmation_final']
        assert len(screening) == 5
        assert screening[0]['estimated_flops'] == 0
        for fraction, v in zip(s['evaluation_fractions'][:-1],screening[1:-1]):
            index = v['attempted_updates']-1
            assert rows[index]['estimated_flops'] >= int(fraction*s['training']['budget_limit'])
            assert index == 0 or rows[index-1]['estimated_flops'] < int(fraction*s['training']['budget_limit'])
        assert screening[-1]['estimated_flops'] == c['estimated_flops']
        confirmation = [v for v in vals if v['subset']=='confirmation_final']
        assert len(confirmation) <= 1
        if confirmation:
            assert confirmation[0]['estimated_flops'] == c['estimated_flops']
        evidence[arm] = dict(counters=c,unused_flops=unused,next_group_flops=next_flops,
            screening_observations=len(screening),confirmation_observations=len(confirmation),
            checkpoint_valid=True, cursor=state['data']['cursor'])
    write_json(root/'integrity.json',evidence)
    return evidence


def report(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(s['output'])
    audit(s)
    fig, axes = plt.subplots(4,3,figsize=(18,19))
    axes = axes.flatten()
    summaries, lines = [], ['# Full toy FP32 IsoFLOPs vocabulary scaling', '',
        'One seed (42), 8K Dense versus 16K tied Low-rank r126. This compares vocabulary size AND representation. No statistical-significance or vocabulary-only causal claim is justified.', '',
        'Confirmation limitation frozen before training: the secondary documents are disjoint from screening and training, but were evaluated in feasibility. Their untouched status cannot be established; this is reused secondary validation, not independent confirmation. No sealed test data was accessed.', '',
        'The unchanged toy configuration repeats the 574-document smoke corpus (2,000,916 unique bytes). Cumulative raw bytes below measure exposure, including repeated passes, rather than unique data. Both arms use identical ordered raw documents and seed policy.', '',
        '## Endpoints', '',
        '| Arm | Screening BPB | Secondary BPB | FLOPs | Unused FLOPs | Raw bytes | Tokens | Updates | Active seconds | Peak allocated / reserved bytes | Vocabulary / inference bytes |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    endpoints = {}
    for arm in s['arms']:
        p = root/arm/'run'
        result = json.loads((p/'result.json').read_text())
        training, validation, diagnostics = [jsonl(p/name) for name in ('training.jsonl','validation.jsonl','diagnostics.jsonl')]
        vals = [v for v in validation if v['subset'] != 'confirmation_final']
        confirm = [v for v in validation if v['subset']=='confirmation_final']
        assert len(confirm)==1
        c, storage, memory = result['counters'],result['storage'],result['peak_memory']
        row = dict(arm=arm,screening_bpb=vals[-1]['bpb'],secondary_bpb=confirm[0]['bpb'],
            **c, **storage, **memory, unused_flops=result['unused_flops'],
            raw_bytes_per_active_second=c['raw_bytes']/c['active_seconds'],
            tokens_per_active_second=c['scored_tokens']/c['active_seconds'],
            flops_per_successful_update=c['estimated_flops']/c['successful_updates'])
        summaries.append(row)
        endpoints[arm] = (row,diagnostics[-1]['values'])
        lines.append(f"| {arm} | {row['screening_bpb']:.6f} | {row['secondary_bpb']:.6f} | {c['estimated_flops']} | {row['unused_flops']} | {c['raw_bytes']} | {c['scored_tokens']} | {c['successful_updates']} | {c['active_seconds']:.3f} | {memory['peak_allocated']} / {memory['peak_reserved']} | {storage['vocabulary_serialized_inference_bytes']} / {storage['serialized_inference_checkpoint_bytes']} |")
        for index,key,label in [(0,'estimated_flops','Estimated training FLOPs'),(1,'raw_bytes','Raw training bytes'),(2,'active_seconds','Active training seconds'),(3,'successful_updates','Successful updates')]:
            axes[index].plot([v[key] for v in vals],[v['bpb'] for v in vals],'o-',label=arm)
            axes[index].set(xlabel=label,ylabel='Screening BPB')
        axes[4].plot([r['estimated_flops'] for r in training],[r['loss'] for r in training],alpha=.6,label=arm)
        axes[4].set(xlabel='Estimated training FLOPs',ylabel='Training token NLL')
        # Bounded contiguous blocks reduce timing noise without selecting observations.
        blocks = np.array_split(np.arange(len(training)),min(100,len(training)))
        for index, numerator, label in [(5,'update_bytes','Raw bytes/s'),(6,'update_tokens','Scored tokens/s')]:
            axes[index].plot([training[int(b[-1])]['estimated_flops'] for b in blocks],
                [sum(training[int(i)][numerator] for i in b)/sum(training[int(i)]['update_seconds'] for i in b) for b in blocks],label=arm)
            axes[index].set(xlabel='Estimated training FLOPs',ylabel=label)
        for offset,key in enumerate(('peak_allocated','peak_reserved')):
            axes[7].bar(f'{arm}\n{key}',memory[key]/1e6,label=f'{arm} {key}')
        axes[7].set(ylabel='CUDA MB')
        for key in ('vocabulary_serialized_inference_bytes','serialized_inference_checkpoint_bytes'):
            axes[8].bar(f'{arm}\n{key.split("_")[0]}',storage[key]/1e6,label=f'{arm} {key.split("_")[0]}')
        axes[8].set(ylabel='Archive MB')
        dv = diagnostics[-1]['values']
        axes[9].semilogy(dv['singular_values'],label=arm)
        axes[9].set(xlabel='Sampled singular index',ylabel='Endpoint singular value')
        for index,key,label in [(10,'entropy_rank','Sampled entropy rank')]:
            axes[index].plot([d['estimated_flops'] for d in diagnostics],[d['values'][key] for d in diagnostics],'o-',label=arm)
            axes[index].set(xlabel='Estimated training FLOPs',ylabel=label)
        fm = vals[-1]['frequency_metrics']
        axes[11].plot([int(k) for k in fm],[v['nll']/v['targets'] for v in fm.values()],'o-',label=arm)
        axes[11].set(xlabel='Token-type frequency stratum (0 frequent)',ylabel='Endpoint screening token NLL')
        lines.extend(['',f'## {arm} details','',
            f"Active byte throughput {row['raw_bytes_per_active_second']:.2f} bytes/s; token throughput {row['tokens_per_active_second']:.2f} targets/s. Mean estimated FLOPs/successful update {row['flops_per_successful_update']:.3f}. Per-update FLOPs are in training.jsonl; they vary with executed shapes.",
            f"Trainable parameters {storage.get('trainable_parameters')}; parameter bytes {training[-1]['parameter_bytes']}; gradient bytes {training[-1]['gradient_bytes']}; initialized optimizer bytes {training[-1]['optimizer_bytes']}. Host peak RSS is unmeasured.",
            f"Endpoint sampled numerical rank {dv['numerical_rank']}; entropy rank {dv['entropy_rank']:.3f}; effective norm mean {np.mean(dv['effective_norms']):.5f}; coordinate variance mean {np.mean(dv['coordinate_variance']):.7f}. Factor norms {dv['factor_norms']}; projection condition {dv.get('projection_condition')}; rank utilization {dv.get('rank_utilization')}.",
            'The deterministic spectrum uses 128 sampled effective rows, so Dense numerical rank cannot exceed 128. This is not the full embedding spectrum. Low-rank utilization near one shows use of available rank, not proof that a larger rank would improve BPB. Factor pre-clipping gradient norms and global pre/post-clipping norms are logged every update; diagnostic gradient norms are post-clipping and endpoint reloading has no live gradients.', '',
            '| Stratum | Screening targets | Screening token NLL | Screening accuracy | Secondary token NLL |','|---|---:|---:|---:|---:|'])
        for k,v in fm.items():
            secondary = confirm[0]['frequency_metrics'][k]
            lines.append(f"| {k} | {v['targets']} | {v['nll']/v['targets']:.6f} | {v['correct']/v['targets']:.6f} | {secondary['nll']/secondary['targets']:.6f} |")
        lines.extend(['','Frequency strata are tokenizer-specific thirds of token types ordered by training-only counts (ties by ID). They are descriptive, not matched cross-tokenizer target groups.','','| Screening point | FLOPs | Raw bytes | Tokens | Updates | Active seconds | Wall seconds | BPB |','|---|---:|---:|---:|---:|---:|---:|---:|'])
        for v in vals:
            lines.append(f"| {v['subset']} | {v['estimated_flops']} | {v['raw_bytes']} | {v['scored_tokens']} | {v['successful_updates']} | {v['active_seconds']:.3f} | {v['wall_seconds']:.3f} | {v['bpb']:.6f} |")
    a,da = endpoints['baseline8k']; b,db = endpoints['vocab16k']
    winner = min(summaries,key=lambda r:r['screening_bpb'])['arm']
    lines.extend(['','## Interpretation','',
        f"At the matched configured compute ceiling, {winner} achieved lower final screening BPB. Low-rank minus Dense screening BPB is {b['screening_bpb']-a['screening_bpb']:+.6f}; reused secondary BPB difference is {b['secondary_bpb']-a['secondary_bpb']:+.6f}. Actual compute differs only by unused complete-group remainders; no interpolation is applied.",
        f"Dense processed {a['raw_bytes']:,} bytes in {a['active_seconds']:.2f} active seconds; Low-rank processed {b['raw_bytes']:,} in {b['active_seconds']:.2f}. Exposure ratio is {b['raw_bytes']/a['raw_bytes']:.4f}; scored token/byte ratio is {a['scored_tokens']/a['raw_bytes']:.6f} Dense versus {b['scored_tokens']/b['raw_bytes']:.6f} Low-rank. The 16K token reduction {'did' if b['raw_bytes']>a['raw_bytes'] else 'did not'} compensate for its output cost under this FLOP convention, judged by raw bytes processed at fixed compute. Wall-clock efficiency is separately measured above.",
        f"Peak allocated GPU memory is {a['peak_allocated']/1e6:.2f} MB Dense versus {b['peak_allocated']/1e6:.2f} MB Low-rank. The measured memory investigation attributes approximately 67.1 MB of the gap to four overlapping vocabulary-sized logits/CE-backward tensors. The HB^T intermediate is only approximately 258 KB; persistent Low-rank state is slightly smaller. This full-softmax memory cost remains part of the trade-off.",
        f"Low-rank has a hard effective-rank ceiling of 126; its sampled endpoint rank is {db['numerical_rank']} with utilization {db.get('rank_utilization')}. Dense sampled rank is {da['numerical_rank']}, bounded by the 128-row diagnostic. Spectra, variance, conditioning and frequency metrics describe potential limits but cannot establish the cause of quality differences. Repeated small-corpus exposure and eight-document screening/secondary samples limit generalization.",
        'The shared pipeline supports frozen compute budgets, atomic periodic checkpoints, CPU/CUDA continuation, and independent job resumes with `python scripts/vocab_scaling.py train --config configs/vocab_scaling_full_toy.yaml --arm baseline8k --resume` (or vocab16k). Global budget, RNG, scheduler, cursor and counters persist; threshold progression is recovered from cumulative FLOPs. No extra job-boundary evaluations are added. A hard kill may replay work after the last checkpoint; logs are reconciled to that checkpoint.',
        'The workflow is technically suitable for future six-hour jobs, but scaling to the approximately 35M, 16K-versus-32K study still requires separately authorized tokenizer preparation, a frozen large-model/storage/budget protocol, hardware feasibility, and a demonstrably untouched confirmation set. This toy single-seed result does not authorize or establish large-scale quality.', '',
        '## Accounting and timing', '',
        'Frozen estimator vocab_scaling_v1: multiply-add is two FLOPs; matrix training uses forward plus twice-forward backward. Backbone matrix work is 6*n*L*(4*d*d+2*d*f); attention products 12*L*B*T*T*d; attention softmax 9*L*B*H*T*T; Dense vocabulary 6*n*V*d; Low-rank 6*n*r*(V+2*d). Actual padded/overlapping positions and full square attention count. Indexed Dense input is a gather. Bias, normalization, GELU, dropout, indexing/scatter, output loss/softmax, optimizer and clipping are excluded. These are analytical FLOPs, not exact executed hardware operations; excluded output CE becomes more relevant with larger V.',
        'Active time covers synchronized complete backward/clipping/AdamW updates, excluding CPU batch construction, validation, diagnostics, checkpoint I/O and initialization. Wall counters include in-session overhead. Throughput curves aggregate consecutive blocks without selecting samples. FP32 parameters/logits/loss; TF32 disabled. Archive convention standalone_torch_save_v1 includes serialized metadata/overhead; tied tensors count once.',
        'Complete machine-readable evidence: root manifest.json, preflight.json, integrity.json; per-arm run/manifest.json, training.jsonl, validation.jsonl, diagnostics.jsonl, result.json, last.pt and best.pt. Final checkpoint, rather than best screening checkpoint, supplies secondary validation. Confirmation is called exactly once per arm after both training runs complete.'])
    for ax in axes:
        ax.legend(fontsize=6)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(root/'curves.png',dpi=140)
    plt.close(fig)
    with (root/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    s=load_settings(sys.argv[2] if len(sys.argv)>2 else 'configs/vocab_scaling_full_toy.yaml')
    if sys.argv[1]=='audit':
        audit(s)
    elif sys.argv[1]=='report':
        report(s)
    else:
        raise ValueError('Use audit or report')
