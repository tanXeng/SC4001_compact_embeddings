"""Bounded Hash diagnostics for the additive extension; never an optimizer step."""
from collections import Counter
import itertools

import torch
from torch import nn
from torch.nn import functional as F


def distribution(x):
    x=x.detach().float().flatten()
    return dict(min=float(x.min()),max=float(x.max()),mean=float(x.mean()),
                std=float(x.std(unbiased=False)),quantiles=torch.quantile(x,torch.tensor([0.,.25,.5,.75,1.],device=x.device)).tolist())


@torch.no_grad()
def hash_structure(v, counts):
    indices=v.indices.cpu(); counts=counts.cpu().double()
    occupancy=torch.bincount(indices.flatten(),minlength=v.buckets)
    weighted=torch.zeros(v.buckets,dtype=torch.float64)
    weighted.scatter_add_(0,indices.flatten(),counts[:,None].expand(-1,2).flatten())
    ordered=Counter(map(tuple,indices.tolist()))
    unordered=Counter(tuple(sorted(x)) for x in indices.tolist())
    pairs=lambda c:sum(n*(n-1)//2 for n in c.values())
    return dict(k=2,buckets=v.buckets,seeds=list(v.seeds),used_buckets=int((occupancy>0).sum()),
                bucket_utilization=float((occupancy>0).float().mean()),occupancy=distribution(occupancy),
                occupancy_histogram=dict(Counter(occupancy.tolist())),frequency_weighted_occupancy=distribution(weighted),
                ordered_full_code_collision_pairs=pairs(ordered),unordered_full_code_collision_pairs=pairs(unordered),
                duplicate_within_token_components=int((indices[:,0]==indices[:,1]).sum()),
                partial_component_sharing='Counts distinct token pairs sharing at least one bucket; excludes unordered full-code collisions',
                importance_coefficients=distribution(v.importance),importance_absolute=distribution(v.importance.abs()),
                negative_importance_fraction=float((v.importance<0).float().mean()))


def shared_pairs(indices):
    # Every token belongs to at most two buckets, so a pair is double-counted only
    # when its unordered complete code is identical (and has two distinct codes).
    occupancy=Counter()
    for row in indices.tolist(): occupancy.update(set(row))
    unordered=Counter(tuple(sorted(row)) for row in indices.tolist())
    pairs=sum(n*(n-1)//2 for n in occupancy.values())
    pairs-=sum(n*(n-1)//2 for code,n in unordered.items() if code[0]!=code[1])
    full=sum(n*(n-1)//2 for n in unordered.values())
    return dict(any_shared_component_token_pairs=pairs,partial_only_token_pairs=pairs-full)


def effective_path_gradients(model, documents, policy):
    """Unaccumulated derivatives of real LM CE through independent tied paths."""
    v=model.vocabulary; device=next(model.parameters()).device
    with torch.no_grad():
        effective=v.lookup(torch.arange(v.vocab_size,device=device)).detach()
    ein=effective.clone().requires_grad_(); eout=effective.clone().requires_grad_()

    class Paths(nn.Module):
        def lookup(self,ids): return F.embedding(ids,ein)
        def project(self,h): return h@eout.T

    previous=model.training
    model.vocabulary=Paths(); model.eval()
    total=0; objective=0
    try:
        for doc in list(documents)[:policy['documents']]:
            n=min(policy['targets'],len(doc.ids)-1)
            x=torch.tensor([doc.ids[:n]],device=device)
            y=torch.tensor(doc.ids[1:n+1],device=device)
            objective=objective+F.cross_entropy(model(x)[0],y,reduction='sum')
            total+=n
        if not total: raise ValueError('No Hash interference targets')
        gin,gout=torch.autograd.grad(objective/total,(ein,eout))
    finally:
        model.vocabulary=v; model.train(previous)
    return gin,gout,objective.detach()/total,total


def interference(model, documents, policy):
    """Actual LM CE derivatives by effective token row, split input/output paths.

    Components are shared, so row derivatives are multiplied by the token's
    learned importance and summed for duplicate hashes. Membership selection is
    fixed from hashes, never from losses/gradients. At most 8x4 contributions per
    path are attributed. This is a bounded document-prefix diagnostic, not BPB.
    """
    v=model.vocabulary
    indices=v.indices.detach().cpu()
    observed=set()
    for doc in list(documents)[:policy['documents']]:
        observed.update(doc.ids[:min(policy['targets'],len(doc.ids)-1)])
    members={}
    for token,row in enumerate(indices.tolist()):
        if token not in observed: continue
        for bucket in set(row): members.setdefault(bucket,[]).append(token)
    buckets=[b for b in sorted(members) if len(members[b])>=2][:policy['shared_buckets']]
    chosen={b:members[b][:policy['tokens_per_bucket']] for b in buckets}
    gin,gout,objective,total=effective_path_gradients(model,documents,policy)
    rows={}
    for path,g in [('input',gin),('output',gout),('combined',gin+gout)]:
        bucket_rows=[]
        for bucket,tokens in chosen.items():
            contributions=[]
            for token in tokens:
                weight=v.importance[token][v.indices[token]==bucket].detach().sum()
                contributions.append(g[token]*weight)
            gstack=torch.stack(contributions); norms=gstack.norm(dim=1); denom=norms.sum()
            cos=[]
            for a,b in itertools.combinations(range(len(tokens)),2):
                if norms[a]>0 and norms[b]>0:
                    cos.append(float(F.cosine_similarity(gstack[a],gstack[b],dim=0)))
            bucket_rows.append(dict(bucket=bucket,tokens=tokens,norms=norms.tolist(),
                cancellation=float(1-gstack.sum(0).norm()/denom) if denom>0 else None,
                dominance=float(norms.max()/denom) if denom>0 else None,
                pairwise_cosines=cos,zero_contributions=int((norms==0).sum())))
        cos=[c for r in bucket_rows for c in r['pairwise_cosines']]
        cancellation=[r['cancellation'] for r in bucket_rows if r['cancellation'] is not None]
        dominance=[r['dominance'] for r in bucket_rows if r['dominance'] is not None]
        rows[path]=dict(buckets=bucket_rows,pairwise_cosine=distribution(torch.tensor(cos)) if cos else None,
                        mean_cancellation=sum(cancellation)/len(cancellation) if cancellation else None,
                        mean_dominance=sum(dominance)/len(dominance) if dominance else None)
    return dict(protocol='actual_lm_loss_effective_row_chain_rule_v1',documents=min(len(documents),policy['documents']),
                targets=total,mean_token_nll=float(objective),paths=rows,
                selected_buckets=len(buckets),
                limitation='Predetermined training-document prefixes and bounded hash-selected observed-input token subsets; not whole-corpus attribution',
                cancellation_definition='1 - norm(sum contributions)/sum(norm contributions)',
                dominance_definition='max(norm contributions)/sum(norm contributions)')
