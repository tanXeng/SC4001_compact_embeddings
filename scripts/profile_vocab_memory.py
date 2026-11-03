"""Bounded memory investigation; never launches corpus training."""
import argparse, json, statistics, time, types
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from scripts.vocab_scaling import load_settings, construct, prepared
from vocab_lm.train import train, accumulated_backward, optimizer_update
from vocab_lm.runtime import environment
from vocab_lm.accounting import tensor_bytes

class Trace(TorchDispatchMode):
 def __init__(self): self.rows=[]; self.high=0
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  out=func(*args,**(kwargs or {}))
  allocated=torch.cuda.memory_allocated()
  if allocated>self.high:
   self.high=allocated
   def desc(x):
    if isinstance(x,torch.Tensor):return dict(shape=list(x.shape),stride=list(x.stride()),bytes=x.numel()*x.element_size(),storage=x.untyped_storage().data_ptr())
    if isinstance(x,(list,tuple)):return [desc(a) for a in x]
    return str(x)[:60]
   self.snapshot=torch.cuda.memory_snapshot(); self.rows.append(dict(op=str(func),allocated=allocated,args=desc(args),output=desc(out)))
  return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--candidate',default='current',choices=['current','flatten','checkpoint']);a=p.parse_args()
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 s=load_settings('configs/vocab_scaling_toy.yaml'); m,_=construct(s,a.arm);m.cuda();v=m.vocabulary
 if a.candidate=='flatten':
  v.project=types.MethodType(lambda self,h: F.linear(F.linear(h.reshape(-1,self.hidden),self.b),self.a).reshape(*h.shape[:-1],self.vocab_size),v)
 if a.candidate=='checkpoint':
  from torch.utils.checkpoint import checkpoint
  v.project=types.MethodType(lambda self,h: checkpoint(lambda x:(x@self.b.T)@self.a.T,h,use_reentrant=False),v)
 data,_,_,identity,_=prepared(s,a.arm)
 from vocab_lm.config import TrainConfig
 cfg=TrainConfig(**s['training']);rows=[]
 # Same real-data sequence as original profiles, two ten-update sessions; shared loop initializes AdamW before peak reset at resume.
 state=train(m,data,cfg,validation_identity=identity,max_attempts=10,on_metrics=rows.append)
 rows.clear();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 state=train(m,data,cfg,validation_identity=identity,resume=state,max_attempts=10,on_metrics=rows.append)
 timed=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),median_seconds=statistics.median(r['update_seconds'] for r in rows[3:]),min_seconds=min(r['update_seconds'] for r in rows[3:]),max_seconds=max(r['update_seconds'] for r in rows[3:]),samples=[r['update_seconds'] for r in rows[3:]],bytes_per_second=sum(r['update_bytes'] for r in rows[3:])/sum(r['update_seconds'] for r in rows[3:]),tokens_per_second=sum(r['update_tokens'] for r in rows[3:])/sum(r['update_seconds'] for r in rows[3:]))
 opt=torch.optim.AdamW(m.parameters(),lr=cfg.learning_rate,betas=tuple(cfg.betas),eps=cfg.epsilon,weight_decay=cfg.weight_decay,foreach=False,fused=False);opt.load_state_dict(state['optimizer']);del state
 scaler=torch.amp.GradScaler('cuda',enabled=False)
 batches=[data.batch(cfg.microbatch_size) for _ in range(cfg.accumulation)]
 opt.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 saved=[]
 def pack(t):
  saved.append(dict(shape=list(t.shape),stride=list(t.stride()),bytes=t.numel()*t.element_size(),storage=t.untyped_storage().data_ptr()));return t
 trace=Trace()
 with trace,torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
  accumulated_backward(m,batches,device=torch.device('cuda'),scaler=scaler,fp16=False)
  optimizer_update(m,opt,scaler,cfg.clip_norm)
 out=dict(arm=a.arm,candidate=a.candidate,environment=environment(torch.device('cuda')),timing=timed,parameters=tensor_bytes(list(m.parameters())),gradients=tensor_bytes([p.grad for p in m.parameters()]),optimizer=tensor_bytes(opt.state_dict()),buffers=tensor_bytes(list(m.buffers())),trace_peak=torch.cuda.max_memory_allocated(),trace=trace.rows,peak_snapshot=trace.snapshot,saved_tensors=saved,batch_shapes=[list(b['inputs'].shape) for b in batches],process_gpu_memory='not measured')
 root=Path('results/vocab_scaling_memory');root.mkdir(exist_ok=True,parents=True);(root/f'{a.arm}_{a.candidate}.json').write_text(json.dumps(out,indent=2))
 print(a.arm,a.candidate,timed)
if __name__=='__main__':main()

