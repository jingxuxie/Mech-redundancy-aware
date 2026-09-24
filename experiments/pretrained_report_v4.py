"""Expanded CPU GPT-2 audit. Shards are computational partitions, not replicates.
Discovery: 128 prompts. Validation: 384 independent prompts. ABC means: 128
independent template-matched prompts, frozen before all interventions.
All head replacements act before c_proj, at all positions. No lattice enumeration.
"""
from pathlib import Path
import argparse,itertools,json,time,sys,platform,hashlib
import numpy as np
import torch
from transformers import AutoTokenizer,GPT2LMHeadModel
HEADS=[(9,9),(9,6),(10,0),(10,7),(11,10),(7,9),(8,10),(8,6),(7,3),(5,5),(5,9),(6,9),(5,8),(0,1),(0,10),(3,0),(4,11),(2,2),(11,2),(10,6),(10,10),(10,2),(9,7),(10,1),(11,9),(9,0)]
NAMES=['John','Mary','Tom','Sarah','James','Emily','David','Alice','Michael','Anna','Robert','Emma','Daniel','Laura','Mark','Julia']
PLACES=['store','park','office','school','garden','station','museum','hospital']
OBJECTS=['book','drink','letter','gift','bag','ball','note','ticket']
REVISION='607a30d783dfa663caf39e06633721c8d4cfcd7e'
def dataset(seed,count,abc=False):
 rng=np.random.default_rng(seed);out=[]
 for k in range(count):
  tid=k%16;a,b,c=rng.choice(NAMES,3,replace=False);loc=PLACES[tid//2];obj=OBJECTS[tid//2]
  first,second=(a,b) if tid%2==0 else (b,a);repeated=c if abc else b
  text=f'When {first} and {second} went to the {loc}, {repeated} gave the {obj} to'
  out.append(dict(text=text,io=a,subject=b,template=tid))
 return out
class Evaluator:
 def __init__(self,patch):
  torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(924)
  self.patch=patch;self.tokenizer=AutoTokenizer.from_pretrained('openai-community/gpt2',revision=REVISION);self.tokenizer.pad_token=self.tokenizer.eos_token
  self.model=GPT2LMHeadModel.from_pretrained('openai-community/gpt2',revision=REVISION,use_safetensors=True,attn_implementation='eager').eval()
  assert all(len(self.tokenizer.encode(' '+a,add_special_tokens=False))==1 for a in NAMES)
  self.records=dataset(92401,128)+dataset(92402,384);self.calib=dataset(92403,128,True)
  alltext=[r['text'] for r in self.records+self.calib];enc=self.tokenizer(alltext,return_tensors='pt',padding=True)
  self.tokens=enc.input_ids;self.attention=enc.attention_mask;self.last=self.attention.sum(-1)-1
  self.tids=torch.tensor([r['template'] for r in self.records+self.calib]);L=self.tokens.shape[1]
  self.means=torch.zeros(12,16,L,768);self.collect=False;self.current=None;self.current_masks=None
  self.hooks=[b.attn.c_proj.register_forward_pre_hook(self.hook(i)) for i,b in enumerate(self.model.transformer.h)]
  self.directions=self.model.lm_head.weight.detach()[torch.tensor([self.tokenizer.encode(' '+r['io'],add_special_tokens=False)[0] for r in self.records])]-self.model.lm_head.weight.detach()[torch.tensor([self.tokenizer.encode(' '+r['subject'],add_special_tokens=False)[0] for r in self.records])]
  self.collect=True
  with torch.inference_mode():
   for tid in range(16):
    ids=torch.where((self.tids==tid)&(torch.arange(len(self.tids))>=512))[0];self.current=ids
    self.model.transformer(input_ids=self.tokens[ids],attention_mask=self.attention[ids],use_cache=False)
  self.collect=False;self.current_masks=None
  self.clean=self.evaluate([0],np.arange(512))[0]
  # Test zero-hook semantics against the standard head_mask interface.
  saved=self.patch;self.patch='zero';ids=np.arange(8);via_hook=self.evaluate([1],ids)[0];self.current_masks=None
  hm=torch.ones(12,12);hm[HEADS[0]]=0
  with torch.inference_mode():
   z=self.model.transformer(input_ids=self.tokens[ids],attention_mask=self.attention[ids],head_mask=hm,use_cache=False).last_hidden_state
   standard=(z[torch.arange(8),self.last[ids]]*self.directions[ids]).sum(-1).numpy()
  self.hook_error=float(np.max(np.abs(via_hook-standard)));assert self.hook_error<1e-4,self.hook_error;self.patch=saved
 def hook(self,layer):
  def run(module,args):
   z=args[0]
   if self.collect:
    tid=int(self.tids[self.current[0]]);self.means[layer,tid]=z.detach().mean(0);return None
   if self.current_masks is None:return None
   mask=torch.zeros(len(z),12,dtype=torch.bool)
   for j,(l,h) in enumerate(HEADS):
    if l==layer:mask[:,h]=(self.current_masks&(1<<j))!=0
   if not mask.any():return None
   zz=z.reshape(len(z),z.shape[1],12,64)
   base=torch.zeros_like(zz) if self.patch=='zero' else self.means[layer,self.tids[self.current]].reshape_as(zz)
   return (torch.where(mask[:,None,:,None],base,zz).reshape_as(z),)
  return run
 @torch.inference_mode()
 def evaluate(self,masks,ids):
  ids=np.asarray(ids);out=[]
  for m in masks:
   parts=[]
   for k in range(0,len(ids),32):
    ix=torch.tensor(ids[k:k+32]);self.current=ix;self.current_masks=torch.full((len(ix),),int(m),dtype=torch.int64)
    z=self.model.transformer(input_ids=self.tokens[ix],attention_mask=self.attention[ix],use_cache=False).last_hidden_state
    parts.append((z[torch.arange(len(ix)),self.last[ix]]*self.directions[ix]).sum(-1).numpy())
   out.append(np.concatenate(parts))
  self.current_masks=None;return np.stack(out)
def planned():
 discovery=[1<<i for i in range(26)]+[(1<<i)|(1<<j) for i,j in itertools.combinations(range(26),2)]
 rng=np.random.default_rng(92404);pairs=[]
 for p in (.2,.5):
  for draw in range(128):
   i=int(rng.integers(26));a=sum(1<<j for j in range(26) if j!=i and rng.random()<p)
   pairs.append(dict(p=p,draw=draw,lower=a,upper=a|(1<<i),head_index=i,layer=HEADS[i][0],head=HEADS[i][1]))
 validation=[int(a) for a in rng.integers(0,1<<26,size=96)]
 return discovery,pairs,validation
def shard(args):
 start=time.perf_counter();out=args.output;out.mkdir(parents=True,exist_ok=True);ev=Evaluator(args.patch)
 disc,pairs,val=planned();parts={};d=disc[args.shard::4];pp=pairs[args.shard::4];a=sorted({r[x] for r in pp for x in ('lower','upper')});v=val[args.shard::4]
 for name,masks,ids in [('discovery',d,np.arange(128)),('adjacent',a,np.arange(128,512)),('validation',v,np.arange(128,512))]:
  rows=[]
  for k,m in enumerate(masks):
   rows.append(ev.evaluate([m],ids)[0])
   if k%16==0:print(args.patch,args.shard,name,k,len(masks),round(time.perf_counter()-start,1),flush=True)
  np.savez_compressed(out/(name+'.npz'),masks=np.array(masks,dtype=np.int64),logits=np.stack(rows),clean=ev.clean[ids],prompt_ids=ids)
 meta=dict(patch=args.patch,shard=args.shard,model_revision=REVISION,heads=HEADS,prompts=ev.records,calibration_prompts=ev.calib,pairs=pp,
  hook_check_max_error=ev.hook_error,clean_accuracy=float(np.mean(ev.clean>0)),elapsed_seconds=time.perf_counter()-start,
  torch=torch.__version__,python=sys.version,platform=platform.platform(),mean_policy='ABC template-matched mean pre-projection head vectors at each absolute position; fixed independently before interventions',
  note='128 discovery prompts, 384 independent validation prompts, 128 separate ABC mean-calibration prompts. All 16 templates fixed; generalization across templates is not claimed.')
 (out/'metadata.json').write_text(json.dumps(meta,indent=2));(out/'executed_source.py').write_text(Path(__file__).read_text())
def finalize(args):
 start=time.perf_counter();ev=Evaluator(args.patch);out=args.output;out.mkdir(parents=True,exist_ok=True)
 tables={};meta=None
 for path in args.input.rglob('metadata.json'):
  m=json.loads(path.read_text())
  if m['patch']!=args.patch:continue
  meta=m
  for name in ('discovery','adjacent','validation'):
   z=np.load(path.parent/(name+'.npz'));t=tables.setdefault(name,{})
   for a,v in zip(z['masks'],z['logits']):t[int(a)]=v
 assert len(tables['discovery'])==351
 transcript=[];K=[];discovery_rejection=None
 for a,v in sorted(tables['discovery'].items(),key=lambda x:(x[0].bit_count(),x[0])):
  y=bool(np.mean((v>0)!=(ev.clean[:128]>0))>.1);pred=any(a&e==e for e in K)
  if pred and not y:
   discovery_rejection=dict(mask=a,label=y,prediction=pred);break
  if y and not pred:K.append(a)
  transcript.append([a,y])
 # Fixed reference and matching prefix; stop at actual contradiction, never silently repair it.
 g=lambda a:any(a&e==e for e in K);observed={0}|{a for a,_ in transcript};N=[a for a,y in transcript if not y]
 def model_label(a):
  v=tables['discovery'].get(a)
  if v is None:v=ev.evaluate([a],np.arange(128))[0]
  return bool(np.mean((v>0)!=(ev.clean[:128]>0))>.1),v
 runs=[];saved=[]
 # S0 finite candidates = observed nodes, their neighbors, and reference edges.
 # Both constant backgrounds are tested; this is not an S1 optimization claim.
 for policy in ('uniform','predicted_positive','finite_S0_positive','finite_S0_negative'):
  seen=set(observed);added=[];rng=np.random.default_rng(92405)
  for step in range(8):
   if policy.startswith('finite'):
    pool={a^(1<<i) for a in seen for i in range(26)}|set(K)
    target=policy.endswith('positive');pool=sorted(a for a in pool if a not in seen and g(a)==target)
   else:pool=[]
   if pool:a=int(rng.choice(pool))
   else:
    a=None
    for _ in range(10000):
     x=int(rng.integers(1,1<<26))
     if x not in seen and (policy!='predicted_positive' or g(x)):a=x;break
    if a is None:
     # No estimated failure mass is a documented fallback, not silent success.
     while True:
      a=int(rng.integers(1,1<<26))
      if a not in seen:break
   y,v=model_label(a);pred=g(a);added.append(dict(mask=a,label=y,prediction=pred));saved.append((a,v));seen.add(a)
   if y!=pred:break
  runs.append(dict(policy=policy,additional=added,rejected=bool(added and added[-1]['label']!=added[-1]['prediction']),queries=len(added)))
 vm=sorted(tables['validation']);vl=np.stack([tables['validation'][a] for a in vm]);vd=(vl>0)!=(ev.clean[128:]>0)
 report=dict(patch=args.patch,K=K,transcript=transcript,discovery_rejection=discovery_rejection,discovery_queries=len(transcript)+(discovery_rejection is not None),
  b=26,tau=.1,mask_distribution='uniform masks on 26-head intervention universe',runs=runs,
  population_note='Published head set supplies the intervention universe, not the derived minimal-failure explanation or the full published edge/position circuit.',elapsed_seconds=time.perf_counter()-start)
 (out/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'validation.npz',masks=np.array(vm),disagreements=vd,logits=vl,clean=ev.clean[128:],g=np.array([g(a) for a in vm]))
 if saved:np.savez_compressed(out/'falsification_logits.npz',masks=np.array([a for a,v in saved]),logits=np.stack([v for a,v in saved]),clean=ev.clean[:128])
 (out/'executed_source.py').write_text(Path(__file__).read_text());print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['shard','finalize']);p.add_argument('--patch',choices=['zero','mean'],required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--input',type=Path,default=Path('incoming'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 (shard if a.mode=='shard' else finalize)(a)
