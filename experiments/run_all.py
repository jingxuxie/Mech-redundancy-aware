#!/usr/bin/env python3
"""CPU-only finite-domain experiments. PYTHONPATH=src python experiments/run_all.py
Truth is used only by evaluation metrics, never by discovery or stopping.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
import argparse, json, math, platform, sys, time
import numpy as np
import pandas as pd
from redundancy_audit.core import (AuditSpace, audit, hypergraph_oracle, mask_of,
    masks_upto, minimal_edges, monotonicity_violations)
from redundancy_audit.certificates import check_certificate, certificate_cover
from redundancy_audit.noise import BernoulliConfidenceOracle

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
METHODS=['greedy','random','sight','rc','exhaustive','pairs']

def write_json(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else str(x))+'\n')

@lru_cache(maxsize=32)
def space(n,r,b): return AuditSpace(n,r,b)

def graph(family,seed,n=12):
    rng=np.random.default_rng(seed)
    if family=='empty': es=[]
    elif family=='disjoint': es=[mask_of(range(i,i+3)) for i in (0,3,6,9)]
    elif family=='shared_core': es=[mask_of([0,1,j]) for j in range(2,10)]
    elif family=='random_sparse':
        es=list(map(int,rng.choice([mask_of(c) for c in combinations(range(n),3)],size=6,replace=False)))
    elif family=='mixed_order':
        es=[1,mask_of([1,2]),mask_of([2,3]),mask_of([4,5,6]),mask_of([5,6,7]),mask_of([8,9,10])]
    elif family=='dense_module': es=[mask_of(c) for c in combinations(range(6),3)]
    else: raise ValueError(family)
    perm=rng.permutation(n)
    return sorted(mask_of(perm[i] for i in range(n) if e&(1<<i)) for e in es)

def metrics(result,edges):
    truth,found=set(edges),set(result.edges)
    first_full=0 if not truth else None
    for rec in result.trace:
        if truth.issubset(rec['edges']):
            first_full=rec['queries']; break
    return dict(precision=len(truth&found)/len(found) if found else float(not truth),
                recall=len(truth&found)/len(truth) if truth else 1.,
                exact_recovery=truth==found,queries_first_full=first_full)

def combinatorial(seeds):
    start=time.perf_counter(); rows=[]; curves=[]
    for family in ['empty','disjoint','shared_core','random_sparse','mixed_order','dense_module']:
        for b in (3,4,6,8):
            for seed in range(seeds):
                es=graph(family,seed); oracle=hypergraph_oracle(es)
                for method in METHODS:
                    res=audit(space(12,3,b),oracle,method=method,seed=seed,max_queries=2000)
                    cert=check_certificate(12,3,b,res.edges,res.transcript)
                    rows.append(dict(family=family,n=12,r=3,b=b,seed=seed,method=method,
                        true_edges=len(es),max_degree=max((sum(bool(e&(1<<i)) for e in es) for i in range(12)),default=0),
                        **{k:v for k,v in res.summary().items() if k!='edges'},
                        **metrics(res,es),certificate_valid=cert['valid']))
                    for q in (10,20,40,80,160,320):
                        available=[t for t in res.trace if t['queries']<=q]
                        found=set(available[-1]['edges']) if available else set()
                        curves.append(dict(family=family,b=b,seed=seed,method=method,budget=q,
                            recall=len(set(es)&found)/len(es) if es else 1.))
            print('combinatorial',family,b,'elapsed',round(time.perf_counter()-start,1),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'combinatorial.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'budget_curves.csv',index=False)
    es=graph('disjoint',0); res=audit(space(12,3,6),hypergraph_oracle(es),seed=0)
    write_json('example_certificate.json',dict(n=12,r=3,b=6,true_edges=es,**res.summary(),
        transcript=res.transcript,trace=res.trace,
        certificate=check_certificate(12,3,6,res.edges,res.transcript)))

def theorem_check():
    candidates=masks_upto(4,3); rows=[]
    for selected in range(1<<len(candidates)):
        es=[e for i,e in enumerate(candidates) if selected&(1<<i)]
        if any(e&f in (e,f) for e,f in combinations(es,2)): continue
        for b in (3,4):
            sp=space(4,3,b); res=audit(sp,hypergraph_oracle(es),seed=selected)
            cov=certificate_cover(sp,es); cert=check_certificate(4,3,b,res.edges,res.transcript)
            assert res.edges==sorted(es) and cert['valid'] and cov['optimum'] is not None
            assert cov['optimum']+len(es)<=res.query_count<=cov['greedy_bound']+1e-8
            rows.append(dict(antichain=selected,b=b,edges=len(es),query_count=res.query_count,**cov))
    pd.DataFrame(rows).to_csv(OUT/'theorem_checks.csv',index=False)
    write_json('theorem_check_summary.json',dict(cases=len(rows),distinct_antichains=len(rows)//2,
        passed=True,scope='Exhaustive finite checks supplement, not replace, written proofs.'))
    print('theorem checks',len(rows),flush=True)

def locality(seeds):
    rng=np.random.default_rng(20260915); trials=max(2000,seeds*100); rows=[]
    for b in (2,3,4,6,8,12,16):
        p=math.comb(b,3)/math.comb(32,3) if b>=3 else 0.
        for q in (1,10,50,200,1000):
            # Intersection size with a fixed triple in independently sampled
            # uniform size-b pools; not Bernoulli draws from the predicted p.
            hit=(rng.hypergeometric(3,29,b,size=(trials,q))==3).any(axis=1)
            mean=float(hit.mean())
            rows.append(dict(n=32,order=3,b=b,queries=q,repetitions=trials,
                empirical_detection=mean,standard_error=math.sqrt(mean*(1-mean)/trials),
                predicted_detection=1-(1-p)**q,single_query_probability=p,
                zero_error_lower_bound=math.ceil(1/p) if p else None))
    pd.DataFrame(rows).to_csv(OUT/'locality.csv',index=False)

def noise(seeds):
    rows=[]; truth=[7,56]; sp=space(8,3,4); binary=hypergraph_oracle(truth)
    for gamma in (.1,.2):
        for seed in range(max(20,seeds)):
            for label in ('confidence','fixed_32'):
                oracle=BernoulliConfidenceOracle(lambda a:.5+gamma if binary(a) else .5-gamma,
                    seed=seed,fixed_samples=32 if label=='fixed_32' else None)
                res=audit(sp,oracle,seed=seed,max_queries=1000)
                incorrect=sum(y is not None and y!=binary(a) for a,_,y in oracle.records)
                cert=check_certificate(8,3,4,res.edges,res.transcript)
                rows.append(dict(gamma=gamma,seed=seed,method=label,status=res.status,
                    total_samples=oracle.total_samples,mask_requests=oracle.calls,
                    accepted_labels=res.query_count,incorrect_labels=incorrect,**metrics(res,truth),
                    false_completion=res.status=='complete' and set(res.edges)!=set(truth),
                    certificate_valid=cert['valid'],false_certificate=cert['valid'] and set(res.edges)!=set(truth)))
    pd.DataFrame(rows).to_csv(OUT/'noise.csv',index=False)

def neural(seeds,steps):
    import torch
    from torch import nn
    import torch.nn.functional as F
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    all_x=torch.tensor(list(product([-1.,1.],repeat=8)),dtype=torch.float32)
    y=(all_x[:,:4].sum(1)>=0).float(); n,group_width=8,4
    rows=[]; models=[]; artifact={}
    for seed in range(min(10,seeds)):
        for positive_readout in (True,False):
            torch.manual_seed(seed); first=nn.Linear(8,n*group_width)
            raw_weight=nn.Parameter(torch.full((n*group_width,),-1.0) if positive_readout else torch.randn(n*group_width)*.1)
            bias=nn.Parameter(torch.tensor(-1.))
            opt=torch.optim.Adam(list(first.parameters())+[raw_weight,bias],lr=.02)
            for step in range(steps):
                h=F.relu(first(all_x)); gm=(torch.rand(len(all_x),n)>.2).float()
                h=h*gm.repeat_interleave(group_width,dim=1)
                w=F.softplus(raw_weight) if positive_readout else raw_weight
                loss=F.binary_cross_entropy_with_logits(h@w+bias,y)
                opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():
                h=F.relu(first(all_x)); w=F.softplus(raw_weight) if positive_readout else raw_weight
                base=h@w+bias>=0; clean_acc=float((base==y.bool()).float().mean())
                contributions=(h*w).reshape(256,n,group_width).sum(2).numpy()
                full_logits=(h@w+bias).numpy()
                mask_arr=np.array([[bool(m&(1<<i)) for i in range(n)] for m in range(1<<n)])
                for patch in ('zero','mean'):
                    delta=contributions if patch=='zero' else contributions-contributions.mean(0)
                    edited_logits=full_logits[None,:]-mask_arr@delta.T
                    disagreement=((edited_logits>=0)!=base.numpy()[None,:]).mean(1)
                    table=disagreement>.10;table[0]=False
                    es=minimal_edges(table,n);violations,adjacent=monotonicity_violations(table,n)
                    readout='positive' if positive_readout else 'signed'
                    models.append(dict(seed=seed,readout=readout,patch=patch,clean_accuracy=clean_acc,
                        threshold=.1,training_steps=steps,min_edges=len(es),
                        max_order=max(map(int.bit_count,es),default=0),
                        monotone_violations=violations,adjacent_pairs=adjacent,monotone=violations==0))
                    # r=n: no ground-truth rank information is supplied to discovery.
                    for method in ('greedy','random','sight','exhaustive'):
                        res=audit(space(n,n,n),lambda a:bool(table[a]),method=method,seed=seed)
                        cert=check_certificate(n,n,n,res.edges,res.transcript)
                        pred=np.array([any(e&a==e for e in res.edges) for a in range(1<<n)])
                        rows.append(dict(seed=seed,readout=readout,patch=patch,method=method,
                            monotone=violations==0,query_count=res.query_count,status=res.status,**metrics(res,es),
                            all_mask_accuracy=float((pred==table).mean()),conditional_certificate=cert['valid'],
                            globally_valid_certificate=cert['valid'] and violations==0))
                    if seed==0:
                        artifact[readout+'_'+patch]=dict(disagreement=disagreement.tolist(),true_edges=es,
                            first_weight=first.weight.numpy().tolist(),first_bias=first.bias.numpy().tolist(),
                            readout_weight=w.numpy().tolist(),readout_bias=float(bias))
            print('neural',seed,positive_readout,'accuracy',clean_acc,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'neural_audits.csv',index=False)
    pd.DataFrame(models).to_csv(OUT/'neural_models.csv',index=False)
    write_json('neural_seed0.json',artifact)

def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
    if (OUT/'combinatorial.csv').exists():
        df=pd.read_csv(OUT/'combinatorial.csv');fig,ax=plt.subplots(figsize=(6.2,3.8))
        for method in ('greedy','random','sight','rc','exhaustive'):
            v=df[df.method==method].groupby(['b','seed']).query_count.mean().groupby('b');m,s=v.mean(),v.sem()
            ax.errorbar(m.index,m,yerr=s,marker='o',capsize=3,label=method)
        ax.set(xlabel='Maximum intervention size b',ylabel='Distinct oracle queries to certificate',xticks=[3,4,6,8])
        ax.legend(ncol=2,fontsize=9);fig.tight_layout()
        fig.savefig(figdir/'query_cost.pdf');fig.savefig(figdir/'query_cost.png',dpi=180);plt.close(fig)
        g=df[(df.method=='greedy')&(df.true_edges>0)]
        fig,ax=plt.subplots(figsize=(6.2,3.8));v=g.groupby('b')[['queries_first_full','query_count']].mean()
        ax.plot(v.index,v.queries_first_full,marker='o',label='All edges first discovered (evaluator)')
        ax.plot(v.index,v.query_count,marker='s',label='Completeness certified (algorithm)')
        ax.set(xlabel='Maximum intervention size b',ylabel='Mean distinct queries',xticks=[3,4,6,8])
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figdir/'discovery_vs_certification.pdf')
        fig.savefig(figdir/'discovery_vs_certification.png',dpi=180);plt.close(fig)
    if (OUT/'locality.csv').exists():
        df=pd.read_csv(OUT/'locality.csv');fig,ax=plt.subplots(figsize=(6.2,3.8))
        for b in (3,4,8,16):
            g=df[df.b==b];ax.plot(g.queries,g.predicted_detection,label=f'b={b}')
            ax.scatter(g.queries,g.empirical_detection,s=18)
        ax.set(xscale='log',xlabel='Random intervention queries',ylabel='Probability of finding the hidden triple',ylim=(-.03,1.03))
        ax.legend();fig.tight_layout();fig.savefig(figdir/'locality.pdf')
        fig.savefig(figdir/'locality.png',dpi=180);plt.close(fig)
    if (OUT/'neural_audits.csv').exists():
        df=pd.read_csv(OUT/'neural_audits.csv');g=df[(df.readout=='positive')&(df.patch=='zero')]
        v=g.groupby('method').query_count;m,s=v.mean(),v.sem();fig,ax=plt.subplots(figsize=(6.2,3.8))
        ax.bar(m.index,m,yerr=s,capsize=4);ax.set(ylabel='Queries to full-mask certificate',xlabel='Audit method')
        fig.tight_layout();fig.savefig(figdir/'neural_queries.pdf');fig.savefig(figdir/'neural_queries.png',dpi=180);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=['all','combinatorial','theory','locality','noise','neural','plot'],default='all')
    p.add_argument('--seeds',type=int,default=20);p.add_argument('--steps',type=int,default=1200)
    args=p.parse_args();start=time.perf_counter()
    if args.suite in ('all','theory'):theorem_check()
    if args.suite in ('all','locality'):locality(args.seeds)
    if args.suite in ('all','combinatorial'):combinatorial(args.seeds)
    if args.suite in ('all','noise'):noise(args.seeds)
    if args.suite in ('all','neural'):neural(args.seeds,args.steps)
    if args.suite in ('all','plot'):plot()
    import scipy,matplotlib
    write_json('environment_'+args.suite+'.json',dict(python=sys.version,platform=platform.platform(),
        numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,
        command=sys.argv,elapsed_seconds=time.perf_counter()-start,device='CPU'))
    print('completed',args.suite,'seconds',time.perf_counter()-start,flush=True)
if __name__=='__main__':main()
