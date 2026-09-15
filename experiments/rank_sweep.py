#!/usr/bin/env python3
"""Same six disjoint failure pairs, different DECLARED rank assumptions.

The hidden function never changes. All labels and the ground-truth certificate
optimization are exact. The optimizer is evaluation-only, not used by the audit.
"""
from pathlib import Path
import argparse, json, math, time
import numpy as np
import pandas as pd
from redundancy_audit.core import AuditSpace, audit, hypergraph_oracle
from redundancy_audit.certificates import certificate_cover, check_certificate
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=20)
    p.add_argument('--radius-only',action='store_true')
    args=p.parse_args();radius(args.seeds)
    if args.radius_only: return
    rows=[];covers=[];start=time.perf_counter()
    for r in (2,3,4,5,6,12):
        sp=AuditSpace(12,r,6);es=[3<<(2*j) for j in range(6)]
        cover=certificate_cover(sp,es,time_limit=15)
        covers.append(dict(n=12,pairs=6,r=r,b=6,**cover))
        for seed in range(args.seeds):
            rng=np.random.default_rng(seed);perm=rng.permutation(12)
            truth=[sum(1<<int(perm[i]) for i in range(12) if e&(1<<i)) for e in es]
            for method in ('greedy','random','sight','exhaustive'):
                res=audit(sp,hypergraph_oracle(truth),seed=seed,method=method)
                cert=check_certificate(12,r,6,res.edges,res.transcript)
                assert set(res.edges)==set(truth) and cert['valid']
                first=next(t['queries'] for t in res.trace if set(truth).issubset(t['edges']))
                rows.append(dict(n=12,r=r,b=6,seed=seed,method=method,
                    query_count=res.query_count,first_full=first,
                    negative_queries=sum(not y for _,y in res.transcript),
                    outer_negative=res.outer_negative,certified=True))
        print('rank',r,'certificate',cover,'elapsed',time.perf_counter()-start,flush=True)
    out=ROOT/'results';pd.DataFrame(rows).to_csv(out/'rank_sweep.csv',index=False)
    pd.DataFrame(covers).to_csv(out/'rank_certificate_covers.csv',index=False)
    (out/'environment_rank.json').write_text(json.dumps(dict(seeds=args.seeds,elapsed_seconds=time.perf_counter()-start),indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.2,3.8));df=pd.DataFrame(rows)
    for method in ('greedy','random','sight','exhaustive'):
        g=df[df.method==method].groupby('r').query_count
        ax.errorbar(g.mean().index,g.mean(),yerr=g.sem(),marker='o',capsize=3,label=method)
    ax.set(xlabel='Declared maximum failure-set order r',ylabel='Distinct queries to completeness',xticks=[2,3,4,5,6,12])
    ax.legend(ncol=2,fontsize=9);fig.tight_layout();fig.savefig(ROOT/'figures/rank_sweep.pdf')
    fig.savefig(ROOT/'figures/rank_sweep.png',dpi=180);plt.close(fig)
def radius(seeds):
    rows=[];truth=[3<<(2*j) for j in range(6)]
    for b in (2,3,4,5,6):
        sp=AuditSpace(12,12,b)
        for seed in range(seeds):
            res=audit(sp,hypergraph_oracle(truth),seed=seed)
            expected=sum(math.comb(6,k)*2**k for k in range(b+1,7))
            assert res.unresolved==expected and set(res.edges)==set(truth)
            assert (res.status=='complete')==(b==6)
            rows.append(dict(b=b,seed=seed,queries=res.query_count,
                unresolved=res.unresolved,predicted_unresolved=expected,status=res.status))
    pd.DataFrame(rows).to_csv(ROOT/'results/radius_sweep.csv',index=False)
    print('radius sweep',len(rows),'runs',flush=True)

if __name__=='__main__':main()
