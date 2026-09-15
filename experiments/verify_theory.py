#!/usr/bin/env python3
"""Independent finite-hypothesis verification of the radius and certificate theorems.

Unlike the audit, this verifier knows every alternative function. It solves a
separate minimum distinguishing-transcript ILP, not just the negative-cover ILP.
This is evaluation only; finite checks are not a substitute for the proofs.
"""
from pathlib import Path
from itertools import combinations
import json, time
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from redundancy_audit.core import AuditSpace, audit, masks_upto, hypergraph_oracle
from redundancy_audit.certificates import certificate_cover, check_certificate
ROOT=Path(__file__).resolve().parents[1]

def main():
    start=time.perf_counter();n=4;all_candidates=masks_upto(n,n);families=[]
    for selected in range(1<<len(all_candidates)):
        edges=[e for i,e in enumerate(all_candidates) if selected&(1<<i)]
        if any(e&f in (e,f) for e,f in combinations(edges,2)):continue
        table=np.array([hypergraph_oracle(edges)(a) for a in range(1<<n)])
        families.append((edges,table))
    rows=[];solved=0
    for r in range(1,n+1):
        hypotheses=[(es,t) for es,t in families if max(map(int.bit_count,es),default=0)<=r]
        for b in range(1,n+1):
            sp=AuditSpace(n,r,b)
            signatures=np.array([[table[a] for a in sp.queries] for _,table in hypotheses])
            for j,(es,table) in enumerate(hypotheses):
                rho=max(map(int.bit_count,es),default=0)
                alpha=max(a.bit_count() for a in range(1<<n) if not table[a])
                radius=max(rho,min(r,alpha));differences=signatures!=signatures[j]
                matches=~differences.any(axis=1);identified=matches.sum()==1
                assert identified==(b>=radius)
                row=dict(n=n,r=r,b=b,hypothesis=j,edges=len(es),actual_rank=rho,
                         independence_number=alpha,minimum_radius=radius,
                         identifiable=identified,indistinguishable_hypotheses=int(matches.sum()))
                if identified:
                    matrix=differences[np.arange(len(hypotheses))!=j].astype(float)
                    sol=milp(np.ones(len(sp.queries)),integrality=np.ones(len(sp.queries)),
                        bounds=Bounds(0,1),constraints=LinearConstraint(matrix,1,np.inf))
                    assert sol.status==0
                    minimum_transcript=int(round(sol.fun));cover=certificate_cover(sp,es)
                    assert minimum_transcript==cover['optimum']+len(es)
                    res=audit(sp,hypergraph_oracle(es),seed=j)
                    assert res.edges==sorted(es) and check_certificate(n,r,b,res.edges,res.transcript)['valid']
                    assert minimum_transcript<=res.query_count<=cover['greedy_bound']+1e-8
                    row.update(minimum_transcript=minimum_transcript,negative_cover=cover['optimum'],
                               greedy_queries=res.query_count,greedy_upper_bound=cover['greedy_bound'])
                    solved+=1
                rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT/'results/independent_theory_checks.csv',index=False)
    summary=dict(radius_cases=len(rows),minimum_transcript_optimizations=solved,
                 antichains=len(families),passed=True,elapsed_seconds=time.perf_counter()-start,
                 scope='All nonconstant-one monotone functions on four components; all declared ranks and query ceilings 1..4.')
    (ROOT/'results/independent_theory_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary,flush=True)
if __name__=='__main__':main()
