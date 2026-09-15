#!/usr/bin/env python3
"""Adversarial assumption tests on a frozen, trained neural model.

Cancellation units are an explicit construction, not learned or naturally
occurring mechanisms. Adding them preserves every clean prediction exactly.
"""
from pathlib import Path
from itertools import product
import json
import numpy as np
from redundancy_audit.core import AuditSpace,audit,minimal_edges,monotonicity_violations
from redundancy_audit.certificates import check_certificate
ROOT=Path(__file__).resolve().parents[1]

def summarize(table,n,r,b):
    es=minimal_edges(table,n);result=audit(AuditSpace(n,r,b),lambda a:bool(table[a]),seed=0)
    check=check_certificate(n,r,b,result.edges,result.transcript)
    bad,total=monotonicity_violations(table,n)
    return dict(n=n,r=r,b=b,true_edges=es,actual_rank=max(map(int.bit_count,es),default=0),
        monotone_violations=bad,adjacent_pairs=total,
        exact_recovery=set(result.edges)==set(es),conditional_certificate=check['valid'],
        globally_valid_certificate=check['valid'] and bad==0 and all(e.bit_count()<=r for e in es),
        **result.summary(),transcript=result.transcript)

def main():
    data=json.loads((ROOT/'results/neural_seed0.json').read_text())['positive_zero']
    x=np.array(list(product([-1.,1.],repeat=8)))
    h=np.maximum(x@np.array(data['first_weight']).T+data['first_bias'],0)
    logits=h@np.array(data['readout_weight'])+data['readout_bias'];base=logits>=0
    amplitude=1+float(np.abs(logits).max())
    edits=np.array([logits,logits-amplitude,logits+amplitude,logits])
    disagreement=((edits>=0)!=base[None,:]).mean(1);table=disagreement>.1
    assert table.tolist()==[False,True,True,False]
    native=np.array(data['disagreement'])>.1
    result=dict(cancellation=dict(amplitude=amplitude,clean_logit_change=0.,
        disagreement=disagreement.tolist(),**summarize(table,2,2,2)),
        underspecified_rank=summarize(native,8,2,2),
        exposed_rank_violation=summarize(native,8,2,8),
        correctly_specified=summarize(native,8,8,8))
    assert result['cancellation']['conditional_certificate'] and not result['cancellation']['exact_recovery']
    assert result['underspecified_rank']['conditional_certificate'] and not result['underspecified_rank']['exact_recovery']
    assert result['exposed_rank_violation']['status']=='rank_bound_violated'
    assert result['correctly_specified']['globally_valid_certificate'] and result['correctly_specified']['exact_recovery']
    (ROOT/'results/assumption_stress.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:(v['status'],v['exact_recovery']) for k,v in result.items()})
if __name__=='__main__':main()
