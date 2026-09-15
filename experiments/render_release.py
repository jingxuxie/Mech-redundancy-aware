#!/usr/bin/env python3
"""Build manuscript figures directly from the shipped measured plotting inputs.

No model retraining is needed. Run `make experiments` to regenerate raw records
and use run_all.py/summarize.py to independently recompute these measurements.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]

def main():
    data = json.loads((ROOT/'results/figure_data.json').read_text())
    out = ROOT/'figures'; out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2,3.8))
    for name, v in data['query_cost'].items():
        ax.errorbar(v['x'], v['mean'], yerr=v['sem'], marker='o', capsize=3, label=name)
    ax.set(xlabel='Maximum intervention size b', ylabel='Distinct oracle queries to certificate', xticks=[3,4,6,8])
    ax.legend(ncol=2, fontsize=9)
    save(fig,out,'query_cost')
    v=data['rank_gap'];fig,ax=plt.subplots(figsize=(6.2,3.8))
    ax.plot(v['x'],v['discovery'],marker='o',label='All six pairs discovered (evaluator)')
    ax.plot(v['x'],v['certification'],marker='s',label='Completeness certified (algorithm)')
    ax.set(xlabel='Declared maximum failure-set order r',ylabel='Mean distinct oracle queries',xticks=[2,3,4,5,6,12])
    ax.legend(fontsize=9)
    save(fig,out,'rank_discovery_gap')
    fig,ax=plt.subplots(figsize=(6.2,3.8))
    for b,v in data['locality'].items():
        ax.plot(v['x'],v['predicted'],label=f'b={b}')
        ax.scatter(v['x'],v['observed'],s=18)
    ax.set(xscale='log',xlabel='Random intervention queries',ylabel='Probability of finding the hidden triple',ylim=(-.03,1.03))
    ax.legend()
    save(fig,out,'locality')

def save(fig,out,name):
    fig.tight_layout();fig.savefig(out/f'{name}.pdf');fig.savefig(out/f'{name}.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
