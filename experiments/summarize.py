#!/usr/bin/env python3
"""Generate manuscript tables and exact summary numbers from released raw records."""
from pathlib import Path
import json, platform, sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';TABLES=ROOT/'paper/tables'
TABLES.mkdir(parents=True,exist_ok=True)
LABELS=dict(greedy='Coverage-directed',random='Random + deletion',sight='SIGHT-style',rc='RC-style',exhaustive='Increasing order',pairs='Pairs only')

def tex_table(name,header,rows,columns):
    text='\\begin{tabular}{'+columns+'}\n\\toprule\n'+' & '.join(header)+' \\\\\n\\midrule\n'
    text+=''.join(' & '.join(map(str,row))+' \\\\\n' for row in rows)
    text+='\\bottomrule\n\\end{tabular}\n';(TABLES/name).write_text(text)

def pm(x):return f'${x.mean():.1f} \\pm {x.sem():.1f}$'

def main():
    df=pd.read_csv(OUT/'combinatorial.csv');rank=pd.read_csv(OUT/'rank_sweep.csv')
    cover=pd.read_csv(OUT/'rank_certificate_covers.csv');nn=pd.read_csv(OUT/'neural_audits.csv')
    models=pd.read_csv(OUT/'neural_models.csv');noise=pd.read_csv(OUT/'noise.csv')
    theory=json.loads((OUT/'independent_theory_summary.json').read_text())
    byseed=df.groupby(['method','b','seed']).query_count.mean()
    methods=['greedy','random','sight','rc','exhaustive']
    rows=[[LABELS[m]]+[pm(byseed.loc[m,b]) for b in (3,4,6,8)] for m in methods]
    tex_table('queries.tex',['Method','$b=3$','$b=4$','$b=6$','$b=8$'],rows,'lrrrr')
    rows=[]
    for r in (2,3,4,5,6,12):
        g=rank[(rank.r==r)&(rank.method=='greedy')];c=cover[cover.r==r].iloc[0]
        rows.append([r,int(c.optimum),f'{g.first_full.mean():.2f}',pm(g.query_count),int(c.optimum)+6])
    tex_table('rank.tex',['Declared $r$','Min. negative cover','Discovery','Certification','Min. certificate'],rows,'rrrrr')
    rows=[]
    for readout,patch in [('positive','zero'),('positive','mean'),('signed','zero'),('signed','mean')]:
        mm=models[(models.readout==readout)&(models.patch==patch)]
        row=[readout.capitalize()+' / '+patch,f'{mm.min_edges.mean():.1f}',f'{int(mm.max_order.min())}--{int(mm.max_order.max())}']
        row +=[f'{nn[(nn.readout==readout)&(nn.patch==patch)&(nn.method==m)].query_count.mean():.1f}' for m in ['greedy','random','sight','exhaustive']]
        rows.append(row)
    tex_table('neural.tex',['Readout / patch','Edges','Max. rank','Coverage','Random','SIGHT','Increasing'],rows,'lrrrrrr')
    rows=[]
    for gamma in (.1,.2):
        for m in ['confidence','fixed_32']:
            g=noise[(noise.gamma==gamma)&(noise.method==m)]
            rows.append([gamma,'Confidence' if m=='confidence' else 'Fixed 32',
                f'{g.mask_requests.mean():.1f}',f'{g.total_samples.mean():,.0f}',
                f'{int(g.exact_recovery.sum())}/20',f'{int(g.certificate_valid.sum())}/20',f'{int(g.false_certificate.sum())}/20'])
    tex_table('noise.tex',['Margin','Labels','Masks','Input samples','Exact','Checked cert.','False cert.'],rows,'rlrrrrr')
    radius=pd.read_csv(OUT/'radius_sweep.csv');rows=[]
    for b,g in radius.groupby('b'):
        rows.append([b,f'{g.queries.mean():.1f}',int(g.unresolved.iloc[0]),'Complete' if b==6 else 'Locality limited'])
    tex_table('radius.tex',['Ceiling $b$','Queries','Residual candidates','Output'],rows,'rrrl')
    families=df.groupby(['family','b','method']).query_count.agg(['mean','std','count'])
    families.to_csv(OUT/'query_summary.csv')
    neural_summary=nn.groupby(['readout','patch','method']).agg(queries=('query_count','mean'),query_sd=('query_count','std'),exact=('exact_recovery','mean'),accuracy=('all_mask_accuracy','mean'))
    neural_summary.to_csv(OUT/'neural_summary.csv')
    p=df.pivot(index=['family','b','seed'],columns='method',values='query_count')
    summary=dict(date='2026-09-15',theory=theory,
        combinatorial_runs=len(df),certified_methods={m:dict(runs=int((df.method==m).sum()),exact=int(df[df.method==m].exact_recovery.sum()),certified=int(df[df.method==m].certificate_valid.sum())) for m in methods},
        paired_greedy_vs_random=dict(wins=int((p.greedy<p.random).sum()),ties=int((p.greedy==p.random).sum()),losses=int((p.greedy>p.random).sum()),mean_query_reduction_fraction=float(1-p.greedy.mean()/p.random.mean()),mean_per_instance_ratio=float((p.greedy/p.random).mean())),
        query_means={str(b):{m:float(df[(df.b==b)&(df.method==m)].query_count.mean()) for m in methods} for b in (3,4,6,8)},
        rank_runs=len(rank),radius_runs=len(radius),trained_models=int(models[['seed','readout']].drop_duplicates().shape[0]),neural_conditions=len(models),neural_audits=len(nn),neural_exact=int(nn.exact_recovery.sum()),neural_ground_truth_masks=len(models)*256,neural_ground_truth_mask_input_pairs=len(models)*256*256,
        neural_all_tested_conditions_monotone=bool(models.monotone.all()),
        noise_runs=len(noise),noise_false_checked_certificates={m:int(noise[noise.method==m].false_certificate.sum()) for m in ['confidence','fixed_32']},
        main_run_seconds=json.loads((OUT/'environment_all.json').read_text())['elapsed_seconds'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import torch,scipy,matplotlib
    (OUT/'environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,torch=torch.__version__,matplotlib=matplotlib.__version__,training_device='CPU',torch_threads=1,note='See environment_all/rank/noise JSON for measured run durations. Parallel research runs are not a laptop timing benchmark.'),indent=2)+'\n')
    import matplotlib.pyplot as plt
    g=rank[rank.method=='greedy'].groupby('r')[['first_full','query_count']].mean()
    fig,ax=plt.subplots(figsize=(6.2,3.8));ax.plot(g.index,g.first_full,marker='o',label='All six pairs discovered (evaluator)')
    ax.plot(g.index,g.query_count,marker='s',label='Completeness certified (algorithm)')
    ax.set(xlabel='Declared maximum failure-set order r',ylabel='Mean distinct oracle queries',xticks=[2,3,4,5,6,12]);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(ROOT/'figures/rank_discovery_gap.pdf');fig.savefig(ROOT/'figures/rank_discovery_gap.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
