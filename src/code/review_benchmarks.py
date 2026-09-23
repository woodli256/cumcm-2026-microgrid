# -*- coding: utf-8 -*-
"""经济性复核。事后基准只用于评价，不回写正式合同或控制策略。"""
from pathlib import Path
import json
import hashlib
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from solve import ROOT, read_inputs, base_load, predict


def hindsight(net, contract, price, initial, terminal, method='highs-ds'):
    n=len(net); residual=net-contract
    positive=np.maximum(residual,0); negative=np.maximum(-residual,0)
    # Variables are emergency, charge, discharge, waste and stored energy.
    A=lil_matrix((2*n,5*n+1))
    for t in range(n):
        A[t,t]=1; A[t,n+t]=-1; A[t,2*n+t]=1; A[t,3*n+t]=-1
        A[n+t,4*n+t]=-1; A[n+t,4*n+t+1]=1
        A[n+t,n+t]=-.9; A[n+t,2*n+t]=1/.9
    bounds=[(0,v) for v in positive]
    bounds += [(0,min(v,5000/6)) for v in negative]
    bounds += [(0,min(v,5000/6)) for v in positive]
    bounds += [(0,v) for v in negative]+[(1200,10800)]*(n+1)
    bounds[4*n]=(initial,initial); bounds[5*n]=(terminal,terminal)
    objective=np.r_[5*price,np.zeros(4*n+1)]
    sol=linprog(objective,A_eq=A.tocsr(),b_eq=np.r_[residual,np.zeros(n)],
                bounds=bounds,method=method)
    if not sol.success:raise RuntimeError(sol.message)
    x=sol.x
    return {'e':x[:n],'c':x[n:2*n],'d':x[2*n:3*n],
            'w':x[3*n:4*n],'s':x[4*n:],'cost':float(sol.fun)}


def audit(saved, original, net, price):
    """Rebuild constraints directly, without calling the LP matrix builder."""
    e,c,d,w,s=(saved[k] for k in ['e','c','d','w','s'])
    r=net-original['a'];positive=np.maximum(r,0);negative=np.maximum(-r,0)
    violation=max(0.,float(np.max(-np.stack([e,c,d,w]))),
        float(np.max(e-positive)),float(np.max(c-np.minimum(negative,5000/6))),
        float(np.max(d-np.minimum(positive,5000/6))),float(np.max(w-negative)),
        float(1200-s.min()),float(s.max()-10800))
    before=(5*price*original['e']).sum(axis=1)
    after=(5*price*e).sum(axis=1)
    baseline_violation=max(0.,float(np.max(original['e']-positive)),
        float(np.max(original['c']-np.minimum(negative,5000/6))),
        float(np.max(original['d']-np.minimum(positive,5000/6))),
        float(np.max(original['w']-negative)))
    checks={'balance_max_abs':float(np.max(np.abs(e-c+d-w-r))),
        'state_max_abs':float(np.max(np.abs(np.diff(s,axis=1)-.9*c+d/.9))),
        'boundary_max_abs':float(np.max(np.abs(s[:,[0,-1]]-original['s'][:,[0,-1]]))),
        'bounds_violation':violation,'simultaneous_max':float(np.minimum(c,d).max()),
        'daily_dominance_violation':float(max(0,np.max(after-before))),
        'original_branch_violation':baseline_violation,
        'cost_reconstruction_max_abs':float(np.max(np.abs(after-saved['cost'])))}
    assert max(checks.values())<1e-6,checks
    return checks


def repriced(o,p):
    return np.c_[(p*o['g']).sum(1),(5*p*o['e']).sum(1),
        (1.5*p*np.maximum(o['a']-o['g'],0)).sum(1),
        (.5*p*np.maximum(o['g']-o['a'],0)).sum(1)]


def main():
    a=read_inputs(ROOT/'data/inputs.npz');ids=np.arange(31,365)
    report={'scope':'2025-02-01 to 2025-12-31; fixed contracts and original daily endpoints',
            'hindsight':{},'price_value':{},'prediction':{},'sources':{}}
    for name in ['q2','q3','q4_2','q4_3']:
        path=ROOT/f'results/{name}.npz';z=dict(np.load(path))
        assert np.array_equal(z['days'],np.arange(365))
        o={k:z[k][ids] for k in ['a','g','e','c','d','w','s','cost']}
        p=a['price'][ids] if name.startswith('q4') else np.tile(a['base'][:,0],(334,1))
        net=(a['load'][ids]-a['pv'][ids])/6
        rows=[];ipm_diff=[]
        for j,day in enumerate(ids):
            args=(net[j],o['a'][j],p[j],o['s'][j,0],o['s'][j,-1])
            row=hindsight(*args);rows.append(row)
            if day in [78,171,265,354]:
                ipm=hindsight(*args,method='highs-ipm')
                ipm_diff.append(abs(ipm['cost']-row['cost']))
        saved={k:np.array([row[k] for row in rows]) for k in rows[0]}
        checks=audit(saved,o,net,p)
        checks['independent_ipm_max_cost_diff']=max(ipm_diff)
        assert checks['independent_ipm_max_cost_diff']<1e-6
        before=float((5*p*o['e']).sum());after=float(saved['cost'].sum())
        report['hindsight'][name]={'original_emergency_cost':before,
            'hindsight_emergency_cost':after,'gap':before-after,
            'gap_share_original_emergency_percent':100*(before-after)/before,'checks':checks}
        np.savez_compressed(ROOT/f'results/hindsight_{name}.npz',days=ids,**saved)
        report['sources'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
        print(name,report['hindsight'][name],flush=True)
    p=a['price'][ids]
    for fixed,dynamic in [('q2','q4_2'),('q3','q4_3')]:
        z=dict(np.load(ROOT/f'results/{fixed}.npz'));o={k:z[k][ids] for k in ['g','a','e']}
        costs=repriced(o,p);ref=float(costs.sum())
        dz=dict(np.load(ROOT/f'results/{dynamic}.npz'))
        target=float(dz['cost'][ids].sum())
        report['price_value'][dynamic]={'fixed_policy_repriced_cost':ref,
            'price_prediction_policy_cost':target,'saving':ref-target,
            'saving_percent':100*(ref-target)/ref,
            'fixed_policy_components':costs.sum(axis=0).tolist(),
            'initial_soc_fixed':float(z['s'][31,0]),'initial_soc_dynamic':float(dz['s'][31,0]),
            'terminal_soc_fixed':float(z['s'][-1,-1]),'terminal_soc_dynamic':float(dz['s'][-1,-1])}
    loadhat=np.array([base_load(a,day) for day in ids])
    report['prediction']['weekday_load_mae_kw']=float(np.abs(loadhat-a['load'][ids]).mean())
    report['prediction']['previous_day_load_mae_kw']=float(np.abs(a['load'][ids-1]-a['load'][ids]).mean())
    true=(a['load'][ids]-a['pv'][ids])/6
    for use,label in [(False,'history'),(True,'forecast0')]:
        preds=np.array([predict(a,day,0,use)[0] for day in ids])
        report['prediction'][label+'_net_mae_kw']=float(np.abs(preds-true).mean()*6)
    prices=np.array([predict(a,day,0,False)[1] for day in ids])
    report['prediction']['price_mae_yuan_per_kwh']=float(np.abs(prices-p).mean())
    (ROOT/'results/review_benchmarks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding="utf-8")
    print(json.dumps(report['price_value'],indent=2),flush=True)


if __name__=='__main__':main()
