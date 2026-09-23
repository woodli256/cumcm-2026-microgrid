# -*- coding: utf-8 -*-
"""共同小时末储量的场景风险滚动控制。预测规划仅是代理，实际逐段因果执行。"""
from pathlib import Path
from functools import lru_cache
import argparse,json,time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import block_diag,lil_matrix,csr_matrix,vstack
from solve import ROOT,read_inputs,forecasts,predict,structure,step,DT,LO,HI,ETA,POWER,Q_MAIN
HORIZON=36
SCENARIOS=5
ALPHA=.8

@lru_cache(None)
def matrices(n,m,risk):
    a,nv=structure(n,ETA,False)
    extra=1+m if risk else 0
    eq=lil_matrix((2*n*m+m-1,nv*m+extra))
    eq[:2*n*m,:nv*m]=block_diag([a]*m,format='csr')
    h=min(6,n)
    for z in range(1,m):
        eq[2*n*m+z-1,z*nv+4*n+h]=1
        eq[2*n*m+z-1,4*n+h]=-1
    return eq.tocsr(),nv

def reserve_lp(net,contract,price,initial,weight,billed=None,method='highs'):
    m,n=net.shape;r=net-contract[None,:];pos=np.maximum(r,0);neg=np.maximum(-r,0)
    A,nv=matrices(n,m,weight>0);obj=np.zeros(A.shape[1]);bounds=[];rhs=[]
    loss=lil_matrix((m,A.shape[1]))
    for z in range(m):
        offset=z*nv;obj[offset:offset+n]=5*price[z]/m
        loss[z,offset:offset+n]=5*price[z]
        bounds += [(0,float(v)) for v in pos[z]]
        bounds += [(0,float(min(v,POWER*DT))) for v in neg[z]]
        bounds += [(0,float(min(v,POWER*DT))) for v in pos[z]]
        bounds += [(0,float(v)) for v in neg[z]]+[(LO,HI)]*(n+1)
        bounds[offset+4*n]=(initial,initial)
        rhs.extend(r[z]);rhs.extend(np.zeros(n))
    rhs.extend(np.zeros(m-1))
    Aub=bub=None
    if weight>0:
        index=m*nv;obj[index]=weight;obj[index+1:]=weight/(m*(1-ALPHA))
        bounds += [(0,None)]*(m+1)
        for z in range(m):loss[z,index]=-1;loss[z,index+1+z]=-1
        Aub=loss.tocsr();bub=-(price@(contract if billed is None else billed))
    sol=linprog(obj,A_eq=A,b_eq=rhs,A_ub=Aub,b_ub=bub,bounds=bounds,method=method)
    if not sol.success:raise RuntimeError(sol.message)
    target=np.mean(np.stack([sol.x[z*nv+4*n+1:z*nv+4*n+min(6,n)+1] for z in range(m)]),axis=0)
    residual=float(np.max(np.abs(A@sol.x-rhs)))
    return target,float(sol.fun+np.mean(price@(contract if billed is None else billed))),residual

def prepare(a):
    fc=forecasts(a)
    ph=np.array([predict(a,j,0,False)[1] for j in range(365)])
    pe=a['price']-ph
    return fc,ph,pe

def scenarios(a,prep,day,t,use,dynamic,contract,mode="scenario"):
    fc,ph,pe=prep;end=min(144,t+HORIZON);issue=t//36 if use else 0
    pred,err=fc[use]
    ids=np.unique(np.linspace(max(1,day-28),day-1,min(SCENARIOS,day-1),dtype=int))
    net=pred[day,issue,t:end][None,:]+err[ids,issue,t:end]
    price=np.maximum(.001,ph[day,t:end][None,:]+pe[ids,t:end]) if dynamic else np.tile(a['base'][t:end,0],(len(ids),1))
    if mode=="nominal":
        net=pred[day,issue,t:end][None,:].copy()
        price=(ph[day,t:end] if dynamic else a['base'][t:end,0])[None,:].copy()
    # At the update instant only the present interval is observed.
    net[:,0]=(a['load'][day,t]-a['pv'][day,t])*DT
    if dynamic:price[:,0]=a['price'][day,t]
    assert np.max(ids)<day
    return net,price

def run(a,prep,name,weight,end=365,mode="scenario"):
    orig=dict(np.load(ROOT/f'results/{name}.npz'))
    use=name in ['q3','q4_3'];dynamic=name.startswith('q4')
    o={k:np.zeros((end,144)) for k in ['g','a','c','d','e','w','up','down']}
    o['s']=np.zeros((end,145));o['cost']=np.zeros((end,4));o['reserve']=np.zeros((end,144))
    state=6000.;worst=0.;calls=0
    for j in range(end):
        o['s'][j,0]=state
        # Original contracts are a causally generated shadow-policy control input.
        # For forecasts of future contracts use only the latest issued original plan,
        # never read a later day's or later update's final contract.
        g=orig['g'][j].copy();active=g.copy();o['g'][j]=g
        price=a['price'][j] if dynamic else a['base'][:,0]
        target=LO
        for t in range(144):
            if use and t in [36,72,108]:
                # Reconstruct issued contract using the original (shadow) state at t.
                from solve import plan,safety
                issue=t//36
                pred,err=prep[0][use]
                p=prep[1][j,t:] if dynamic else a['base'][t:,0]
                active[t:]=plan(pred[j,issue,t:]+safety(err,j,issue,Q_MAIN[name]),p,float(orig['s'][j,t]),original=g[t:])['g']
            if t%6==0:
                if j>=7:
                    net,pr=scenarios(a,prep,j,t,use,dynamic,active[t:],mode=mode)
                    nn=net.shape[1];aa=active[t:t+nn];gg=g[t:t+nn]
                    billed=gg+1.5*np.maximum(aa-gg,0)+.5*np.maximum(gg-aa,0)
                    target,_,res=reserve_lp(net,aa,pr,state,weight,billed=billed)
                    worst=max(worst,res);calls+=1
                else:target=np.full(min(6,144-t),LO)
                o['reserve'][j,t:t+len(target)]=target
            actual=(a['load'][j,t]-a['pv'][j,t])*DT
            balance=active[t]-actual
            if balance>=0:
                ch=min(balance,POWER*DT,max(0,(HI-state)/ETA));di=em=0.;w=balance-ch
            else:
                ch=w=0.;di=min(-balance,POWER*DT,max(0,ETA*(state-max(LO,o['reserve'][j,t]))))
                em=-balance-di
            state+=ETA*ch-di/ETA
            for k,v in [('a',active[t]),('c',ch),('d',di),('e',em),('w',w)]:o[k][j,t]=v
            o['s'][j,t+1]=state
        o['up'][j]=np.maximum(o['a'][j]-g,0);o['down'][j]=np.maximum(g-o['a'][j],0)
        o['cost'][j]=[price@g,5*price@o['e'][j],1.5*price@o['up'][j],.5*price@o['down'][j]]
        if j%30==0:print(name,weight,j,float(o['cost'][:j+1].sum()),flush=True)
    o['days']=np.arange(end)
    o['lp_max_residual']=np.array(worst);o['lp_calls']=np.array(calls)
    return o

def audit(a,o,name,original):
    n=(a['load'][:len(o['days'])]-a['pv'][:len(o['days'])])*DT
    checks={'balance':float(np.max(np.abs(o['a']+o['e']+o['d']-o['c']-o['w']-n))),
        'state':float(np.max(np.abs(np.diff(o['s'],axis=1)-ETA*o['c']+o['d']/ETA))),
        'bounds':float(max(0,LO-o['s'].min(),o['s'].max()-HI,o['c'].max()-POWER*DT,o['d'].max()-POWER*DT)),
        'simultaneous':float(np.minimum(o['c'],o['d']).max()),
        'continuity':float(np.max(np.abs(o['s'][:-1,-1]-o['s'][1:,0]))),
        'contract_difference':float(np.max(np.abs(o['a']-original['a'][:len(o['days'])]))),
        'lp_residual':float(o['lp_max_residual'])}
    assert max(checks.values())<1e-5,checks
    return checks

def metrics(o,first=31):
    cost=o['cost'][first:].sum(1);order=np.sort(cost);n=len(cost);tail=max(1,int(np.ceil(.05*n)))
    return {'cost':float(cost.sum()),'emergency_cost':float(o['cost'][first:,1].sum()),'emergency_kwh':float(o['e'][first:].sum()),
        'daily_cost_p95':float(np.quantile(cost,.95)),'worst_5_percent_daily_mean':float(order[-tail:].mean()),
        'initial_soc':float(o['s'][first,0]),'final_soc':float(o['s'][-1,-1]),'days':n}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',choices=['pilot','train','test'],default='test')
    args=ap.parse_args()
    if args.phase=='test':
        import runpy
        runpy.run_path(str(ROOT/'code/run_control_comparison.py'),run_name='__main__')
        return
    a=read_inputs(ROOT/'data/inputs.npz');prep=prepare(a)
    report_path=ROOT/'results/risk_mpc.json'
    report=json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {
        'parameters':{'scenario_count':SCENARIOS,'horizon_steps':HORIZON,'update_steps':6,
                      'alpha':ALPHA,'candidate_weights':[0.,.5,1.]},'training':{},'test':{}}
    if args.phase=='pilot':
        started=time.monotonic();o=run(a,prep,'q2',.5,end=9)
        print('pilot seconds',time.monotonic()-started)
        print(audit(a,o,'q2',dict(np.load(ROOT/'results/q2.npz'))))
        return
    for name in ['q2','q3','q4_2','q4_3']:
        original=dict(np.load(ROOT/f'results/{name}.npz'))
        values=[]
        for weight in report['parameters']['candidate_weights']:
            o=run(a,prep,name,weight,end=31)
            values.append({'weight':weight,'cost':float(o['cost'][14:].sum()),
                           'checks':audit(a,o,name,original)})
        chosen=min(values,key=lambda x:x['cost'])
        report['training'][name]={'candidates':values,'selected_weight':chosen['weight'],
                                 'baseline_cost':float(original['cost'][14:31].sum())}
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding="utf-8")
        print('done',name,args.phase,flush=True)
if __name__=='__main__':main()
