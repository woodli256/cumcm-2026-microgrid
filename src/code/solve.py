# -*- coding: utf-8 -*-
"""2026C 因果预测、线性储能调度及顺序回测。"""
from pathlib import Path
from functools import lru_cache
import argparse, json, time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

ROOT=Path(__file__).resolve().parents[1]
DT=1/6; LO=1200.; HI=10800.; POWER=5000.; ETA=.9
QS=[.5,.65,.8,.9,.95]
# 解析风险分位数：补救手段决定最优水平。
# 仅有 5 倍紧急购电时，单时段模型给出 F_X(g)=0.8；
# 存在 1.5 倍增购选项时，超出部分只多付 0.5p，而多发计划要全额照付，
# 临界比降到 1/3 至 0.8 之间，故这类问题取 0.65。
Q_MAIN={'q2':.8,'q3':.65,'q4_2':.8,'q4_3':.65}

def read_inputs(path):
    z=np.load(path)
    a={k:z[k] for k in z.files}
    a['forecast']=a['forecast'].reshape(365,4,24)
    return a

def base_load(a,day):
    ids=np.arange(max(0,day-28),day)
    ids=ids[(day-ids)%7==0]
    if len(ids)==0: ids=np.arange(max(0,day-7),day)
    if len(ids)==0: return np.full(144,4000.)
    return np.median(a['load'][ids],axis=0)

def predict(a,day,issue,use_forecast):
    k=issue*36; l=base_load(a,day)
    if k:
        ratio=np.median(a['load'][day,k-6:k]/np.maximum(l[k-6:k],1))
        l=l*np.clip(ratio,.7,1.3)
    if use_forecast:
        left=a['pv'][day,k-1] if k else 0.
        xp=np.arange(25)*6+k
        fp=np.r_[left,a['forecast'][day,issue]]
        pv=np.interp(np.arange(k+1,145),xp,fp)
    else:
        pv=np.mean(a['pv'][max(0,day-7):day],axis=0)[k:] if day else np.zeros(144)
    price=np.median(a['price'][max(0,day-7):day],axis=0) if day else a['base'][:,0].copy()
    return (l[k:]-pv)*DT, price[k:]

def forecasts(a):
    out={}
    for use in [False,True]:
        pred=np.full((365,4,144),np.nan)
        err=np.full_like(pred,np.nan)
        for day in range(365):
            for issue in range(4 if use else 1):
                k=issue*36
                n,_=predict(a,day,issue,use)
                pred[day,issue,k:]=n
                err[day,issue,k:]=(a['load'][day,k:]-a['pv'][day,k:])*DT-n
        out[use]=(pred,err)
    return out

def safety(err,day,issue,q):
    k=issue*36; b=np.zeros(144-k)
    if day<7: return b
    for lo in range(k,144,12):
        pool=err[max(1,day-28):day,issue,lo:lo+12]
        pool=pool[np.isfinite(pool)]
        if len(pool): b[lo-k:lo-k+12]=np.quantile(pool,q)
    return b

@lru_cache(None)
def structure(n,eta,adjust):
    # g,c,d,w,s(0..n), [u,v]
    count=5*n+1+(2*n if adjust else 0)
    A=lil_matrix((2*n+(n if adjust else 0),count))
    for t in range(n):
        A[t,t]=1; A[t,n+t]=-1; A[t,2*n+t]=1; A[t,3*n+t]=-1
        A[n+t,4*n+t]=-1; A[n+t,4*n+t+1]=1
        A[n+t,n+t]=-eta; A[n+t,2*n+t]=1/eta
        if adjust:
            A[2*n+t,t]=1; A[2*n+t,5*n+1+t]=-1
            A[2*n+t,6*n+1+t]=1
    return A.tocsr(),count

def plan(net,p,s0,original=None,eta=ETA,target=6000.,refund=False,method='highs'):
    n=len(net); adj=original is not None
    A,m=structure(n,float(eta),adj)
    c=np.zeros(m)
    if adj:
        c[5*n+1:6*n+1]=1.5*p
        c[6*n+1:]=(-.5 if refund else .5)*p
    else: c[:n]=p
    bounds=[(0,None)]*n+[(0,POWER*DT)]*(2*n)+[(0,None)]*n+[(LO,HI)]*(n+1)
    bounds[4*n]=(s0,s0);bounds[5*n]=(target,target)
    if adj: bounds += [(0,None)]*(2*n)
    b=np.r_[net,np.zeros(n),original if adj else []]
    res=linprog(c,A_eq=A,b_eq=b,bounds=bounds,method=method)
    if not res.success: raise RuntimeError(res.message)
    primary=float(res.fun)
    # Lexicographic throughput minimization removes gratuitous charge-discharge.
    c2=np.zeros(m);c2[n:3*n]=1
    sec=linprog(c2,A_ub=csr_matrix(c.reshape(1,-1)),b_ub=[primary+1e-7],
                A_eq=A,b_eq=b,bounds=bounds,method=method)
    if not sec.success: raise RuntimeError(sec.message)
    x=sec.x
    return {'g':x[:n],'c':x[n:2*n],'d':x[2*n:3*n],
            'w':x[3*n:4*n],'s':x[4*n:5*n+1],
            'objective':float(c@x),'primary':primary}

def step(g,net,s,eta=ETA):
    balance=g-net
    if balance>=0:
        charge=min(balance,POWER*DT,max(0,(HI-s)/eta))
        discharge=emergency=0.;w=balance-charge
    else:
        charge=w=0.;discharge=min(-balance,POWER*DT,max(0,(s-LO)*eta))
        emergency=-balance-discharge
    return charge,discharge,emergency,w,s+eta*charge-discharge/eta

def run(a,fc,use=False,dynamic=False,q=.8,updates=(1,2,3),refund=False,start=0,end=365):
    pred,err=fc[use]; size=end-start
    out={k:np.zeros((size,144)) for k in ['g','a','c','d','e','w','up','down']}
    out['s']=np.zeros((size,145));out['cost']=np.zeros((size,4))
    s=6000.
    for day in range(start,end):
        row=day-start;out['s'][row,0]=s
        price=a['price'][day] if dynamic else a['base'][:,0]
        _,ph=predict(a,day,0,use)
        p=ph if dynamic else a['base'][:,0]
        n=pred[day,0]+safety(err,day,0,q)
        g=plan(n,p,s)['g'];out['g'][row]=g
        active=g.copy()
        for t in range(144):
            if use and t in [36*j for j in updates]:
                issue=t//36
                _,ph=predict(a,day,issue,True)
                pp=ph if dynamic else a['base'][t:,0]
                nn=pred[day,issue,t:]+safety(err,day,issue,q)
                adj=plan(nn,pp,s,original=g[t:],refund=refund)
                active[t:]=adj['g']
            actual=(a['load'][day,t]-a['pv'][day,t])*DT
            ch,dis,em,w,s=step(active[t],actual,s)
            out['a'][row,t]=active[t]
            for key,val in [('c',ch),('d',dis),('e',em),('w',w)]:out[key][row,t]=val
            out['s'][row,t+1]=s
        out['up'][row]=np.maximum(out['a'][row]-g,0)
        out['down'][row]=np.maximum(g-out['a'][row],0)
        out['cost'][row]=[price@g,price@out['e'][row]*5,
             price@out['up'][row]*1.5,price@out['down'][row]*(-.5 if refund else .5)]
    out['days']=np.arange(start,end)
    return out

def totals(o,first=31):
    keep=o['days']>=first
    return {'days':int(keep.sum()),'cost_components':o['cost'][keep].sum(axis=0).tolist(),
            'total_cost':float(o['cost'][keep].sum()),
            **{k:float(o[k][keep].sum()) for k in ['g','a','e','w','c','d','up','down']},
            'initial_soc':float(o['s'][keep][0,0]),'final_soc':float(o['s'][keep][-1,-1]),
            'emergency_intervals':int((o['e'][keep]>1e-6).sum())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,default=ROOT/'data/inputs.npz')
    ap.add_argument('--output',type=Path,default=ROOT/'results');ap.add_argument('--quick',action='store_true')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    a=read_inputs(args.inputs);fc=forecasts(a)
    np.savez_compressed(args.output/'forecasts.npz',p2=fc[False][0],p3=fc[True][0])
    q1=plan((a['base'][:,1]-a['base'][:,2])*DT,a['base'][:,0],6000.)
    np.savez_compressed(args.output/'q1.npz',**q1)
    summary={'q1':{'cost':float(a['base'][:,0]@q1['g']),'g':float(q1['g'].sum()),
        'no_storage_cost':float(a['base'][:,0]@np.maximum((a['base'][:,1]-a['base'][:,2])*DT,0))},'training':{},'strategies':{}}
    names=[('q2',False,False),('q3',True,False),('q4_2',False,True),('q4_3',True,True)]
    for name,use,dynamic in names:
        train=[]
        for q in QS:
            o=run(a,fc,use,dynamic,q,start=0,end=31)
            train.append(float(o['cost'][14:].sum()))
        qtr=QS[int(np.argmin(train))]
        qmain=Q_MAIN[name]
        summary['training'][name]={'quantiles':QS,'costs_jan15_31':train,
            'selected_q':qtr,'main_q':qmain}
        print(name,'training',train,'training_pick',qtr,'main_q',qmain,flush=True)
        o=run(a,fc,use,dynamic,qmain,end=40 if args.quick else 365)
        np.savez_compressed(args.output/f'{name}.npz',**o)
        summary['strategies'][name]=totals(o)
        print(name,summary['strategies'][name],flush=True)
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2), encoding="utf-8")

if __name__=='__main__':main()
