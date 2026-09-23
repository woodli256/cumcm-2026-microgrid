# -*- coding: utf-8 -*-
"""评审后追加的顺序回测。原策略和正式工作簿不覆盖。

本次设计已经接触过全年汇总，属于回顾性方法修订，不是未触碰的测试。
所有动作仍仅使用当时已发布预报与已发生的历史。
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from solve import (ROOT, DT, ETA, POWER, LO, HI, QS, Q_MAIN, read_inputs, forecasts,
                   predict, safety, plan, step, structure, run, totals)
from validate import check

OUT = ROOT/'results/revision'
OUT.mkdir(exist_ok=True)

def extended_plan(net, price, s0, original):
    """当天合同按调整费结算，次日虚拟购电按普通预测价估值。"""
    n=len(net); m0=len(original); A,m=structure(n,ETA,True)
    c=np.zeros(m); c[m0:n]=price[m0:]
    c[5*n+1:5*n+1+m0]=1.5*price[:m0]
    c[6*n+1:6*n+1+m0]=.5*price[:m0]
    bounds=[(0,None)]*n+[(0,POWER*DT)]*(2*n)+[(0,None)]*n+[(LO,HI)]*(n+1)
    bounds[4*n]=(s0,s0); bounds[5*n]=(6000,6000)
    bounds += [(0,None)]*(2*n)
    b=np.r_[net,np.zeros(n),original,np.zeros(n-m0)]
    r=linprog(c,A_eq=A,b_eq=b,bounds=bounds,method='highs')
    if not r.success: raise RuntimeError(r.message)
    c2=np.zeros(m); c2[n:3*n]=1
    r2=linprog(c2,A_ub=csr_matrix(c.reshape(1,-1)),b_ub=[r.fun+1e-7],A_eq=A,b_eq=b,bounds=bounds,method='highs')
    if not r2.success: raise RuntimeError(r2.message)
    return r2.x[:m0]

def next_forecast(a,fc,day,issue,q):
    """当日issue发布的24小时预报中，次日0点以后的部分。"""
    length=issue*36
    ids=np.arange(max(0,day-28),day) # 严格排除当日尚未完成的实际数据
    same=ids[(day+1-ids)%7==0]
    if len(same): ids=same
    if len(ids): load=np.median(a['load'][ids,:length],axis=0)
    else: load=np.full(length,4000.)
    k=issue*36
    pv=np.interp(np.arange(145,145+length),np.arange(25)*6+k,
                 np.r_[a['pv'][day,k-1],a['forecast'][day,issue]])
    margin=safety(fc[True][1],day,0,q)[:length]
    price=np.median(a['price'][max(0,day-7):day,:length],axis=0) if day else a['base'][:length,0]
    return (load-pv)*DT+margin,price

def weighted_margin(a,err,day,issue,q):
    """保持分位数水平，仅用配对历史真实价格加权历史缺口。"""
    k=issue*36; out=safety(err,day,issue,q)
    if day<7:return out
    for lo in range(k,144,12):
        vals=err[max(1,day-28):day,issue,lo:lo+12].ravel()
        weights=a['price'][max(1,day-28):day,lo:lo+12].ravel()
        keep=np.isfinite(vals); vals=vals[keep]; weights=weights[keep]
        idx=np.argsort(vals); vals=vals[idx]; weights=weights[idx]
        out[lo-k:lo-k+12]=vals[min(np.searchsorted(np.cumsum(weights),q*weights.sum()),len(vals)-1)]
    return out

def variant(a,fc,use=False,dynamic=False,qs=None,crossday=False,weighted=False,q=.65):
    out={k:np.zeros((365,144)) for k in ['g','a','c','d','e','w','up','down']}
    out['s']=np.zeros((365,145));out['cost']=np.zeros((365,4));out['days']=np.arange(365)
    out['q']=np.full(365,q) if qs is None else np.array(qs)
    pred,err=fc[use];s=6000.
    for day in range(365):
        q=out['q'][day];out['s'][day,0]=s
        price=a['price'][day] if dynamic else a['base'][:,0]
        def margin(issue):
            return weighted_margin(a,err,day,issue,q) if weighted else safety(err,day,issue,q)
        ph=predict(a,day,0,use)[1] if dynamic else a['base'][:,0]
        g=plan(pred[day,0]+margin(0),ph,s)['g'];out['g'][day]=g;active=g.copy()
        for t in range(144):
            if use and t in (36,72,108):
                issue=t//36;ph=predict(a,day,issue,True)[1] if dynamic else a['base'][t:,0]
                net=pred[day,issue,t:]+margin(issue)
                if crossday and day<364:
                    nxt,px=next_forecast(a,fc,day,issue,q)
                    if not dynamic:px=a['base'][:len(nxt),0]
                    active[t:]=extended_plan(np.r_[net,nxt],np.r_[ph,px],s,g[t:])
                else:active[t:]=plan(net,ph,s,original=g[t:])['g']
            ch,di,em,w,s=step(active[t],(a['load'][day,t]-a['pv'][day,t])*DT,s)
            for key,v in [('a',active[t]),('c',ch),('d',di),('e',em),('w',w)]:out[key][day,t]=v
            out['s'][day,t+1]=s
        out['up'][day]=np.maximum(active-g,0);out['down'][day]=np.maximum(g-active,0)
        out['cost'][day]=[price@g,5*price@out['e'][day],1.5*price@out['up'][day],.5*price@out['down'][day]]
    return out

def reserve_control(a,base):
    """固定已发布合同，低于未来六小时最高电价时保留3600kWh。"""
    out={k:np.array(v,copy=True) for k,v in base.items()};s=6000.;p=a['base'][:,0]
    for day in range(365):
        out['s'][day,0]=s
        for t in range(144):
            net=(a['load'][day,t]-a['pv'][day,t])*DT
            reserve=3600. if p[t]<p[t:min(144,t+36)].max()-1e-9 else LO
            ch,di,em,w,snew=step(out['a'][day,t],net,s)
            limited=min(di,max(0,(s-reserve)*ETA))
            em+=di-limited;snew+=(di-limited)/ETA;di=limited
            for key,v in [('c',ch),('d',di),('e',em),('w',w)]:out[key][day,t]=v
            out['s'][day,t+1]=s=snew
        out['cost'][day,1]=5*p@out['e'][day]
    return out

def main():
    a=read_inputs(ROOT/'data/inputs.npz');fc=forecasts(a);report={}
    def record(name,fn,dynamic=False):
        path=OUT/(name+'.npz')
        o=dict(np.load(path)) if path.exists() else fn()
        price=a['price'] if dynamic else np.tile(a['base'][:,0],(365,1))
        validation=check(o,a['load'],a['pv'],price)
        assert validation['pass'],(name,validation)
        np.savez_compressed(path,**o)
        total=totals(o)
        # 同一参考价格给测试首末储量记账，数值不当作真实可交易收入。
        v=float(np.mean(a['base'][:,0])*.9)
        total['inventory_adjusted_cost']=total['total_cost']+v*(total['initial_soc']-total['final_soc'])
        report[name]={'totals':total,'validation':validation}
        (OUT/'experiments.json').write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding="utf-8")
        print(name,round(total['total_cost'],2),flush=True)
        return o
    for name in ['q2','q3','q4_2','q4_3']:
        record('baseline_'+name,lambda name=name:dict(np.load(ROOT/'results'/f'{name}.npz')),name.startswith('q4'))
    for name,updates in [('without6',(2,3)),('without12',(1,3)),('without18',(1,2)),('only0',())]:
        record(name,lambda updates=updates:run(a,fc,True,False,Q_MAIN['q3'],updates=updates))
    experts=[]
    for q in QS:
        experts.append(record('expert_q2_'+str(q),lambda q=q:run(a,fc,False,False,q)))
    from datetime import date,timedelta
    dates=[date(2025,1,1)+timedelta(days=d) for d in range(365)]
    qs=np.full(365,Q_MAIN['q2']);choices=[]
    for d in range(31,365):
        if dates[d].day==1:
            scores=[o['cost'][max(0,d-28):d].sum() for o in experts]
            selected=QS[int(np.argmin(scores))]
            choices.append({'date':dates[d].isoformat(),'q':selected,'past28_costs':scores})
        qs[d]=selected
    record('rolling_q2',lambda:variant(a,fc,qs=qs))
    report['rolling_choices']=choices
    record('crossday_q3',lambda:variant(a,fc,use=True,crossday=True,q=Q_MAIN['q3']))
    record('weighted_q4_2',lambda:variant(a,fc,dynamic=True,weighted=True,q=Q_MAIN['q4_2']),True)
    record('weighted_q4_3',lambda:variant(a,fc,use=True,dynamic=True,weighted=True,q=Q_MAIN['q4_3']),True)
    for name in ['q2','q3']:
        record('reserve_'+name,lambda name=name:reserve_control(a,dict(np.load(ROOT/'results'/f'{name}.npz'))))
    # 原12组失败控制的时段费用与储量、未利用量证据，避免仅以猜测解释。
    diagnostics=[]
    for name in ['q2','q3','q4_2','q4_3']:
        base=dict(np.load(ROOT/'results'/f'{name}.npz'));p=a['price'] if name.startswith('q4') else np.tile(a['base'][:,0],(365,1))
        for mode in ['nominal','scenario_mean','scenario_cvar']:
            o=dict(np.load(ROOT/'results'/f'risk_mpc_{name}_{mode}.npz'))
            inc=(o['e'][31:]-base['e'][31:])*5*p[31:]
            hours=inc.reshape(334,24,6).sum((0,2));top=np.argsort(hours)[-3:][::-1]
            diagnostics.append({'strategy':name,'mode':mode,'extra_emergency_by_hour_yuan':hours.tolist(),
                'top_hours':top.tolist(),'extra_waste_kwh':float((o['w'][31:]-base['w'][31:]).sum()),
                'mean_extra_soc_kwh':float((o['s'][31:]-base['s'][31:]).mean()),
                'extra_charge_kwh':float((o['c'][31:]-base['c'][31:]).sum()),
                'extra_discharge_kwh':float((o['d'][31:]-base['d'][31:]).sum())})
    report['control_diagnostics']=diagnostics
    report['design']={'evaluation':'retrospective sequential reanalysis','rolling':'monthly past28 expert cost; full365 causal states; fixed grid',
       'crossday':'published24h forecast; nextday purchase surrogate only; terminal6000 at horizon end',
       'weighted':'past28 paired realized price/error weights at original q0.65',
       'reserve':'fixed baseline contracts; reserve3600 when current price below remaining6h maximum',
       'inventory_price_yuan_per_kwh':float(np.mean(a['base'][:,0])*.9)}
    (OUT/'experiments.json').write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding="utf-8")

if __name__=='__main__': main()
