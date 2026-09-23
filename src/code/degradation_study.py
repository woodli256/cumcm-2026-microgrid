# -*- coding: utf-8 -*-
"""问题一的电芯吞吐费用与固定充放电控制尺度情景实验。

rho单位为元/电芯吞吐kWh，只是敏感性假设，不是识别出的真实寿命成本。
负载、光伏、购电结算保持10分钟，只有c、d的控制动作按10/30/60分钟固定。
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix,csr_matrix,coo_matrix
ROOT=Path(__file__).resolve().parents[1]
ETA=.9;B=12000.;DT=1/6;POWER=5000.;LO=1200.;HI=10800.;TARGET=6000.
RHOS=np.array([0.,.02,.05,.10,.20]);MINUTES=np.array([10,30,60])
TOL=1e-6

def solve_model(net,price,rho,block):
    n=len(net);m=5*n+1
    row_count=2*n+2*(n-n//block)
    A=lil_matrix((row_count,m));rhs=np.r_[net,np.zeros(row_count-n)]
    for t in range(n):
        A[t,[t,n+t,2*n+t,3*n+t]]=[1,-1,1,-1]
        A[n+t,[n+t,2*n+t,4*n+t,4*n+t+1]]=[-ETA,1/ETA,-1,1]
    row=2*n
    for start in range(0,n,block):
        for t in range(start+1,start+block):
            A[row,[n+t,n+start]]=[1,-1];row+=1
            A[row,[2*n+t,2*n+start]]=[1,-1];row+=1
    A=A.tocsr()
    f=np.r_[price,np.full(n,rho*ETA),np.full(n,rho/ETA),np.zeros(2*n+1)]
    bounds=[(0,None)]*n+[(0,POWER*DT)]*(2*n)+[(0,None)]*n+[(LO,HI)]*(n+1)
    bounds[4*n]=(TARGET,TARGET);bounds[5*n]=(TARGET,TARGET)
    sol=linprog(f,A_eq=A,b_eq=rhs,bounds=bounds,method='highs-ds')
    if not sol.success:raise RuntimeError(sol.message)
    ml=sol.lower.marginals;mu=sol.upper.marginals;y=sol.eqlin.marginals
    lo=np.array([v[0] for v in bounds]);hi=np.array([np.inf if v[1] is None else v[1] for v in bounds]);fin=np.isfinite(hi)
    dual=float(rhs@y+lo@ml+hi[fin]@mu[fin])
    certificate={'primal_residual':float(np.max(abs(A@sol.x-rhs))),'stationarity':float(np.max(abs(f-A.T@y-ml-mu))),'sign_violation':float(max(0,-ml.min(),mu.max())),'duality_gap':abs(float(sol.fun)-dual),'complementarity':float(max(np.max(abs((sol.x-lo)*ml)),np.max(abs((sol.x[fin]-hi[fin])*mu[fin]))))}
    assert max(certificate.values())<TOL,certificate
    throughput=np.r_[np.zeros(n),np.full(n,ETA),np.full(n,1/ETA),np.zeros(2*n+1)]
    sec=linprog(throughput,A_ub=csr_matrix(f.reshape(1,-1)),b_ub=[sol.fun+1e-7],A_eq=A,b_eq=rhs,bounds=bounds,method='highs-ds')
    if not sec.success:raise RuntimeError(sec.message)
    x=sec.x;g=x[:n];c=x[n:2*n];d=x[2*n:3*n];w=x[3*n:4*n];s=x[4*n:]
    power_violation=max(0,float(c.max()/DT-POWER),float(d.max()/DT-POWER))
    physical={'balance':float(np.max(abs(g+d-c-w-net))),'state':float(np.max(abs(np.diff(s)-ETA*c+d/ETA))),'bounds':float(max(0,LO-s.min(),s.max()-HI,-g.min(),-c.min(),-d.min(),-w.min())),'power':power_violation,'terminal':float(max(abs(s[0]-TARGET),abs(s[-1]-TARGET))),'simultaneous_charge_discharge':float(np.min(np.vstack([c,d]),axis=0).max()),'fixed_block':float(max(np.ptp(c.reshape(-1,block),axis=1).max(),np.ptp(d.reshape(-1,block),axis=1).max())),'objective_allowance':float(f@x-sol.fun)}
    assert max(physical.values())<TOL,physical
    cell_throughput=float(ETA*c.sum()+d.sum()/ETA)
    state=np.where(c>1e-5,1,np.where(d>1e-5,-1,0))
    switches=int(np.count_nonzero(np.diff(state)))
    direct=int(np.count_nonzero(state[1:]*state[:-1]==-1))
    operating=float(price@g);degradation=float(rho*cell_throughput)
    record={'rho_yuan_per_cell_kwh':float(rho),'control_minutes':int(block*10),'grid_minutes':10,'grid_electricity_kwh':float(g.sum()),'purchase_cost_yuan':operating,'cell_throughput_kwh':cell_throughput,'efc':cell_throughput/(2*B),'degradation_cost_yuan':degradation,'total_objective_yuan':operating+degradation,'primary_objective_yuan':float(sol.fun),'mode_switches':switches,'direct_charge_discharge_switches':direct,'primary_certificate':certificate,'physical_checks':physical}
    return record,(g,c,d,w,s)

def independent_cell_model(net,price,rho,block):
    # Independent coordinate formulation: h,k are cell-side charge/withdrawal.
    n=len(net);rows=[];cols=[];vals=[];rhs=[]
    def eq(entries,b):
        row=len(rhs)
        for col,val in entries:rows.append(row);cols.append(col);vals.append(val)
        rhs.append(b)
    for t in range(n):eq([(t,1),(n+t,-1/ETA),(2*n+t,ETA),(3*n+t,-1)],float(net[t]))
    for t in range(n):eq([(n+t,-1),(2*n+t,1),(4*n+t,-1),(4*n+t+1,1)],0.)
    for start in range(0,n,block):
        for t in range(start+1,start+block):
            eq([(n+t,1),(n+start,-1)],0.);eq([(2*n+t,1),(2*n+start,-1)],0.)
    mat=coo_matrix((vals,(rows,cols)),shape=(len(rhs),5*n+1)).tocsr()
    f=np.r_[price,np.full(2*n,rho),np.zeros(2*n+1)]
    bd=[(0,None)]*n+[(0,POWER*DT*ETA)]*n+[(0,POWER*DT/ETA)]*n+[(0,None)]*n+[(LO,HI)]*(n+1)
    bd[4*n]=(TARGET,TARGET);bd[5*n]=(TARGET,TARGET)
    sol=linprog(f,A_eq=mat,b_eq=rhs,bounds=bd,method='highs-ipm')
    assert sol.success,sol.message
    return float(sol.fun)

def main():
    q1=ROOT/'results/q1.npz';source_hash=hashlib.sha256(q1.read_bytes()).hexdigest()
    a=np.load(ROOT/'data/inputs.npz')['base'];net=(a[:,1]-a[:,2])*DT;price=a[:,0]
    records=[];arrays={k:[] for k in ['g','c','d','w','s']}
    for minutes in MINUTES:
        for rho in RHOS:
            record,flows=solve_model(net,price,float(rho),int(minutes//10))
            records.append(record)
            for key,value in zip(arrays,flows):arrays[key].append(value)
    representative=[]
    for index in [0,8,14]:
        rec=records[index];val=independent_cell_model(net,price,rec['rho_yuan_per_cell_kwh'],rec['control_minutes']//10)
        diff=abs(val-rec['primary_objective_yuan']);assert diff<1e-6
        representative.append({'rho':rec['rho_yuan_per_cell_kwh'],'control_minutes':rec['control_minutes'],'independent_cell_ipm_objective':val,'difference_yuan':diff})
    totals=np.array([r['total_objective_yuan'] for r in records]).reshape(3,5)
    through=np.array([r['cell_throughput_kwh'] for r in records]).reshape(3,5)
    assert np.all(np.diff(totals,axis=1)>=-1e-5)
    assert np.all(np.diff(totals,axis=0)>=-1e-5)
    assert np.all(np.diff(through,axis=1)<=1e-4)
    old_cost=float(np.load(q1)['objective']);assert abs(records[0]['purchase_cost_yuan']-old_cost)<1e-6
    assert hashlib.sha256(q1.read_bytes()).hexdigest()==source_hash
    report={'scope':'问题一确定性单日情景实验，保持负载/光伏/价格的10分钟数据分辨率，g按10分钟自由选择，只有母线c、d在各控制块内分别固定。','parameters':{'B_kwh':B,'eta':ETA,'rho_grid_yuan_per_cell_kwh':RHOS.tolist(),'control_minutes':MINUTES.tolist()},'definitions':{'cell_throughput':'eta*sum(c)+sum(d)/eta','efc':'cell_throughput/(2*B)','mode_switches':'0=idle,1=charge,-1=discharge; count adjacent 10-minute mode changes above 1e-5 kWh, no last-to-first wrap','degradation_cost':'rho*cell_throughput; rho is an assumed scenario parameter, not fitted battery lifetime cost'},'records':records,'independent_cross_checks':representative,'monotonicity_checks':{'objective_nondecreasing_in_rho':True,'objective_nondecreasing_with_block_length':True,'throughput_nonincreasing_in_rho':True},'main_q1_unchanged':True,'main_q1_sha256':source_hash,'baseline_purchase_cost_difference':abs(records[0]['purchase_cost_yuan']-old_cost)}
    (ROOT/'results/degradation_study.json').write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding="utf-8")
    np.savez_compressed(ROOT/'results/degradation_study.npz',rho=np.tile(RHOS,3),control_minutes=np.repeat(MINUTES,5),**{k:np.array(v) for k,v in arrays.items()})
    print(json.dumps({'records':[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in records],'cross_checks':representative},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
