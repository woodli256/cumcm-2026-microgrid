# -*- coding: utf-8 -*-
"""新增建模证据：独立稀疏LP对偶证书、设备边际价值及补救反例。"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
ROOT=Path(__file__).resolve().parents[1]

def solve_day(net,price,capacity=12000.,power=5000.,method='highs-ds'):
    n=len(net);m=5*n+1;eta=.9
    mat=lil_matrix((2*n,m));rhs=np.r_[net,np.zeros(n)]
    for t in range(n):
        mat[t,[t,n+t,2*n+t,3*n+t]]=[1,-1,1,-1]
        mat[n+t,[n+t,2*n+t,4*n+t,4*n+t+1]]=[-eta,1/eta,-1,1]
    mat=mat.tocsr();obj=np.r_[price,np.zeros(m-n)]
    bounds=[(0,None)]*n+[(0,power/6)]*(2*n)+[(0,None)]*n+[(.1*capacity,.9*capacity)]*(n+1)
    bounds[4*n]=(6000,6000);bounds[5*n]=(6000,6000)
    sol=linprog(obj,A_eq=mat,b_eq=rhs,bounds=bounds,method=method)
    assert sol.success,sol.message
    lo=np.array([v[0] for v in bounds]);hi=np.array([np.inf if v[1] is None else v[1] for v in bounds]);finite=np.isfinite(hi)
    y=sol.eqlin.marginals;ml=sol.lower.marginals;mu=sol.upper.marginals;x=sol.x
    dual=rhs@y+lo@ml+hi[finite]@mu[finite]
    checks={'primal_residual':float(np.max(abs(mat@x-rhs))),
        'dual_stationarity':float(np.max(abs(obj-mat.T@y-ml-mu))),
        'dual_sign_violation':float(max(0,-ml.min(),mu.max())),
        'duality_gap':float(abs(sol.fun-dual)),
        'complementarity':float(max(np.max(abs((x-lo)*ml)),np.max(abs((x[finite]-hi[finite])*mu[finite]))))}
    assert max(checks.values())<1e-6,checks
    cap_slope=.1*ml[4*n+1:5*n].sum()+.9*mu[4*n+1:5*n].sum()
    power_slope=mu[n:3*n].sum()/6
    return sol,checks,{'capacity_cost_slope':float(cap_slope),'power_cost_slope':float(power_slope)},y[:n]

def main():
    a=np.load(ROOT/'data/inputs.npz')['base'];net=(a[:,1]-a[:,2])/6;price=a[:,0]
    sol,checks,slopes,shadow=solve_day(net,price)
    other,_,_,_=solve_day(net,price,method='highs-ipm')
    assert abs(other.fun-sol.fun)<1e-6
    old=np.load(ROOT/'results/q1.npz');oldcost=float(price@old['g']);assert abs(oldcost-sol.fun)<1e-6
    report={'objective':float(sol.fun),'old_cost_difference':float(abs(oldcost-sol.fun)),
        'ipm_cost_difference':float(abs(other.fun-sol.fun)),'certificate':checks,'marginal':slopes}
    finite_checks={}
    for key,base in [('capacity',12000.),('power',5000.)]:
        costs=[]
        for delta in [-1.,0.,1.]:costs.append(float(solve_day(net,price,**{key:base+delta})[0].fun))
        left=costs[1]-costs[0];right=costs[2]-costs[1];slope=slopes[key+'_cost_slope']
        assert left-1e-6<=slope<=right+1e-6
        finite_checks[key]={'left_slope':left,'dual_slope':slope,'right_slope':right}
    report['finite_difference']=finite_checks
    curves={}
    for key,grid in [('capacity',np.arange(7000,20001,500)),('power',np.arange(1000,8001,250))]:
        vals=np.array([solve_day(net,price,**{key:float(v)})[0].fun for v in grid])
        assert np.all(np.diff(vals)<=1e-6)
        assert np.all(np.diff(vals,2)>=-1e-6)
        curves[key]={'grid':grid.tolist(),'cost':vals.tolist()}
    report['curves']=curves
    for param in ['capacity','power']:
        report[param+'_plus10percent_cost']=float(solve_day(net,price,**{param:{'capacity':13200,'power':5500}[param]})[0].fun)
    # Two periods, equal 90 kWh deficits, 100 kWh usable cell energy, eta=.9.
    prices=np.array([.2,1.]);deficits=np.array([90.,90.]);available=90.
    toy=linprog(-5*prices,A_eq=[[1,1]],b_eq=[available],bounds=[(0,90)]*2,method='highs')
    grid=np.linspace(0,90,901);enum_cost=5*(prices[0]*(90-grid)+prices[1]*grid)
    opt=float(5*prices@(deficits-toy.x));assert abs(opt-enum_cost.min())<1e-8
    report['two_period']={'prices':prices.tolist(),'initial_soc':1300.,'terminal_soc':1200.,'greedy_discharge':[90.,0.],'optimal_discharge':toy.x.tolist(),'greedy_cost':450.,'optimal_cost':opt,'enumeration_min':float(enum_cost.min())}
    # Chronological decision support: retained summary, not a new fitted forecasting model.
    np.savez_compressed(ROOT/'results/model_depth.npz',shadow_price=shadow,price=price,net=net)
    (ROOT/'results/model_depth.json').write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!='curves'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
