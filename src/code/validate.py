# -*- coding: utf-8 -*-
"""独立按原始电量公式审计已保存的调度，不调用执行器计算残差。"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from solve import ROOT,read_inputs,predict,safety,forecasts,plan,step,QS

def check(o,load,pv,price,refund=False):
    net=(load-pv)/6
    bal=o['a']+o['e']+o['d']-o['c']-o['w']-net
    state=np.diff(o['s'],axis=1)-.9*o['c']+o['d']/.9
    exp=np.c_[(o['g']*price).sum(1),(o['e']*price*5).sum(1),
        (np.maximum(o['a']-o['g'],0)*price*1.5).sum(1),
        (np.maximum(o['g']-o['a'],0)*price*(-.5 if refund else .5)).sum(1)]
    errors={'energy_max_abs':float(np.abs(bal).max()),'state_max_abs':float(np.abs(state).max()),
        'cost_max_abs':float(np.abs(exp-o['cost']).max()),
        'soc_min':float(o['s'].min()),'soc_max':float(o['s'].max()),
        'power_max_kw':float(max(o['c'].max(),o['d'].max())*6),
        'simultaneous_max':float(np.minimum(o['c'],o['d']).max()),
        'continuity_max_abs':float(np.abs(o['s'][1:,0]-o['s'][:-1,-1]).max()),
        'minimum_flow':float(min(o[k].min() for k in ['g','a','c','d','e','w']))}
    errors['pass']=bool(errors['energy_max_abs']<1e-6 and errors['state_max_abs']<1e-6 and errors['cost_max_abs']<1e-6 and errors['soc_min']>=1200-1e-6 and errors['soc_max']<=10800+1e-6 and errors['power_max_kw']<=5000+1e-6 and errors['simultaneous_max']<1e-6 and errors['continuity_max_abs']<1e-6 and errors['minimum_flow']>=-1e-6)
    return errors

def independent_q1(a):
    # Separate net-state formulation g + eta*withdraw - deposit/eta - waste = net.
    n=144;eta=.9;M=5*n+1
    A=lil_matrix((2*n,M));b=np.r_[(a['base'][:,1]-a['base'][:,2])/6,np.zeros(n)]
    for t in range(n):
        A[t,t]=1;A[t,n+t]=-1/eta;A[t,2*n+t]=eta;A[t,3*n+t]=-1
        A[n+t,4*n+t+1]=1;A[n+t,4*n+t]=-1;A[n+t,n+t]=-1;A[n+t,2*n+t]=1
    c=np.r_[a['base'][:,0],np.zeros(4*n+1)]
    bd=[(0,None)]*n+[(0,5000/6*eta)]*n+[(0,5000/6/eta)]*n+[(0,None)]*n+[(1200,10800)]*(n+1)
    bd[4*n]=bd[5*n]=(6000,6000)
    sol=linprog(c,A_eq=A.tocsr(),b_eq=b,bounds=bd,method='highs-ipm')
    assert sol.success
    return float(sol.fun)

def main():
    a=read_inputs(ROOT/'data/inputs.npz');report={}
    names=['q2','q3','q4_2','q4_3','only0','add6','add12','add18',
           'refund','refund_q3','refund_q4_3','no_newpv']+[f'{t}_q{q}' for t in ['q2','q3','q4_2','q4_3'] for q in QS]
    for name in names:
        f=ROOT/f'results/{name}.npz'
        if not f.exists():continue
        o=dict(np.load(f));ids=o['days'];p=a['price'][ids] if name.startswith('q4') or name=='refund_q4_3' else np.tile(a['base'][:,0],(len(ids),1))
        report[name]=check(o,a['load'][ids],a['pv'][ids],p,name=='refund' or name.startswith('refund_'))
        assert report[name]['pass'],(name,report[name])
    q1=dict(np.load(ROOT/'results/q1.npz'))
    p=a['base'][:,0];n=(a['base'][:,1]-a['base'][:,2])/6
    opt=independent_q1(a)
    qcheck={'independent_cost':opt,'saved_cost':float(p@q1['g']),
      'difference':abs(opt-float(p@q1['g'])),'balance_max_abs':float(np.abs(q1['g']-q1['c']+q1['d']-q1['w']-n).max()),
      'state_max_abs':float(np.abs(np.diff(q1['s'])-.9*q1['c']+q1['d']/.9).max()),
      'simultaneous_max':float(np.minimum(q1['c'],q1['d']).max()),
      'initial':float(q1['s'][0]),'final':float(q1['s'][-1])}
    assert qcheck['difference']<1e-5 and qcheck['balance_max_abs']<1e-6 and qcheck['state_max_abs']<1e-6
    assert qcheck['simultaneous_max']<1e-5
    report['q1']=qcheck
    # Mutation checks cover each public prediction at its information cutoff.
    mutation=[]
    for use in [False,True]:
        for issue in range(4 if use else 1):
            day=171;k=36*issue;b={key:v.copy() for key,v in a.items()}
            b['load'][day,k:]+=20000;b['pv'][day,k:]+=10000
            b['load'][day+1:]+=17000;b['pv'][day+1:]+=13000;b['price'][day:]+=9
            b['forecast'][day,issue+1:]+=9000;b['forecast'][day+1:]+=9000
            for x,y in zip(predict(a,day,issue,use),predict(b,day,issue,use)):
                mutation.append(float(np.max(np.abs(x-y))))
    assert max(mutation)==0
    report['information_set']={'mutations':len(mutation),'max_difference':max(mutation),'pass':True}
    # Physical branch boundaries are checked with known values.
    checks=[step(0,1000,1200),step(2000,0,10800),step(0,0,6000),step(0,1000,10800)]
    assert checks[0][2]==1000 and checks[1][3]==2000 and checks[2][-1]==6000 and abs(checks[3][1]-5000/6)<1e-8
    report['boundary_tests']={'count':4,'pass':True}
    # Exact repeatability of one representative full-day LP.
    one=plan(n,p,6000);two=plan(n,p,6000)
    report['repeatability']={'max_difference':float(np.max(np.abs(one['g']-two['g'])))}
    assert report['repeatability']['max_difference']==0
    (ROOT/'results/validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
