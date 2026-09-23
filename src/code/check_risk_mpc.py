# -*- coding: utf-8 -*-
"""独立复核风险控制。电芯侧重建 LP、未来真值扰动、全样本逐段验算。"""
from pathlib import Path
import sys,json,hashlib,argparse
sys.dont_write_bytecode=True
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
import risk_mpc as rm
from solve import ROOT,DT,ETA,LO,HI,POWER,plan,safety

def independent_lp(net,contract,price,initial,weight,billed):
    # Independent cell-side variables e, h=eta*c, k=d/eta, w, s.
    m,n=net.shape;nv=5*n+1;N=m*nv+int(weight>0)
    eq=lil_matrix((2*m*n+m-1,N));b=np.zeros(eq.shape[0]);obj=np.zeros(N);bounds=[]
    loss=lil_matrix((m,N));const=price@billed
    for z in range(m):
        q=z*nv;r=net[z]-contract;pos=np.maximum(r,0);neg=np.maximum(-r,0)
        bounds.extend([(0,float(v)) for v in pos])
        bounds.extend([(0,float(ETA*min(v,POWER*DT))) for v in neg])
        bounds.extend([(0,float(min(v,POWER*DT)/ETA)) for v in pos])
        bounds.extend([(0,float(v)) for v in neg]);bounds.extend([(LO,HI)]*(n+1));bounds[q+4*n]=(initial,initial)
        for t in range(n):
            row=z*2*n+t;eq[row,q+t]=1;eq[row,q+n+t]=-1/ETA;eq[row,q+2*n+t]=ETA;eq[row,q+3*n+t]=-1;b[row]=r[t]
            row=z*2*n+n+t;eq[row,q+4*n+t+1]=1;eq[row,q+4*n+t]=-1;eq[row,q+n+t]=-1;eq[row,q+2*n+t]=1
        obj[q:q+n]=5*price[z]/m;loss[z,q:q+n]=5*price[z]
    for z in range(1,m):eq[2*m*n+z-1,z*nv+4*n+min(6,n)]=1;eq[2*m*n+z-1,4*n+min(6,n)]=-1
    ub=None;bu=None
    if weight>0:
        assert m==5 and rm.ALPHA==.8
        obj[-1]=weight;bounds.append((0,None));loss[:,-1]=-1;ub=loss.tocsr();bu=-const
    res=linprog(obj,A_eq=eq.tocsr(),b_eq=b,A_ub=ub,b_ub=bu,bounds=bounds,method='highs-ipm')
    assert res.success,res.message
    states=np.stack([res.x[z*nv+4*n:z*nv+5*n+1] for z in range(m)])
    losses=const+np.array([5*price[z]@res.x[z*nv:z*nv+n] for z in range(m)])
    return {'objective':float(res.fun+const.mean()),'equality_residual':float(np.max(np.abs(eq@res.x-b))),
        'shared_end_range':float(np.ptp(states[:,min(6,n)])), 'trajectory':states[:,1:min(6,n)+1].mean(0).tolist(),
        'scenario_losses':losses.tolist(),'cvar':float(losses.max()),'contract_cost_range':float(np.ptp(const))}

def issued(a,prep,orig,day,t,use,dynamic):
    g=orig['g'][day].copy();active=g.copy()
    if use:
        for k in [36,72,108]:
            if k>t:break
            issue=k//36;pred,err=prep[0][use]
            p=prep[1][day,k:] if dynamic else a['base'][k:,0]
            active[k:]=plan(pred[day,issue,k:]+safety(err,day,issue,.65),p,float(orig['s'][day,k]),original=g[k:])['g']
    return active

def core_checks(a,prep):
    cases=[];causality=[]
    for name,day,t,weight,initial in [('q2',20,0,0.,6000.),('q3',40,36,.5,5000.),('q4_2',120,60,1.,7200.),('q4_3',240,108,.5,6000.),('q3',78,138,.5,5800.)]:
        orig=dict(np.load(ROOT/f'results/{name}.npz'));use=name in ['q3','q4_3'];dynamic=name.startswith('q4')
        active=issued(a,prep,orig,day,t,use,dynamic);net,pr=rm.scenarios(a,prep,day,t,use,dynamic,active[t:]);n=net.shape[1]
        aa=active[t:t+n];gg=orig['g'][day,t:t+n];billed=gg+1.5*np.maximum(aa-gg,0)+.5*np.maximum(gg-aa,0)
        trajectory,value,res=rm.reserve_lp(net,aa,pr,initial,weight,billed,method='highs-ds')
        ipm=independent_lp(net,aa,pr,initial,weight,billed)
        gap=abs(value-ipm['objective']);assert gap<1e-6,(name,value,ipm)
        assert ipm['shared_end_range']<1e-6 and ipm['equality_residual']<1e-6
        # Constant storage is a feasible shared-end witness: curtail every surplus and purchase every deficit.
        excess=aa[None,:]-net;e=np.maximum(-excess,0);w=np.maximum(excess,0)
        witness=float(np.max(np.abs(aa+e-w-net)));assert witness<1e-8
        cases.append({'name':name,'day_index':day,'time_index':t,'weight':weight,'primary_objective':value,'independent_cellside_ipm':ipm,
            'objective_gap':gap,'primary_residual':res,'constant_state_feasibility_residual':witness,
            'trajectory_solver_difference':float(np.max(np.abs(trajectory-np.array(ipm['trajectory']))))})
        altered={k:v.copy() for k,v in a.items()}
        for key in ['load','pv','price']:
            altered[key][day,t+1:]+=12345.67;altered[key][day+1:]+=23456.78
        issue=t//36 if use else 0
        altered['forecast'][day,issue+1:]+=34567.89;altered['forecast'][day+1:]+=45678.9
        prep2=rm.prepare(altered);active2=issued(altered,prep2,orig,day,t,use,dynamic)
        net2,pr2=rm.scenarios(altered,prep2,day,t,use,dynamic,active2[t:])
        tr2,value2,_=rm.reserve_lp(net2,active2[t:t+n],pr2,initial,weight,billed,method='highs-ds')
        errors={'net':float(np.max(abs(net-net2))),'price':float(np.max(abs(pr-pr2))),'issued_contract':float(np.max(abs(active-active2))),
            'target':float(np.max(abs(trajectory-tr2))),'objective':abs(value-value2)}
        assert max(errors.values())==0.,errors
        ids=np.unique(np.linspace(max(1,day-28),day-1,min(rm.SCENARIOS,day-1),dtype=int))
        pred,err=prep[0][use];expected=pred[day,issue,t:t+n][None,:]+err[ids,issue,t:t+n]
        expected[:,0]=(a['load'][day,t]-a['pv'][day,t])*DT
        assert np.array_equal(net,expected) and ids.max()<day
        if dynamic:
            expectp=np.maximum(.001,prep[1][day,t:t+n]+prep[2][ids,t:t+n]);expectp[:,0]=a['price'][day,t]
            assert np.array_equal(pr,expectp)
        nominal,pn=rm.scenarios(a,prep,day,t,use,dynamic,active[t:],mode='nominal')
        nominal2,pn2=rm.scenarios(altered,prep2,day,t,use,dynamic,active2[t:],mode='nominal')
        tn,vn,_=rm.reserve_lp(nominal,aa,pn,initial,0.,billed,method='highs-ds')
        tn2,vn2,_=rm.reserve_lp(nominal2,aa,pn2,initial,0.,billed,method='highs-ds')
        ni=independent_lp(nominal,aa,pn,initial,0.,billed)
        assert abs(vn-ni['objective'])<1e-6
        nominal_errors={'net':float(np.max(abs(nominal-nominal2))),'price':float(np.max(abs(pn-pn2))),'target':float(np.max(abs(tn-tn2))),'objective':abs(vn-vn2)}
        assert max(nominal_errors.values())==0.
        causality.append({'name':name,'day_index':day,'time_index':t,'strictly_past_paired_ids':ids.tolist(),'future_mutation_errors':errors,'nominal_mutation_errors':nominal_errors,'nominal_cellside_objective_gap':abs(vn-ni['objective'])})
    rng=np.random.default_rng(20260911);losses=rng.uniform(0,100000,(1000,5));err=0.
    for row in losses:
        z=np.r_[0.,row];values=z+np.maximum(row[None,:]-z[:,None],0).sum(1)/(5*(1-rm.ALPHA));err=max(err,abs(values.min()-row.max()))
    assert err<1e-8
    return {'representative_lps':cases,'future_truth_mutation':causality,'empirical_cvar_worst_case_max_error_1000_examples':err}

def output_checks(a,report):
    results={};pending=[]
    for name,label in [(name,label) for name in ['q2','q3','q4_2','q4_3'] for label in ['nominal','scenario_mean','scenario_cvar']]:
        key=name+'_'+label
        path=ROOT/f'results/risk_mpc_{name}_{label}.npz'
        if not path.exists() or label not in report.get('test',{}).get(name,{}).get('variants',{}):pending.append(key);continue
        o=dict(np.load(path));orig=dict(np.load(ROOT/f'results/{name}.npz'))
        assert o['s'].shape==(365,145) and o['reserve'].shape==(365,144), (name,'old controller or incomplete output')
        net=(a['load']-a['pv'])*DT;price=a['price'] if name.startswith('q4') else np.tile(a['base'][:,0],(365,1))
        cost=np.column_stack([(price*o['g']).sum(1),5*(price*o['e']).sum(1),1.5*(price*o['up']).sum(1),.5*(price*o['down']).sum(1)])
        reserve=o['reserve'];state=o['s'][:,:144];bal=o['a']-net
        expected_c=np.where(bal>=0,np.minimum(np.minimum(bal,POWER*DT),np.maximum(0,(HI-state)/ETA)),0)
        expected_d=np.where(bal<0,np.minimum(np.minimum(-bal,POWER*DT),np.maximum(0,ETA*(state-np.maximum(LO,reserve)))),0)
        checks={'energy_balance':float(np.max(abs(o['a']+o['e']+o['d']-o['c']-o['w']-net))),
            'storage_equation':float(np.max(abs(np.diff(o['s'],axis=1)-ETA*o['c']+o['d']/ETA))),
            'nonnegativity_violation':float(max(0.,max(-o[k].min() for k in ['g','a','c','d','e','w','up','down']))),
            'storage_power_bounds':float(max(0.,LO-o['s'].min(),o['s'].max()-HI,o['c'].max()-POWER*DT,o['d'].max()-POWER*DT)),
            'reserve_bounds':float(max(0.,LO-reserve.min(),reserve.max()-HI)),
            'simultaneous_charge_discharge':float(np.minimum(o['c'],o['d']).max()),'initial':float(abs(o['s'][0,0]-6000)),
            'interday_continuity':float(np.max(abs(o['s'][:-1,-1]-o['s'][1:,0]))),'cost':float(np.max(abs(o['cost']-cost))),
            'dayahead_contract_difference':float(np.max(abs(o['g']-orig['g']))),'executed_contract_difference':float(np.max(abs(o['a']-orig['a']))),
            'adjustments':float(max(np.max(abs(o['up']-np.maximum(o['a']-o['g'],0))),np.max(abs(o['down']-np.maximum(o['g']-o['a'],0))))),
            'execution_charge_rule':float(np.max(abs(o['c']-expected_c))),'execution_discharge_rule':float(np.max(abs(o['d']-expected_d))),
            'emergency_rule':float(np.max(abs(o['e']-np.maximum(-bal,0)+o['d']))),'warmup_reserve':float(np.max(abs(reserve[:7]-LO)))}
        assert max(checks.values())<1e-6,(name,checks)
        tr=report['training'][name];chosen=min([x for x in tr['candidates'] if x['weight']>0],key=lambda x:x['cost'])
        record=report['test'][name]['variants'][label]
        assert record['weight']==(chosen['weight'] if label=='scenario_cvar' else 0)
        training_costs={'baseline':tr['baseline_cost'],'nominal':tr['nominal_cost'],'scenario_mean':next(x['cost'] for x in tr['candidates'] if x['weight']==0),'scenario_cvar':chosen['cost']}
        assert tr['recommended_policy']==min(training_costs,key=training_costs.get)
        monthly=o['cost'][31:].sum()
        assert abs(monthly-record['metrics']['cost'])<1e-6
        assert abs(orig['cost'][31:].sum()-monthly-record['saving'])<1e-6
        assert len(o['cost'][31:])==334
        results[key]={'checks':checks,'training_start':'01-15','training_end':'01-31','test_start':'02-01','test_days':334,'selected_weight':record['weight'],'training_recommended_policy':tr['recommended_policy'],
            'test_cost':float(monthly),'saving':float(record['saving']),'daily_tail_count':17,
            'trajectory_reference_unattained_count':int(np.count_nonzero(o['s'][:,1:]+1e-6<reserve)),
            'output_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    return results,pending

def main():
    p=argparse.ArgumentParser();p.add_argument('--outputs-only',action='store_true');p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    auditpath=ROOT/'audit/风险控制独立检查.json';a=rm.read_inputs(ROOT/'data/inputs.npz')
    if args.outputs_only:
        audit=json.loads(auditpath.read_text(encoding="utf-8"))
        assert audit['risk_code_sha256']==hashlib.sha256((ROOT/'code/risk_mpc.py').read_bytes()).hexdigest(),'risk code changed; rerun complete checks'
    else:
        audit=core_checks(a,rm.prepare(a))
    report=json.loads((ROOT/'results/risk_mpc.json').read_text(encoding="utf-8"));out,pending=output_checks(a,report)
    audit.update({'output_checks':out,'pending_outputs':pending,'status':'PASS' if not pending else 'CORE_PASS_OUTPUTS_PENDING',
        'risk_code_sha256':hashlib.sha256((ROOT/'code/risk_mpc.py').read_bytes()).hexdigest(),
        'scope_notes':['全预见场景内部控制只用于生成储量参考，不是多阶段非预见策略树。','平均储量轨迹是实际放电保留参考，不保证实际储量达到该轨迹。','固定基准合同的控制消融，不能宣称联合优化合同。','评价最差 5% 日均值用 334 天中最差 17 天，与优化中 5 场景 alpha=0.8 的 CVaR 不同。']})
    auditpath.write_text(json.dumps(audit,ensure_ascii=False,indent=2), encoding="utf-8");print(json.dumps({'status':audit['status'],'outputs':list(out),'pending':pending},ensure_ascii=False))
    if args.require_complete:assert not pending,pending
if __name__=='__main__':main()
