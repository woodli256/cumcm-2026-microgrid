# -*- coding: utf-8 -*-
"""固定训练选择后，对原合同运行三种控制方案。"""
from risk_mpc import *
a=read_inputs(ROOT/'data/inputs.npz');prep=prepare(a);rp=ROOT/'results/risk_mpc.json'
report=json.loads(rp.read_text(encoding="utf-8"))
clipped=entries=0
for day in range(31,365):
    ids=np.unique(np.linspace(max(1,day-28),day-1,min(SCENARIOS,day-1),dtype=int))
    for t in range(0,144,6):
        end=min(144,t+HORIZON)
        raw=prep[1][day,t+1:end][None,:]+prep[2][ids,t+1:end]
        clipped+=int((raw<.001).sum());entries+=raw.size
report['scenario_price_clipping']={'floor_yuan_per_kwh':.001,'clipped':clipped,
    'forecast_price_entries':entries,'fraction':clipped/entries,
    'scope':'334 days x 24 updates; future intervals only; identical price scenarios for q4-2 and q4-3'}
report['test']={};report['training_scope']='Jan15-31; baseline also eligible for recommendation'
for name in ['q2','q3','q4_2','q4_3']:
    orig=dict(np.load(ROOT/f'results/{name}.npz'))
    o=run(a,prep,name,0.,end=31,mode='nominal')
    train=report['training'][name];train['nominal_cost']=float(o['cost'][14:].sum())
    train['selected_positive_weight']=min([x for x in train['candidates'] if x['weight']>0],key=lambda x:x['cost'])['weight']
    candidate_costs={'baseline':train['baseline_cost'],'nominal':train['nominal_cost'],'scenario_mean':next(x['cost'] for x in train['candidates'] if x['weight']==0),'scenario_cvar':min(x['cost'] for x in train['candidates'] if x['weight']>0)}
    train['recommended_policy']=min(candidate_costs,key=candidate_costs.get)
    rp.write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding="utf-8")
    report['test'][name]={'baseline':metrics(orig),'variants':{}}
    for label,mode,weight in [('nominal','nominal',0.),('scenario_mean','scenario',0.),('scenario_cvar','scenario',train['selected_positive_weight'])]:
        start=time.monotonic();o=run(a,prep,name,weight,mode=mode);elapsed=time.monotonic()-start
        checks=audit(a,o,name,orig);out=ROOT/f'results/risk_mpc_{name}_{label}.npz'
        np.savez_compressed(out,**o)
        met=metrics(o);base=metrics(orig)
        report['test'][name]['variants'][label]={'weight':weight,'metrics':met,'saving':base['cost']-met['cost'],'saving_percent':100*(base['cost']-met['cost'])/base['cost'],'checks':checks,'seconds':elapsed}
        rp.write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding="utf-8")
        print('FINISHED',name,label,report['test'][name]['variants'][label],flush=True)
