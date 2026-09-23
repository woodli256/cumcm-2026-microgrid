# -*- coding: utf-8 -*-
"""预先声明的发布时刻、分位数与结算假设消融。"""
from solve import *

def isolate_forecast():
    a=read_inputs(ROOT/'data/inputs.npz')
    for day in range(365):
        initial=a['forecast'][day,0].copy()
        for issue in range(1,4):
            hours=24-6*issue
            a['forecast'][day,issue,:hours]=initial[6*issue:]
    fc=forecasts(a)
    summary=json.loads((ROOT/'results/summary.json').read_text(encoding="utf-8"))
    o=run(a,fc,True,False,summary['training']['q3']['main_q'])
    np.savez_compressed(ROOT/'results/no_newpv.npz',**o)
    path=ROOT/'results/sensitivity.json'
    report=json.loads(path.read_text(encoding="utf-8"))
    report['no_newpv']=totals(o)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding="utf-8")
    print('no_newpv',totals(o)['total_cost'],flush=True)

def main():
    a=read_inputs(ROOT/'data/inputs.npz');fc=forecasts(a)
    summary=json.loads((ROOT/'results/summary.json').read_text(encoding="utf-8"))
    main_q={k:v['main_q'] for k,v in summary['training'].items()}
    report={'forecast_updates':{},'quantile':{},'quantile_all':{},'q1_assumptions':{},'refund':{},'main_q':main_q}
    q=main_q['q3']
    for name,updates in [('only0',()),('add6',(1,)),('add12',(2,)),('add18',(3,))]:
        o=run(a,fc,True,False,q,updates=updates)
        np.savez_compressed(ROOT/f'results/{name}.npz',**o)
        report['forecast_updates'][name]=totals(o)
        print(name,totals(o)['total_cost'],flush=True)
    # 四个问题在 334 天测试期上的完整分位数曲线，用于说明主参数的选择依据。
    for name,use,dynamic in [('q2',False,False),('q3',True,False),('q4_2',False,True),('q4_3',True,True)]:
        curve={}
        for qq in QS:
            if qq==main_q[name]:
                curve[str(qq)]=summary['strategies'][name]
            else:
                f=ROOT/f'results/{name}_q{qq}.npz'
                o=dict(np.load(f)) if f.exists() else run(a,fc,use,dynamic,qq)
                np.savez_compressed(f,**o)
                curve[str(qq)]=totals(o)
            print('curve',name,qq,round(curve[str(qq)]['total_cost'],2),flush=True)
        report['quantile_all'][name]=curve
        if name=='q2':
            report['quantile']=curve
    for name,eta,s0 in [('roundtrip90',np.sqrt(.9),6000),('initial3000',.9,3000),('initial9000',.9,9000)]:
        p=plan((a['base'][:,1]-a['base'][:,2])*DT,a['base'][:,0],s0,eta=eta,target=s0)
        report['q1_assumptions'][name]={'cost':float(a['base'][:,0]@p['g']),'g':float(p['g'].sum())}
    o=run(a,fc,True,False,main_q['q3'],refund=True)
    np.savez_compressed(ROOT/'results/refund.npz',**o)
    report['refund']=totals(o)
    # Point forecast accuracy uses unbuffered predictions and only test days.
    report['forecast_accuracy']={}
    true=(a['load']-a['pv'])*DT
    for use,label in [(False,'history'),(True,'forecast')]:
        pred=fc[use][0]
        for issue in range(4 if use else 1):
            k=issue*36;err=(pred[31:,issue,k:]-true[31:,k:])/DT
            report['forecast_accuracy'][f'{label}_{6*issue}']={'mae_kw':float(np.abs(err).mean()),'rmse_kw':float(np.sqrt((err**2).mean()))}
    # Price forecasts use only completed prior days.
    ph=np.array([predict(a,d,0,False)[1] for d in range(31,365)])
    report['price_forecast_mae']=float(np.abs(ph-a['price'][31:]).mean())
    (ROOT/'results/sensitivity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding="utf-8")

if __name__=='__main__':
    import sys
    if '--isolate-only' not in sys.argv:main()
    isolate_forecast()
