# -*- coding: utf-8 -*-
"""从验证结果生成报告数据与工作簿数值，不生成旧版图形。"""
from solve import *
import datetime as dt
DATES=['2025-03-20','2025-06-21','2025-09-23','2025-12-21']

def hm(t):return f'{t//6}:{t%6*10:02d}'
def interval(t,end=None):return f'{hm(t)}-{hm(t+1 if end is None else end)}'
def events(e):
    rows=[];t=0
    while t<144:
        if e[t]<=1e-6:t+=1;continue
        start=t
        while t<144 and e[t]>1e-6:t+=1
        rows.append([interval(start,t),float(e[start:t].sum())])
    return rows

def daily(o,i):
    return {'g6':[float(o['g'][i,t]) for t in [60,72,84,96,108,120]],
      'a6':[float(o['a'][i,t]) for t in [60,72,84,96,108,120]],
      'g':float(o['g'][i].sum()),'a':float(o['a'][i].sum()),
      'cost':o['cost'][i].tolist(),'total':float(o['cost'][i].sum()),
      'blocks':[[float(o['c'][i,j:j+24].sum()),float(o['d'][i,j:j+24].sum())] for j in range(0,144,24)],
      's0':float(o['s'][i,0]),'s24':float(o['s'][i,-1]),'events':events(o['e'][i])}

def main():
    a=read_inputs(ROOT/'data/inputs.npz');report={'dates':DATES,'daily':{}};exports={}
    q1=dict(np.load(ROOT/'results/q1.npz'))
    report['q1']={'g6':[float(q1['g'][t]) for t in [60,72,84,96,108,120]],
        'g':float(q1['g'].sum()),'cost':float(a['base'][:,0]@q1['g']),
        'blocks':[[float(q1['c'][j:j+24].sum()),float(q1['d'][j:j+24].sum())] for j in range(0,144,24)],'s0':6000.,'s24':6000.}
    exports['result1']={'计划购电量':[[interval(i),float(q1['g'][i])] for i in range(144)],
        '充放电量':[[interval(j,j+24),float(q1['c'][j:j+24].sum()),float(q1['d'][j:j+24].sum()),'0:00' if j==0 else '24:00' if j==24 else None,6000 if j in [0,24] else None] for j in range(0,144,24)]}
    labels={'q2':'问题2','q3':'问题3','q4_2':'问题4-2','q4_3':'问题4-3'}
    monthly={};all_o={}
    for name,file in [('q2','result2'),('q3','result3'),('q4_2','result4-2'),('q4_3','result4-3')]:
        o=dict(np.load(ROOT/f'results/{name}.npz'));all_o[name]=o
        report['daily'][name]={date:daily(o,(dt.date.fromisoformat(date)-dt.date(2025,1,1)).days) for date in DATES}
        wb={'计划购电量':[],'充放电量':[],'紧急购电量':[]}
        if name in ['q3','q4_3']:wb['调整购电量']=[]
        for i in range(31,365):
            date=(dt.date(2025,1,1)+dt.timedelta(days=i)).isoformat()
            wb['计划购电量'].append([date,*o['g'][i].tolist(),float(o['g'][i].sum()),float(o['cost'][i].sum())])
            if '调整购电量' in wb:
                wb['调整购电量'].append([date,*o['a'][i].tolist(),float(o['a'][i].sum()),float(o['cost'][i].sum())])
            for j in range(0,144,24):
                wb['充放电量'].append([date if j==0 else None,interval(j,j+24),float(o['c'][i,j:j+24].sum()),float(o['d'][i,j:j+24].sum()),'0:00' if j==0 else '24:00' if j==24 else None,float(o['s'][i,0]) if j==0 else float(o['s'][i,-1]) if j==24 else None])
            ev=events(o['e'][i]) or [['无',0.]]
            for j,row in enumerate(ev):wb['紧急购电量'].append([date if j==0 else None,*row])
        exports[file]=wb
        monthly[name]=[]
        for m in range(2,13):
            ids=[i for i in range(31,365) if (dt.date(2025,1,1)+dt.timedelta(days=i)).month==m]
            monthly[name].append(float(o['cost'][ids].sum()))
    report['monthly']=monthly
    (ROOT/'results/report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding="utf-8")
    (ROOT/'results/export_values.json').write_text(json.dumps(exports,ensure_ascii=False), encoding="utf-8")
    print('report and export payload ready')

if __name__=='__main__':main()
