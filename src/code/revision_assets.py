# -*- coding: utf-8 -*-
"""由保存数组生成题定正文表和简洁决策图，保留原策略正式结果。"""
import sys,json,hashlib
from pathlib import Path
from datetime import date,timedelta
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'_utils'))
from paper_figure_style import setup,clean,paper_fig,BLUE,ORANGE,GREEN,PURPLE,INK,BLUE_L
from paper_figure_style import save_bundle as ps_save_bundle
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from solve import read_inputs,forecasts,safety,plan,step
OUT=ROOT/'results/revision'; TABLES=ROOT/'tables';TABLES.mkdir(exist_ok=True)
NAMES=['q2','q3','q4_2','q4_3'];LABELS=['问题二','问题三','问题四-2','问题四-3']
DAYS=[78,171,265,354]
def time(t):return f'{t//6}:{{}}'.format(f'{t%6*10:02d}')
def interval(t,u=None):return f'{time(t)}--{time(t+1 if u is None else u)}'
def num(x):return f'{float(x):.2f}'
def line(items):return ' & '.join(map(str,items))+r' \\'+'\n'
def save(fig,stem,source):
    info=ps_save_bundle(fig,stem,{}, {'sources':list(source)})
    meta={'sources':source,'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source},
          'width_inches':round(fig.get_figwidth(),3),'data_status':'real','png_dpi':info['png_dpi'],
          'min_effective_font_pt':info['min_effective_font_pt']}
    (OUT/(stem+'.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2), encoding="utf-8")
def answer_block(g,c,d,s,cost,title,label):
    out=r'\begin{table}[H]\centering\small'+'\n'+r'\caption{'+title+r'}\label{'+label+'}\n'
    out+=r'\begin{tabular}{lrlrlr}\toprule'+'\n'
    out+=line(['时间段','购电量','时间段','购电量','时间段','购电量'])+r'\midrule'+'\n'
    for row in [[60,72,84],[96,108,120]]:
        out+=line([v for t in row for v in [interval(t),num(g[t])]])
    out+=r'\midrule'+'\n'+line(['全天购电量',num(g.sum()),r'\multicolumn{2}{l}{全天总购电费（元）}',r'\multicolumn{2}{r}{'+num(cost)+'}'])
    out+=r'\bottomrule\end{tabular}\par\vspace{4pt}'+'\n'
    out+=r'\begin{tabular}{lrrlrr}\toprule'+'\n'+line(['时间段','充电量','放电量','时间段','充电量','放电量'])+r'\midrule'+'\n'
    for row in [[0,24],[48,72],[96,120]]:
        out+=line([v for t in row for v in [interval(t,t+24),num(c[t:t+24].sum()),num(d[t:t+24].sum())]])
    out+=r'\midrule'+'\n'+line(['0:00储电量',r'\multicolumn{2}{r}{'+num(s[0])+'}','24:00储电量',r'\multicolumn{2}{r}{'+num(s[-1])+'}'])
    return out+r'\bottomrule\end{tabular}\end{table}'+'\n'
def emergency_table(o,title,label):
    groups=[]
    for day in DAYS:
        e=o['e'][day];pos=np.flatnonzero(e>1e-6);rows=[]
        for block in np.split(pos,np.flatnonzero(np.diff(pos)>1)+1):
            if len(block):rows.append([interval(int(block[0]),int(block[-1])+1),num(e[block].sum())])
        groups.append(rows or [['无','0.00']])
    out=r'\begin{table}[H]\centering\small\setlength{\tabcolsep}{3pt}'+'\n'+r'\caption{'+title+r'}\label{'+label+'}\n'+r'\begin{tabular}{lr lr lr lr}\toprule'+'\n'
    out+=line([r'\multicolumn{2}{c}{'+(date(2025,1,1)+timedelta(days=d)).strftime('%m月%d日')+'}' for d in DAYS])
    out+=line(['时间段','购电量']*4)+r'\midrule'+'\n'
    for r in range(max(map(len,groups))):out+=line([x for g in groups for x in (g[r] if r<len(g) else ['',''])])
    out+=r'\midrule'+'\n'+line([x for d in DAYS for x in ['合计',num(o['e'][d].sum())]])
    return out+r'\bottomrule\end{tabular}\end{table}'+'\n'
def make_tables():
    q1=dict(np.load(ROOT/'results/q1.npz'));a=read_inputs(ROOT/'data/inputs.npz')
    (TABLES/'q1_required.tex').write_text(answer_block(q1['g'],q1['c'],q1['d'],q1['s'],a['base'][:,0]@q1['g'],'问题一指定时段购电与储能结果','tab:q1-required'), encoding="utf-8")
    for name,title in [('q2','问题二'),('q3','问题三')]:
        o=dict(np.load(ROOT/'results'/f'{name}.npz'));text=''
        for day in DAYS:
            stamp=(date(2025,1,1)+timedelta(days=day)).strftime('%m月%d日')
            text+=answer_block(o['g'][day],o['c'][day],o['d'][day],o['s'][day],o['cost'][day].sum(),f'{title}{stamp}指定结果',f'tab:{name}-{day}')
        text+=emergency_table(o,f'{title}四个指定日期紧急购电明细',f'tab:{name}-emergency')
        (TABLES/f'{name}_required.tex').write_text(text, encoding="utf-8")
    text=r'\begin{table}[H]\centering\small\caption{波动电价下指定日期的费用与储量}\label{tab:q4-dates}'+'\n'+r'\begin{tabular}{llrrrr}\toprule'+'\n'+line(['策略','日期','计划费用','紧急费用','调整费用','总费用'])+r'\midrule'+'\n'
    for name,title in [('q4_2','问题四-2'),('q4_3','问题四-3')]:
        o=dict(np.load(ROOT/'results'/f'{name}.npz'))
        for day in DAYS:
            cost=o['cost'][day];text+=line([title,(date(2025,1,1)+timedelta(days=day)).strftime('%m-%d'),num(cost[0]),num(cost[1]),num(cost[2:].sum()),num(cost.sum())])
    text+=r'\bottomrule\end{tabular}\end{table}'+'\n'
    (TABLES/'q4_required.tex').write_text(text.replace('费用与储量','费用分解（元）'), encoding="utf-8")

def system_figure():
    fig,ax=plt.subplots(figsize=paper_fig(6.3,2.6));ax.set(xlim=(0,10),ylim=(-.2,3.6));ax.axis('off')
    boxes={'光伏发电':(0.2,2.5,1.65,.65),'外网购电':(.2,.7,1.65,.65),'微网母线':(3.2,1.7,1.7,.65),'小区负载':(7.8,2.6,1.8,.65),'未利用电量':(7.8,.6,1.8,.65),'储能设备':(4.0,.05,1.8,.65)}
    for label,(x,y,w,h) in boxes.items():
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor='#E8F1F5',edgecolor=BLUE,lw=.9));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11)
    def arrow(p,q,label,xy):
        ax.annotate('',xy=q,xytext=p,arrowprops={'arrowstyle':'->','color':BLUE,'lw':1.3});ax.text(*xy,label,fontsize=9,ha='center',va='center',bbox={'facecolor':'white','edgecolor':'none','pad':1})
    arrow((1.95,2.8),(3.15,2.2),'光伏电量',(2.65,2.85))
    arrow((1.95,1.05),(3.15,1.8),'计划、调整和紧急购电',(2.2,.5))
    arrow((4.98,2.15),(7.7,2.9),'供应负载',(6.45,2.9))
    arrow((4.98,1.9),(7.7,1),'剩余电量',(6.45,1.05))
    arrow((4.15,1.62),(4.35,.77),'充电',(3.85,1.15))
    arrow((5.55,.65),(4.85,1.62),'放电',(5.55,1.3))
    fig.subplots_adjust(0,.02,1,.98);save(fig,'fig_system_overview',['code/solve.py'])
def dispatch_figure():
    a=read_inputs(ROOT/'data/inputs.npz');o=dict(np.load(ROOT/'results/q1.npz'));x=np.arange(144)/6
    fig,axs=plt.subplots(4,1,figsize=paper_fig(6.3,5.6),sharex=True,gridspec_kw={'height_ratios':[1.4,1,1,.8]})
    axs[0].plot(x,a['base'][:,1]/1000,label='负载',color=BLUE);axs[0].plot(x,a['base'][:,2]/1000,label='光伏',color=GREEN);axs[0].step(x,o['g']*6/1000,label='购电',color=ORANGE,where='post');axs[0].set_ylabel('功率 / MW');axs[0].legend(ncol=3,loc='upper right')
    axs[1].bar(x,o['c']*6/1000,width=1/6,color=GREEN,label='充电');axs[1].bar(x,-o['d']*6/1000,width=1/6,color=ORANGE,label='放电');axs[1].set_ylabel('功率 / MW');axs[1].legend(ncol=2,loc='upper right')
    axs[2].plot(np.arange(145)/6,o['s']/1000,color=PURPLE);axs[2].set_ylabel('储量 / MWh');axs[2].axhline(1.2,color='gray',ls=':',lw=.8);axs[2].axhline(10.8,color='gray',ls=':',lw=.8)
    axs[3].step(np.r_[x,24],np.r_[a['base'][:,0],a['base'][-1,0]],where='post',color=BLUE);axs[3].set_ylabel('元 / kWh');axs[3].set_xlabel('时刻 / h')
    for ax in axs:clean(ax);ax.set_xlim(0,24);ax.set_xticks(np.arange(0,25,2))
    fig.tight_layout(h_pad=.6);save(fig,'fig_dispatch',['data/inputs.npz','results/q1.npz'])
def simple_figures():
    a=read_inputs(ROOT/'data/inputs.npz');net=(a['load']-a['pv'])/1000
    months=np.array([(date(2025,1,1)+timedelta(days=d)).month for d in range(365)])
    stats=np.array([np.quantile(net[months==m],[.25,.5,.75]) for m in range(1,13)])
    fig,ax=plt.subplots(figsize=paper_fig(6.3,2.5));x=np.arange(1,13)
    ax.fill_between(x,stats[:,0],stats[:,2],color='#E9F3EF',label='四分位区间')
    ax.plot(x,stats[:,1],'-o',color=GREEN,ms=4,label='中位数');ax.axhline(0,color='gray',lw=.7)
    ax.set(xticks=x,xlabel='月份',ylabel='净负载 / MW');ax.legend(ncol=2);clean(ax);fig.tight_layout();save(fig,'fig_monthly_profiles',['data/inputs.npz'])
    z=np.load(ROOT/'figures/source_data/fig_training.npz');fig,ax=plt.subplots(figsize=paper_fig(6.3,2.8))
    for i,label in enumerate(LABELS):ax.plot(z['quantiles'],z['cost_wanyuan'][i],marker=['o','s','^','D'][i],label=label,ms=4)
    ax.axvline(.65,color='gray',ls=':',lw=1);ax.set(xlabel='安全分位数 q',ylabel='1月15—31日费用 / 万元',xticks=z['quantiles']);ax.legend(ncol=2);clean(ax);fig.tight_layout();save(fig,'fig_training',['results/summary.json'])
    z=np.load(ROOT/'figures/source_data/fig_recourse.npz');fig,ax=plt.subplots(figsize=paper_fig(6.3,2.9));x=np.arange(4)
    for dx,key,label,c in [(-.18,'original_wanyuan','原紧急费用',ORANGE),(.18,'hindsight_wanyuan','事后基准',BLUE)]:
        bars=ax.bar(x+dx,z[key],width=.34,label=label,color=c);ax.bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
    for i,pct in enumerate(z['reduction_percent']):ax.text(i,z['original_wanyuan'][i]+21,f'下降{pct:.2f}%',ha='center',fontsize=9)
    ax.set(xticks=x,xticklabels=LABELS,ylabel='紧急费用 / 万元',ylim=(0,250));ax.legend(ncol=2,loc='upper center');clean(ax);fig.tight_layout();save(fig,'fig_recourse',['results/review_benchmarks.json'])
    z=np.load(ROOT/'figures/source_data/fig_control_comparison.npz')['cost_changes'];fig,(ax,bx)=plt.subplots(1,2,figsize=paper_fig(6.3,4.8),gridspec_kw={'width_ratios':[3,1]})
    labels=[f'{lab}  {mode}' for lab in LABELS for mode in ['点预测','场景均值','CVaR']];y=np.arange(12);cols=np.repeat([BLUE,GREEN,ORANGE,PURPLE],3)
    bars=ax.barh(y,z[:,:,0].ravel(),color=cols,height=.65);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9);ax.set(yticks=y,yticklabels=labels,xlabel='总费用增量 / 万元',xlim=(0,z[:,:,0].max()*1.25));ax.invert_yaxis();clean(ax,'x')
    bx.axis('off');bx.set(ylim=(11.5,-.5),xlim=(0,1));bx.text(.5,-1.1,'最差17日均费变化\n元 / 日',ha='center',fontsize=9)
    for i,v in enumerate(z[:,:,1].ravel()):bx.text(.5,i,f'{v:+.2f}',ha='center',va='center',color=ORANGE if v>0 else GREEN)
    fig.subplots_adjust(left=.28,right=.98,bottom=.13,top=.87,wspace=.12);save(fig,'fig_control_comparison',['results/risk_mpc.json'])
    z=json.loads((ROOT/'results/sensitivity.json').read_text(encoding="utf-8"))['quantile'];fig,(ax,bx)=plt.subplots(1,2,figsize=paper_fig(6.3,2.6));q=list(map(float,z))
    ax.plot(q,[z[str(v)]['total_cost']/1e4 for v in q],'-o',color=BLUE);ax.set(xlabel='安全分位数 q',ylabel='总费用 / 万元');clean(ax)
    bx.plot(q,[z[str(v)]['e']/1e4 for v in q],'-o',color=ORANGE,label='紧急购电');bx.plot(q,[z[str(v)]['w']/1e4 for v in q],'-s',color=GREEN,label='未利用电量');bx.set(xlabel='安全分位数 q',ylabel='电量 / 万kWh');bx.legend();clean(bx);fig.tight_layout();save(fig,'fig_risk_tradeoff',['results/sensitivity.json'])

def case_figure():
    a=read_inputs(ROOT/'data/inputs.npz');fc=forecasts(a);o=dict(np.load(ROOT/'results/q3.npz'));day=265;g=o['g'][day];n=(a['load'][day]-a['pv'][day])/6;s0=o['s'][day,0]
    plans=[g.copy()];preds=[fc[True][0][day,0].copy()]
    for issue in [1,2,3]:
        k=issue*36;adj=plan(fc[True][0][day,issue,k:]+safety(fc[True][1],day,issue,.65),a['base'][k:,0],o['s'][day,k],original=g[k:])['g']
        plans.append(np.r_[np.full(k,np.nan),adj]);preds.append(fc[True][0][day,issue].copy())
    s=s0;counter={key:[] for key in ['s','e','c','d']};counter['s']=[s]
    for t in range(144):
        c,d,e,w,s=step(g[t],n[t],s)
        for k,v in [('s',s),('e',e),('c',c),('d',d)]:counter[k].append(v)
    counter={k:np.array(v) for k,v in counter.items()};p=a['base'][:,0]
    original_cost=p*g+5*p*counter['e'];actual_cost=p*g+5*p*o['e'][day]+1.5*p*o['up'][day]+.5*p*o['down'][day]
    x=np.arange(144)/6;colors=[BLUE,GREEN,ORANGE,PURPLE]
    fig,axs=plt.subplots(5,1,figsize=paper_fig(6.3,5.8),sharex=True)
    for j in range(4):
        axs[0].step(x,plans[j]*6/1000,where='post',color=colors[j],label=f'{j*6}点计划',lw=1)
        axs[1].plot(x,preds[j]*6/1000,color=colors[j],lw=.85)
    axs[0].set_ylabel('合同 / MW');axs[0].legend(ncol=4,fontsize=8.5,loc='upper left')
    axs[1].plot(x,n*6/1000,color=INK,lw=1.4,label='实际净需求');axs[1].set_ylabel('净需求 / MW');axs[1].legend(loc='upper right',fontsize=8.5)
    axs[2].plot(np.arange(145)/6,o['s'][day]/1000,color=BLUE,label='日内更新');axs[2].plot(np.arange(145)/6,counter['s']/1000,color=ORANGE,ls='--',label='同初态零点合同');axs[2].set_ylabel('储量 / MWh');axs[2].legend(ncol=2,fontsize=8.5,loc='upper left')
    axs[3].bar(x,counter['e'],width=1/6,color=ORANGE,label='零点合同');axs[3].bar(x,o['e'][day],width=.08,color=BLUE,label='日内更新');axs[3].set_ylabel('紧急电量 / kWh');axs[3].legend(ncol=2,fontsize=8.5,loc='upper left')
    axs[4].plot(x,np.cumsum(original_cost)/1e4,color=ORANGE,ls='--');axs[4].plot(x,np.cumsum(actual_cost)/1e4,color=BLUE);axs[4].set(ylabel='累计费用 / 万元',xlabel='9月23日时刻 / h')
    for ax in axs:
        for k in [6,12,18]:ax.axvline(k,color='gray',ls=':',lw=.7)
        ax.set_xlim(0,24);ax.set_xticks(np.arange(0,25,2));clean(ax)
    fig.tight_layout(h_pad=.6);save(fig,'fig_case_decision',['results/q3.npz','data/inputs.npz'])
    data={'date':'2025-09-23','same_start_soc':float(s0),'zero_contract_cost':float(original_cost.sum()),'updated_cost':float(actual_cost.sum()),'saving':float(original_cost.sum()-actual_cost.sum()),'zero_end_soc':float(counter['s'][-1]),'updated_end_soc':float(o['s'][day,-1]),'zero_emergency_kwh':float(counter['e'].sum()),'updated_emergency_kwh':float(o['e'][day].sum()),'up_by_issue_block':[float(o['up'][day,j*36:(j+1)*36].sum()) for j in range(4)]}
    (OUT/'case.json').write_text(json.dumps(data,ensure_ascii=False,indent=2), encoding="utf-8");np.savez_compressed(OUT/'case.npz',plans=np.array(plans),predictions=np.array(preds),counter_soc=counter['s'],counter_emergency=counter['e'],updated_cost=actual_cost,zero_cost=original_cost)
    # 每次更新只执行到下一次发布；与正式结果的合同逐段一致。
    executed=np.concatenate([plans[j][j*36:(j+1)*36] for j in range(4)])
    assert np.max(np.abs(executed-o['a'][day]))<1e-5

if __name__=='__main__':
    setup();make_tables();system_figure();dispatch_figure();simple_figures();case_figure()
