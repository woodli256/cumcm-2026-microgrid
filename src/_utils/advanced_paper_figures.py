# -*- coding: utf-8 -*-
"""基于真实数据的九张多视图论文图。正式交付PDF，预览与可编辑源另存审阅目录。"""
from pathlib import Path
import sys,json,hashlib,datetime as dt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.patches import Rectangle,ConnectionPatch,FancyBboxPatch,PathPatch
from matplotlib.path import Path as MPath
from matplotlib.colors import TwoSlopeNorm,Normalize,LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.cluster.hierarchy import linkage,dendrogram
# numpy 2.0 将 trapz 改名为 trapezoid，这里兼容两种版本。
_trapz=getattr(np,'trapezoid',None) or np.trapz
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
 sys.path.insert(0,str(ROOT))
from _utils import plot_style as PS
AUDIT=ROOT/'audit/复杂图进阶'
BLUE=PS.BLUE;ORANGE=PS.ORANGE;GREEN=PS.GREEN;PURPLE=PS.PURPLE;GRAY=PS.GRAY;INK=PS.INK
COLORS=[PS.BLUE_D,PS.ORANGE,PS.GREEN,PS.PURPLE]
KEYS=['q2','q3','q4_2','q4_3'];CASE=['固定价不调整','固定价日内调整','波动价不调整','波动价日内调整']
CMAP=PS.CMAP_DIVERGING
def paper_fig(w,h):
 """统一图宽（167.6 mm），高度按原比例换算，保持正文版式不变。"""
 return (PS.FIG_WIDTH_IN,round(h*PS.FIG_WIDTH_IN/float(w),3))
def cjk_font():
 return PS.cjk_font_path()
def setup():
 return PS.apply(9.2)
def read(name):return json.loads((ROOT/'results'/f'{name}.json').read_text(encoding="utf-8"))
def inputs():return np.load(ROOT/'data/inputs.npz')
def clean(ax,axis='y'):
 return PS.clean(ax,grid=axis)
def title(ax,s):
 return PS.panel(ax,s)
def box(ax,x,y,w,h,text,color=BLUE):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.02',facecolor=color+'12',edgecolor=color,lw=.8,transform=ax.transAxes,clip_on=False)
 ax.add_patch(p);ax.text(x+w/2,y+h/2,text,ha='center',va='center',transform=ax.transAxes,linespacing=1.35);return p
def arrow(ax,start,end,color=GRAY):ax.annotate('',xy=end,xytext=start,xycoords='axes fraction',arrowprops=dict(arrowstyle='-|>',color=color,lw=.9))
def window_link(fig,main,zoom,x0,x1,y0,y1,color):
 main.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor='none',edgecolor=color,lw=1.1,ls='--'))
 for a,b in [((x0,y0),(0,1)),((x1,y0),(1,1))]:fig.add_artist(ConnectionPatch(a,b,coordsA='data',coordsB='axes fraction',axesA=main,axesB=zoom,color=color,lw=.6,alpha=.6,zorder=0))
def save(fig,stem,arrays,sources,structure,notes):
 meta={'data_status':'real','sources':sources,'structure':structure,'notes':notes,
  'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}}
 info=PS.save_bundle(fig,stem,arrays=arrays,meta=meta,preview_dir=AUDIT/'预览')
 print(json.dumps({'figure':stem,'axes':0,'data_shapes':info.get('shapes',{}),'outputs':info['outputs']},ensure_ascii=False))
 return meta

def dispatch():
 # 当前正文采用用户要求的左右同排结构，数据由专用脚本读取。
 from _utils.dispatch_pair import dispatch as draw_pair
 return draw_pair()

def monthly_profiles():
 setup();a=inputs();net=(a['load']-a['pv'])/1000;months=np.array([(dt.date(2025,1,1)+dt.timedelta(days=i)).month for i in range(365)])
 samples=[net[months==m].ravel() for m in range(1,13)];stats=np.array([[x.mean(),x.std(ddof=1),*np.quantile(x,[.25,.5,.75])] for x in samples])
 fig=plt.figure(figsize=paper_fig(6.6,5.5));gs=fig.add_gridspec(2,2,left=.11,right=.965,bottom=.12,top=.96,height_ratios=[1.1,1],hspace=.72,wspace=.6)
 ax=fig.add_subplot(gs[0,:]);grids=[np.linspace(x.min(),x.max(),400) for x in samples];dens=[gaussian_kde(x,bw_method=.3/x.std(ddof=1))(g) for x,g in zip(samples,grids)];dens=[d/_trapz(d,g) for d,g in zip(dens,grids)];peak=max(d.max() for d in dens)
 for i,(x,g,d) in enumerate(zip(samples,grids,dens),1):
  ax.fill_betweenx(g,i-.36*d/peak,i+.36*d/peak,color=BLUE,alpha=.45,lw=.4);ax.plot([i,i],np.quantile(x,[.25,.75]),color=INK,lw=2.6);ax.scatter(i,np.median(x),s=8,color='white',edgecolors=INK,lw=.4,zorder=4);ax.scatter(i+.12,x.mean(),s=13,marker='D',color=ORANGE)
 ax.axhline(0,color=GRAY,ls=':',lw=.7);ax.set(xlim=(.4,12.6),xticks=range(1,13),ylim=(-7,7),yticks=[-6,-3,0,3,6],ylabel='净负载 / MW',xlabel='');clean(ax);title(ax,'(a)')
 # 统计摘要沿月序对齐，保留全部均值和标准差。
 for i in range(12):ax.text(i+1,-9.5,f'{stats[i,0]:.2f}\n{stats[i,1]:.2f}',ha='center',va='top',clip_on=False,linespacing=1.25)
 ax.text(.35,-9.5,'均值\n标准差',ha='right',va='top',clip_on=False,linespacing=1.25)
 h=fig.add_subplot(gs[1,0]);hb=h.hexbin(a['load'].ravel()/1000,a['pv'].ravel()/1000,gridsize=32,mincnt=1,bins='log',cmap='Blues',linewidths=0)
 h.plot([0,11],[0,11],color=ORANGE,ls='--',lw=1);h.set(xlim=(1.5,8.5),ylim=(0,10.8),xlabel='负载 / MW',ylabel='光伏 / MW');title(h,'(b)')
 cb=fig.colorbar(hb,ax=h,pad=.025,fraction=.035);cb.set_label('观测数（对数色阶）');cb.set_ticks([1,10,100,1000]);cb.set_ticklabels(['1','10','100','1000'])
 r=fig.add_subplot(gs[1,1]);values=np.column_stack([a['load'].ravel(),a['pv'].ravel(),net.ravel()*1000]);corr=np.corrcoef(values,rowvar=False)
 r.imshow(corr,cmap=CMAP,vmin=-1,vmax=1);labels=['负载','光伏','净负载'];r.set(xticks=range(3),xticklabels=labels,yticks=range(3),yticklabels=labels)
 for i in range(3):
  for j in range(3):r.text(j,i,f'{corr[i,j]:+.2f}',ha='center',va='center',color='white' if abs(corr[i,j])>.8 else INK)
 title(r,'(c)');pos=r.get_position();cax=fig.add_axes([pos.x0,.037,pos.width,.01]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-1,1),cmap=CMAP),cax=cax,orientation='horizontal',ticks=[-1,0,1]);cb.ax.tick_params(pad=1,length=2)
 save(fig,'fig_monthly_profiles',{'net_mw':net,'months':months,'monthly_statistics':stats,'load_pv_net_kw':values,'pearson':corr},['data/inputs.npz'],'月度密度与摘要、供需密度散点、同变量相关矩阵','0.30MW固定KDE带宽，共用密度尺度。相关只描述观测关系，净负载为定义差值，不作因果推断；点不抽样。')

def netload_calendar():
 setup();a=inputs();net=(a['load']-a['pv'])/1000;hourly=net.reshape(365,24,6).mean(axis=2);hours=np.arange(24)+.5;days=np.arange(1,366)
 scores=np.convolve(hourly.mean(axis=1),np.ones(7)/7,mode='valid');starts=np.array([scores.argmax(),scores.argmin()]);windows=np.c_[starts+1,starts+7]
 fig=plt.figure(figsize=paper_fig(6.6,4.8));gs=fig.add_gridspec(2,2,left=.105,right=.86,bottom=.13,top=.96,wspace=.58,hspace=.65,width_ratios=[1.18,1])
 ax=fig.add_subplot(gs[:,0]);levels=np.arange(-7,7.01,.5);norm=TwoSlopeNorm(0,-7,7)
 cf=ax.contourf(hours,days,hourly,levels=levels,cmap=CMAP,norm=norm);ax.contour(hours,days,hourly,levels=[0],colors=INK,linewidths=.45)
 monthdays=[(dt.date(2025,m,15)-dt.date(2025,1,1)).days+1 for m in range(1,13)]
 ax.set(xlim=(.5,23.5),ylim=(365,1),xticks=[.5,8.5,16.5,23.5],xticklabels=['0.5','8.5','16.5','23.5'],yticks=monthdays,yticklabels=[f'{i}月' for i in range(1,13)],xlabel='小时中心 / h',ylabel='2025年日期');title(ax,'(a)')
 for j,start in enumerate(starts):
  z=fig.add_subplot(gs[j,1]);ds=days[start:start+7];v=hourly[start:start+7];z.contourf(hours,ds,v,levels=levels,cmap=CMAP,norm=norm);z.contour(hours,ds,v,levels=[0],colors=INK,linewidths=.8)
  dates=[(dt.date(2025,1,1)+dt.timedelta(days=int(d-1))).strftime('%m/%d') for d in ds]
  z.set(xlim=(6.5,18.5),ylim=(ds[-1],ds[0]),xticks=[6.5,10.5,14.5,18.5],yticks=ds[::2],yticklabels=dates[::2],xlabel='小时中心 / h');title(z,f'({chr(98+j)})')
  for tick in z.get_yticklabels():tick.set_bbox(dict(facecolor='white',edgecolor='none',pad=.2))
  ax.add_patch(Rectangle((6.5,ds[0]),12,6,fc='none',ec=COLORS[j],lw=1.4));ax.text(1.0,ds.mean(),chr(98+j),color=COLORS[j],fontweight='bold')
  fig.add_artist(ConnectionPatch((18.5,ds.mean()),(0,.5),coordsA='data',coordsB='axes fraction',axesA=ax,axesB=z,color=COLORS[j],lw=.8))
 cax=fig.add_axes([.89,.20,.016,.64]);cb=fig.colorbar(cf,cax=cax,ticks=[-6,-3,0,3,6]);cb.set_label('净负载 / MW')
 save(fig,'fig_netload_calendar',{'raw_net_mw':net,'hourly_net_mw':hourly,'hours':hours,'days':days,'window_days':windows,'seven_day_mean_mw':scores},['data/inputs.npz'],'全年等高线与两个有真实日期标记的7日局部窗','每小时6点算术平均；局部窗按7日均值最小和最大确定，共用坐标单位与色阶；只作网格视觉插值。')

def training():
 setup();r=read('summary')['training'];qs=np.array(r['q2']['quantiles']);cost=np.array([r[k]['costs_jan15_31'] for k in KEYS])/1e4;delta=cost-cost.min(axis=1)[:,None]
 fig=plt.figure(figsize=paper_fig(6.6,5.0));gs=fig.add_gridspec(2,2,left=.12,right=.96,bottom=.11,top=.93,height_ratios=[1.15,1],wspace=.42,hspace=.6)
 main=fig.add_subplot(gs[0,0]);zoom=fig.add_subplot(gs[0,1]);table=fig.add_subplot(gs[1,:]);sel=np.argmin(cost,axis=1)
 for i,col in enumerate(COLORS):
  for ax in [main,zoom]:ax.plot(qs,delta[i],color=col,marker=['o','s','^','D'][i],ms=4,lw=1.2,label=['问题二','问题三','问题四-2','问题四-3'][i])
 main.set(xlim=(.48,.98),ylim=(-.5,17.5),xticks=[.5,.65,.8,.95],xlabel='安全分位数 q',ylabel='相对本组最低费 / 万元');zoom.set(xlim=(.48,.82),ylim=(-.25,5.6),xticks=[.5,.65,.8],xlabel='安全分位数 q',ylabel='费用增量 / 万元')
 for ax in [main,zoom]:
  clean(ax);ax.axvline(.65,color=GRAY,ls=':',lw=.8);ax.axvline(.8,color=ORANGE,ls=':',lw=.8)
 main.text(.655,16.2,'0.65',color=GRAY,ha='center',fontsize=8);main.text(.805,16.2,'0.8 解析值',color=ORANGE,ha='center',fontsize=8)
 title(main,'(a)');title(zoom,'(b)');main.add_patch(Rectangle((.5,0),.3,5.3,fill=False,ec=GRAY,lw=1,ls='--'));main.text(.78,5.9,'b',ha='right',color=GRAY)
 table.imshow(delta,cmap='Blues',vmin=0,vmax=17,aspect='auto');table.set(xticks=range(5),xticklabels=[f'{q:.2f}' for q in qs],yticks=range(4),yticklabels=['问题二','问题三','问题四-2','问题四-3'],xlabel='安全分位数 q')
 for i in range(4):
  for j in range(5):table.text(j,i,f'{cost[i,j]:.2f}',ha='center',va='center',color='white' if delta[i,j]>10 else INK)
  table.add_patch(Rectangle((sel[i]-.48,i-.45),.96,.9,fill=False,edgecolor=ORANGE,lw=1.5))
 title(table,'(c)')
 fig.legend(handles=[Line2D([],[],color=c,marker=m,label=l) for c,m,l in zip(COLORS,['o','s','^','D'],['问题二','问题三','问题四-2','问题四-3'])],ncol=4,loc='upper center',frameon=False,columnspacing=1.1)
 save(fig,'fig_training',{'quantiles':qs,'cost_wanyuan':cost,'delta_wanyuan':delta,'selected_indices':sel},['results/summary.json'],'全参数响应、同坐标最低点放大、20值训练矩阵','全图与局部图均使用原5个非等距候选参数，连线不表示新增求解。训练窗口四组均取低点0.65，正文按补救手段对无调整问题改用解析值0.8。')

def risk_tradeoff():
 setup();s=read('sensitivity')['quantile'];q=np.array(sorted(map(float,s)));cost=np.array([s[str(x)]['total_cost'] for x in q])/1e4;e=np.array([s[str(x)]['e'] for x in q])/1e4;w=np.array([s[str(x)]['w'] for x in q])/1e4;chosen=read('summary')['training']['q2']['main_q']
 fig=plt.figure(figsize=paper_fig(6.6,4.8));gs=fig.add_gridspec(2,2,left=.12,right=.93,bottom=.12,top=.97,width_ratios=[1.25,1],wspace=.53,hspace=.78)
 ax=fig.add_subplot(gs[:,0]);zoom=fig.add_subplot(gs[0,1]);fees=fig.add_subplot(gs[1,1]);norm=Normalize(cost.min(),cost.max());cm=plt.get_cmap('viridis')
 for i in range(5):
  mark='D' if np.isclose(q[i],chosen) else 'o';col=[GRAY,BLUE,GREEN,ORANGE,PURPLE][i]
  for a in [ax,zoom]:a.scatter(e[i],w[i],s=52,marker=mark,c=[col],edgecolors=INK,lw=.5,zorder=3)
  off=[(-5,-18),(6,7),(8,8),(8,0),(7,10)][i];ax.annotate(f'{q[i]:.2f}',(e[i],w[i]),xytext=off,textcoords='offset points',ha='right' if i==0 else 'left')
  if i>=2:zoom.annotate(f'q={q[i]:.2f}\n{e[i]:.1f}, {w[i]:.1f}',(e[i],w[i]),xytext=(-7,9) if i==2 else (7,4),textcoords='offset points',ha='right' if i==2 else 'left',va='bottom')
  fees.scatter(q[i],cost[i],s=42,marker=mark,c=[col],edgecolors=INK,lw=.5);fees.annotate(f'{cost[i]:.1f}',(q[i],cost[i]),xytext=[(0,7),(0,7),(-8,5),(-6,19),(0,35)][i],textcoords='offset points',ha='center',arrowprops=dict(arrowstyle='-',lw=.4,color=GRAY) if i>=3 else None)
 ax.set(xlim=(-5,90),ylim=(30,520),xlabel='紧急购电 / 万kWh',ylabel='未利用电量 / 万kWh',xticks=[0,30,60,90]);zoom.set(xlim=(-.5,16),ylim=(255,465),xlabel='紧急购电 / 万kWh',ylabel='未利用 / 万kWh',xticks=[0,8,16]);fees.set(xlim=(.46,1),ylim=(1340,1680),xlabel='安全分位数 q',ylabel='总费用 / 万元',xticks=[.5,.65,.8,.95])
 title(ax,'(a)');title(zoom,'(b)');title(fees,'(c)')
 ax.add_patch(Rectangle((0,260),15,195,fill=False,ec=GRAY,lw=1));fig.add_artist(ConnectionPatch((15,455),(0,1),coordsA='data',coordsB='axes fraction',axesA=ax,axesB=zoom,color=GRAY,lw=.8))
 for a in [ax,zoom,fees]:clean(a)
 save(fig,'fig_risk_tradeoff',{'quantiles':q,'total_cost_wanyuan':cost,'emergency_10k_kwh':e,'unused_10k_kwh':w,'selected_q':np.array(chosen)},['results/sensitivity.json','results/summary.json'],'两类电量联合散点、精确局部放大、同参数费用视图','所有15个指标均来自既有敏感性结果；主结果参数由补救手段的边际费用决定，不由测试期反向选出。局部图不新增样本。')

def update_waterfall():
 setup();s=read('sensitivity');m=read('summary');comp=np.array([s['forecast_updates']['only0']['cost_components'],s['no_newpv']['cost_components'],m['strategies']['q3']['cost_components']])/1e4;tot=comp.sum(axis=1);savings=tot[:-1]-tot[1:]
 fig=plt.figure(figsize=paper_fig(6.6,3.0));gs=fig.add_gridspec(1,2,left=.11,right=.96,bottom=.27,top=.90,width_ratios=[1.35,1],wspace=.45)
 bars=fig.add_subplot(gs[0,0]);colors=[BLUE,ORANGE,GREEN,PURPLE];names=['计划购电','紧急购电','上调费用','下调费用'];base=np.zeros(3)
 for j in range(4):
  bars.bar(np.arange(3),comp[:,j],bottom=base,color=colors[j],width=.6,edgecolor='white',lw=.5,label=names[j]);base+=comp[:,j]
 for i in range(3):bars.text(i,tot[i]+22,f'{tot[i]:.2f}',ha='center')
 bars.set(xticks=range(3),xticklabels=['A','B','C'],ylim=(0,1600),yticks=[0,500,1000,1500],ylabel='总费用 / 万元');clean(bars);title(bars,'(a)')
 wf=fig.add_subplot(gs[0,1]);x=np.arange(3);vals=[savings[0],savings[1],savings.sum()];bottom=[0,savings[0],0]
 wf.bar(x,vals,bottom=bottom,width=.58,color=[GREEN,ORANGE,BLUE],edgecolor='white')
 for i in range(3):wf.text(i,bottom[i]+vals[i]+3,f'{vals[i]:.2f}',ha='center')
 wf.plot([.3,.7],[savings[0]]*2,color=GRAY,lw=.8);wf.plot([1.3,1.7],[savings.sum()]*2,color=GRAY,lw=.8);wf.set(xticks=x,xticklabels=['A 至 B','B 至 C','A 至 C'],ylim=(0,115),yticks=[0,25,50,75,100],ylabel='沿比较路径节省 / 万元');clean(wf);title(wf,'(b)')
 fig.legend(handles=[Rectangle((0,0),1,1,fc=colors[j],label=names[j]) for j in range(3)],loc='lower center',bbox_to_anchor=(.5,.012),ncol=3,frameon=False)
 save(fig,'fig_update_waterfall',{'components_wanyuan':comp,'totals_wanyuan':tot,'step_savings_wanyuan':savings},['results/sensitivity.json','results/summary.json'],'三种策略费用构成与同一路径差额瀑布','A/B/C为独立对照而非执行先后；差额为既定比较路径上的嵌套策略差，不解释成独立因果效应。零下调费用不着色。')

def forecast_clock():
 setup();a=inputs();f=np.load(ROOT/'results/forecasts.npz');actual=(a['load']-a['pv'])[31:];err=np.stack([actual-f[k][31:,0,:]*6 for k in ['p2','p3']]);hourly=err.reshape(2,334,24,6).transpose(0,2,1,3).reshape(2,24,2004);bias=hourly.mean(axis=2);mae=np.abs(hourly).mean(axis=2);q10,q90=np.quantile(hourly,[.1,.9],axis=2);h=np.arange(24)+.5;noon=hourly[:,12];diff=np.abs(noon[1])-np.abs(noon[0])
 fig=plt.figure(figsize=paper_fig(6.6,5.1));gs=fig.add_gridspec(2,2,left=.12,right=.90,bottom=.15,top=.93,wspace=.76,hspace=.60)
 b=fig.add_subplot(gs[0,0]);m=fig.add_subplot(gs[1,0]);joint=fig.add_subplot(gs[0,1]);hist=fig.add_subplot(gs[1,1])
 for i,(col,label) in enumerate([(BLUE,'历史光伏均值'),(ORANGE,'零点光伏预报')]):
  b.fill_between(h,q10[i],q90[i],color=col,alpha=.13);b.plot(h,bias[i],color=col,lw=1.4,label=label);m.plot(h,mae[i],color=col,lw=1.4)
 for ax in [b,m]:ax.axvspan(12,13,fc=GREEN,alpha=.13);ax.set(xlim=(0,24),xticks=[0,6,12,18,24],xlabel='目标时刻 / h');clean(ax)
 b.axhline(0,color=GRAY,lw=.7);b.set(ylabel='预测偏差 / kW');m.set(ylabel='MAE / kW',ylim=(0,mae.max()*1.18));title(b,'(a)');title(m,'(b)')
 hb=joint.hexbin(noon[0],noon[1],gridsize=25,mincnt=1,cmap='Blues',linewidths=0);lim=np.ceil(np.max(np.abs(noon))/500)*500;joint.plot([-lim,lim],[-lim,lim],color=GRAY,ls='--',lw=.8);joint.set(xlim=(-lim,lim),ylim=(-lim,lim),xticks=[-lim,0,lim],yticks=[-lim,0,lim],xlabel='历史均值误差 / kW',ylabel='新预报误差 / kW');title(joint,'(c)')
 cb=fig.colorbar(hb,ax=joint,fraction=.035,pad=.03);cb.set_label('观测数');joint.set_aspect('equal',adjustable='box')
 edges=np.linspace(diff.min(),diff.max(),31);counts,_=np.histogram(diff,edges);hist.stairs(counts,edges,fill=True,color=GREEN,alpha=.65);hist.axvline(0,color=INK,ls=':',lw=1);hist.axvline(np.median(diff),color=ORANGE,lw=1.3);hist.set(xlabel='绝对误差变化 / kW',ylabel='观测数');clean(hist);title(hist,'(d)');hist.text(.97,.94,f'中位数 {np.median(diff):.1f} kW',ha='right',va='top',transform=hist.transAxes)
 b.text(12.5,b.get_ylim()[1]*.80,'c',ha='center',color=GREEN)
 fig.legend(handles=[Line2D([],[],color=BLUE,label='历史光伏均值'),Line2D([],[],color=ORANGE,label='零点光伏预报')],loc='upper center',ncol=2,frameon=False)
 save(fig,'fig_forecast_clock',{'errors_kw':err,'hour_centers_h':h,'bias_kw':bias,'mae_kw':mae,'q10_kw':q10,'q90_kw':q90,'noon_paired_errors_kw':noon,'noon_absolute_error_change_kw':diff,'hist_edges_kw':edges,'hist_counts':counts},['data/inputs.npz','results/forecasts.npz'],'全天误差剖面、指定小时同日同刻配对、对应绝对误差变化分布','每小时334天乘6区间共2004点；12时窗口事先按正午选取，未按最优结果筛选；原始分位范围不是置信区间。')

def recourse():
 setup();b=read('review_benchmarks')['hindsight'];old=np.array([b[k]['original_emergency_cost'] for k in KEYS])/1e4;new=np.array([b[k]['hindsight_emergency_cost'] for k in KEYS])/1e4;gap=np.array([b[k]['gap'] for k in KEYS])/1e4;share=new/old;np.testing.assert_allclose(new+gap,old,atol=1e-8)
 fig=plt.figure(figsize=paper_fig(6.6,3.9));gs=fig.add_gridspec(2,2,left=.07,right=.98,bottom=.16,top=.92,hspace=.45,wspace=.20)
 def ribbon(ax,y0,y1,t0,t1,color):
  verts=[(.20,y0),(.44,y0),(.46,t0),(.68,t0),(.68,t1),(.46,t1),(.44,y1),(.20,y1),(.20,y0)]
  codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY];ax.add_patch(PathPatch(MPath(verts,codes),fc=color,ec='white',lw=.6,alpha=.83))
 for i in range(4):
  ax=fig.add_subplot(gs[i//2,i%2]);ax.set(xlim=(0,1.03),ylim=(-.13,1.17));ax.axis('off');title(ax,f'({chr(97+i)})')
  s=share[i];ribbon(ax,0,s,0,s,GRAY);ribbon(ax,s,1,s+.13,1.13,ORANGE)
  ax.text(.17,.50,f'{old[i]:.2f}',ha='right',va='center');ax.text(.71,s/2,f'{new[i]:.2f}',ha='left',va='center');ax.text(.71,(s+1)/2+.13,f'{gap[i]:.2f}\n{(1-s)*100:.1f}%',ha='left',va='center',linespacing=1.2)
  ax.text(.18,-.12,'原费用 / 万元',ha='right');ax.text(.70,-.12,'基准与差额 / 万元',ha='left');ax.plot([.19,.19],[0,1],color=INK,lw=.6)
 fig.legend(handles=[Rectangle((0,0),1,1,fc=GRAY,label='事后基准费用'),Rectangle((0,0),1,1,fc=ORANGE,label='与原规则的费用差额')],loc='lower center',bbox_to_anchor=(.5,.035),ncol=2,frameon=False)
 save(fig,'fig_recourse',{'original_wanyuan':old,'hindsight_wanyuan':new,'gap_wanyuan':gap,'reduction_percent':(1-share)*100},['results/review_benchmarks.json'],'四情景同百分比尺度费用分流','带宽对应各自原紧急费用的比例，不代表能量流或相同绝对金额；保留12项费用与4项差额比例。')

def control_comparison():
 setup();r=read('risk_mpc')['test'];modes=['nominal','scenario_mean','scenario_cvar'];changes=np.array([[(r[k]['variants'][m]['metrics']['cost']-r[k]['baseline']['cost'])/1e4,r[k]['variants'][m]['metrics']['worst_5_percent_daily_mean']-r[k]['baseline']['worst_5_percent_daily_mean']] for k in KEYS for m in modes]);standard=(changes-changes.mean(axis=0))/changes.std(axis=0,ddof=1);tree=linkage(standard,method='ward');order=dendrogram(tree,no_plot=True)['leaves']
 fig=plt.figure(figsize=paper_fig(6.6,5.4));treeax=fig.add_axes([.045,.28,.075,.55]);heat=fig.add_axes([.335,.28,.26,.55]);sc=fig.add_axes([.72,.34,.26,.43]);den=dendrogram(tree,orientation='left',no_labels=True,ax=treeax,color_threshold=0,above_threshold_color=GRAY);treeax.invert_yaxis();treeax.axis('off')
 assert den['leaves']==order
 norms=[Normalize(0,200),TwoSlopeNorm(0,-7000,7000)];cmaps=[plt.get_cmap('YlOrBr'),CMAP]
 labels=[];short=['固定不调','固定调整','波动不调','波动调整'];ms=['点预测','情景均值','CVaR']
 for row,idx in enumerate(order):
  labels.append(f'{idx+1:02d} {short[idx//3]}·{ms[idx%3]}')
  for j in range(2):
   v=changes[idx,j];rgba=cmaps[j](norms[j](v));heat.add_patch(Rectangle((j-.5,row-.5),1,1,fc=rgba,ec='white',lw=1));lum=np.dot(rgba[:3],[.2126,.7152,.0722]);heat.text(j,row,f'{v:+.1f}' if j==0 else f'{v:+.0f}',ha='center',va='center',color='white' if lum<.46 else INK)
 heat.set(xlim=(-.5,1.5),ylim=(11.5,-.5),yticks=range(12),yticklabels=labels,xticks=[0,1],xticklabels=['总费用','最差17日']);heat.tick_params(length=0,pad=5);[s.set_visible(False) for s in heat.spines.values()]
 label_positions=[(25,-6.5),(118,-4.0),(140,-2.6),(58,2.2),(122,5.0),(150,5.8),(52,-4.9),(192,-6.3),(194,-4.3),(88,4.0),(145,7.1),(190,7.5)]
 for idx,(x,y) in enumerate(changes):
  sc.scatter(x,y/1000,c=[COLORS[idx//3]],marker=['o','s','^'][idx%3],s=35,edgecolors='white',lw=.5,zorder=3);sc.annotate(f'{idx+1:02d}',(x,y/1000),xytext=label_positions[idx],textcoords='data',ha='center',va='center',color=COLORS[idx//3],arrowprops=dict(arrowstyle='-',color=COLORS[idx//3],lw=.5))
 sc.axhspan(-7,0,color=PURPLE,alpha=.06);sc.axhline(0,color=INK,lw=.8);sc.set(xlim=(-8,205),ylim=(-7,8),xticks=[0,100,200],yticks=[-6,0,6],xlabel='总费用差 / 万元',ylabel='最差17日日均费用差 / 千元');clean(sc)
 fig.text(.045,.89,'(a)',ha='left');fig.text(.72,.89,'(b)',ha='left')
 for j in range(2):
  cax=fig.add_axes([.08+j*.32,.16,.20,.018]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norms[j],cmap=cmaps[j]),cax=cax,orientation='horizontal',ticks=[0,100,200] if j==0 else [-7000,0,7000]);cb.set_label(['总费用差 / 万元','最差17日日均费用差 / 元'][j])
 fig.legend(handles=[Line2D([],[],color=c,marker='o',ls='',label=l) for c,l in zip(COLORS,short)],loc='lower center',bbox_to_anchor=(.50,.015),ncol=4,frameon=False,columnspacing=1)
 fig.legend(handles=[Line2D([],[],color=INK,marker=m,ls='',label=ms[j]) for j,m in enumerate(['o','s','^'])],loc='lower center',bbox_to_anchor=(.85,.155),ncol=1,frameon=False,handlelength=1,labelspacing=.35)
 save(fig,'fig_control_comparison',{'cost_changes':changes.reshape(4,3,2),'standardized_changes':standard,'linkage':tree,'row_order':np.array(order),'case_index':np.repeat(np.arange(4),3),'mode_index':np.tile(np.arange(3),4)},['results/risk_mpc.json'],'共享编号的树状排序、双指标数值矩阵与费用权衡散点','Ward聚类仅对两列标准化差值作描述性排序，无显著性推断；两色标单位和范围独立；全部12种控制和24项费用差保留。')

PLOTS={name:globals()[name] for name in ['dispatch','monthly_profiles','netload_calendar','training','risk_tradeoff','update_waterfall','forecast_clock','recourse','control_comparison']}
