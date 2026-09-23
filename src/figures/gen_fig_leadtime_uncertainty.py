"""原图11。分布、均值区间、覆盖区间分成三面板，避免尾部拉伸使均值区间不可读。"""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from _utils.paper_figure_style import *
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
# numpy 2.0 将 trapz 改名为 trapezoid，这里兼容两种版本。
_trapz=getattr(np,'trapezoid',None) or np.trapz
setup()
r=json.loads((ROOT/'results/forecast_uncertainty.json').read_text());rows=r['common_targets']['14']['lead_results'];rr=r['rank_corrected']['common_targets']['14']
z=np.load(ROOT/'results/forecast_uncertainty.npz');arr=z['common_14'];leads=z['leads_14'];errors=abs(arr[:,:,0]-arr[:,:,1])
mae=np.array([v['mae_kw'] for v in rows]);ci=np.array([v['mae_kw_ci95'] for v in rows]);np.testing.assert_allclose(mae,errors.mean(axis=0),atol=1e-9,rtol=0)
fig,axs=plt.subplots(1,3,figsize=paper_fig(5.8,3.6),gridspec_kw={'width_ratios':[1.15,1,1.05]})
fig.subplots_adjust(left=.12,right=.99,bottom=.23,top=.74,wspace=.64)
x=np.arange(4);rng=np.random.default_rng(20260912)
for i in x:
 a=errors[:,i];g=np.linspace(0,a.max(),420);k=gaussian_kde(a,bw_method=100/a.std(ddof=1));den=k(g)+k(-g);den/=_trapz(den,g)
 axs[0].fill_betweenx(g,i-.08-.36*den/den.max(),i-.08,color=BLUE_L,alpha=.8,lw=0)
 axs[0].scatter(i+.08+rng.uniform(-.06,.06,len(a)),a,s=2.3,c=BLUE,alpha=.26,lw=0)
axs[0].yaxis.labelpad=2
axs[0].set(ylim=(0,4100),yticks=[0,2000,4000],ylabel='绝对误差 / kW')
axs[1].errorbar(x,mae,yerr=[mae-ci[:,0],ci[:,1]-mae],fmt='D',color=BLUE,ms=4.5,capsize=3,lw=1.1)
for i in x:axs[1].text(i,ci[i,1]+38,f'{mae[i]:.1f}',ha='center',va='bottom',fontsize=9)
axs[1].set(ylim=(0,1100),yticks=[0,500,1000],ylabel='MAE / kW')
cv=[];cis=[]
for group,c,m,off in [(rows,PURPLE,'o',-.15),(rr,ORANGE,'s',.15)]:
 v=np.array([a['coverage'] for a in group])*100;bounds=np.array([a['coverage_ci95'] for a in group])*100
 axs[2].errorbar(x+off,v,yerr=[v-bounds[:,0],bounds[:,1]-v],fmt=m,color=c,ms=4,capsize=2.5,lw=1.1)
 cv.append(v);cis.append(bounds)
axs[2].axhline(80,color=GRAY,ls=':',lw=1)
axs[2].set(ylim=(65,94),yticks=[70,80,90],ylabel='覆盖率 / %')
for ax,title in zip(axs,['(a)','(b)','(c)']):
 ax.set(xlim=(-.65,3.65),xticks=x,xticklabels=leads,xlabel='提前时间 / h');ax.set_title(title,loc='left',pad=10);clean(ax)
fig.legend(handles=[Line2D([],[],color=PURPLE,marker='o',ls='none',label='经验分位数'),Line2D([],[],color=ORANGE,marker='s',ls='none',label='保守秩修正')],
 loc='upper center',bbox_to_anchor=(.81,.995),ncol=1,frameon=False,handletextpad=.4,labelspacing=.5)
save_bundle(fig,'fig_leadtime_uncertainty',{'leads_h':leads,'absolute_errors_kw':errors,'mae_kw':mae,'mae_ci95_kw':ci,'coverage_percent':cv,'coverage_ci95_percent':cis},
 {'sources':['results/forecast_uncertainty.json','results/forecast_uncertainty.npz'],'purpose':'将逐日分布与均值不确定性分开读取','sample_count_each':334,'kde_bandwidth_kw':100,'density_normalization':'each group equal peak width','units':['kW','%']})
plt.close(fig)
