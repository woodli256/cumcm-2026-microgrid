"""两种零点净需求预测的逐日MAE分布。正文宽140.8 mm，最小有效字号约8.65 pt（PDF实测）。"""
from pathlib import Path
import os, sys, json, hashlib
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
import matplotlib
matplotlib.use('Agg')
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS
import _utils.plot_utils as pu
setup_style()
from _utils import plot_style as PS
PS.apply(10.5)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def finish(fig,name):
    from matplotlib.text import Text
    for text in fig.findobj(Text):
        if text.get_text():
            text.set_fontsize(max(10.5,text.get_fontsize()))
    sources=['results/forecasts.npz','data/inputs.npz']
    PS.save_bundle(fig,name,arrays={'day_index':np.arange(31,365),'history_daily_mae_kw':vals[0],'issued_daily_mae_kw':vals[1]},meta={
        'sources':sources,'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        'structure':'净需求预测误差的分布与箱线','notes':'2025-02-01至12-31，各334个逐日MAE，每日144段；两组均为零点预测且不含安全余量',
        'paper_width_mm':140.8,'actual_min_font_pt':10.5,'actual_min_effective_font_pt':round(10.5*140.8/(fig.get_figwidth()*25.4),3)})
# Adapted recipe:basic.raincloud, horizontal to show two complete daily distributions.
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
f=np.load(ROOT/'results/forecasts.npz');a=np.load(ROOT/'data/inputs.npz')
actual=(a['load']-a['pv'])/6
vals=[np.abs(f[k][31:,0,:]-actual[31:]).mean(axis=1)*6 for k in ['p2','p3']]
assert all(len(v)==334 and len(np.unique(v))>20 and np.isfinite(v).all() for v in vals)
fig,ax=plt.subplots(figsize=PS.figsize(5.8,3.25));rng=np.random.default_rng(20260911)
for i,v in enumerate(vals):
    color=pu.PALETTE[i];grid=np.linspace(max(0,v.min()-35),v.max()+35,400)
    den=gaussian_kde(v,bw_method=.3)(grid);height=den/den.max()*.32
    for layer in range(8):
        lower=height*layer/8;upper=height*(layer+1)/8
        ax.fill_between(grid,i+lower,i+upper,color=color,alpha=.12+.07*layer,linewidth=0)
    ax.plot(grid,i+height,color=color,lw=1.2)
    ax.boxplot(v,positions=[i-.15],widths=.13,vert=False,patch_artist=True,showfliers=False,boxprops=dict(facecolor=pu._lighten(color,.6),edgecolor=color,linewidth=1.2),medianprops=dict(color=pu.COLORS['text'],linewidth=1.5),whiskerprops=dict(color=color,linewidth=1),capprops=dict(color=color,linewidth=1))
    ax.scatter(v,i-.35+rng.uniform(-.055,.055,len(v)),s=3,alpha=.32,color=color,edgecolor='none',rasterized=False)
    ax.scatter([v.mean()],[i-.15],s=32,marker='D',facecolor=color,edgecolor='white',linewidth=.9,zorder=6)
    ax.text(760,i-.15,f'{v.mean():.2f}',ha='left',va='center',fontsize=9.2,color=pu.COLORS['text'],bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=1))
ax.set_yticks([0,1],['历史光伏均值','零点光伏预报']);ax.set_xlabel('逐日净需求预测 MAE / kW')
ax.set_xlim(0,1020);ax.set_ylim(-.5,1.57);ax.set_xticks([0,200,400,600,800,1000]);ax.spines[['top','right']].set_visible(False)
ax.legend(handles=[Line2D([0],[0],marker='D',color='none',markerfacecolor=pu.COLORS['text'],markeredgecolor='white',markersize=6,label='均值')],loc='upper right',bbox_to_anchor=(1,1),frameon=False)
fig.subplots_adjust(left=.21,right=.97,bottom=.20,top=.97)
finish(fig,'fig_forecast_distribution')
report={'sample_count_each':334,'unit':'kW','source':'forecasts.npz contains interval-energy predictions in kWh; multiply interval-MAE by 6 for kW','statistic':'daily mean absolute error across 144 same zero-issue intervals; KDE descriptive, not CI','methods':{k:{'mean':float(v.mean()),'median':float(np.median(v)),'q25':float(np.quantile(v,.25)),'q75':float(np.quantile(v,.75)),'min':float(v.min()),'max':float(v.max())} for k,v in zip(['history_pv','issued_pv0'],vals)}}
(ROOT/'results/forecast_distribution.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));np.savez_compressed(ROOT/'results/forecast_daily_mae.npz',history_pv_kw=vals[0],issued_pv0_kw=vals[1]);print(json.dumps(report,ensure_ascii=False))
