"""当前问题三实际储量热力图。334天、每天144段，色值不平滑或插补。"""
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
    fig.canvas.draw()
    for text in fig.findobj(Text):
        if text.get_text():
            text.set_fontsize(max(10.5,text.get_fontsize()))
    sources=['results/q3.npz','data/inputs.npz']
    PS.save_bundle(fig,name,arrays={'day_index':np.arange(31,365),'interval_start_hour':np.arange(144)/6,'state_kwh':states,'soc_percent':soc},meta={
        'sources':sources,'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        'structure':'334天×144段的起点储量热力图','notes':'2025-02-01至12-31；实际储量除以12000 kWh；颜色范围10%至90%；不作平滑或插补',
        'paper_width_mm':140.8,'actual_min_font_pt':10.5,'actual_min_effective_font_pt':round(10.5*140.8/(fig.get_figwidth()*25.4),3)})
# Adapted recipe:competition.spatiotemporal_heatmap. No cell labels for dense 334x144 matrix.
from datetime import date
z=np.load(ROOT/'results/q3.npz');states=z['s'][31:,:144];soc=states/12000*100
assert soc.shape==(334,144) and soc.min()>=10-1e-8 and soc.max()<=90+1e-8
fig,ax=plt.subplots(figsize=PS.figsize(5.8,3.9))
cmap=LinearSegmentedColormap.from_list('macaron_soc',[pu._lighten(pu.PALETTE[4],.75),pu.PALETTE[2],pu.PALETTE[0],pu.PALETTE[3]],N=256)
im=ax.imshow(soc,aspect='auto',origin='upper',extent=[0,24,334,0],cmap=cmap,vmin=10,vmax=90,interpolation='nearest',rasterized=True)
months=[2,4,6,8,10,12];pos=[(date(2025,m,1)-date(2025,2,1)).days+.5 for m in months]
ax.set_yticks(pos,[f'{m}月1日' for m in months]);ax.set_xticks([0,4,8,12,16,20,24]);ax.set_xlabel('日内时刻 / h')
# 两侧纵坐标说明横放在轴的上方，为文字预留顶部空间。
ax.set_ylabel('2025年日期',rotation=0,ha='left',va='bottom')
ax.yaxis.set_label_coords(-.12,1.045)
cbar=fig.colorbar(im,ax=ax,pad=.028,fraction=.045,aspect=24,ticks=[10,30,50,70,90])
cbar.set_label('电芯荷电状态 / %',rotation=0,ha='right',va='bottom')
cbar.ax.yaxis.set_label_coords(2.7,1.045)
cbar.ax.tick_params(labelsize=9.2)
fig.subplots_adjust(left=.155,right=.87,bottom=.16,top=.89)
finish(fig,'fig_soc_heatmap')
report={'days':334,'intervals_per_day':144,'unit':'percent of nominal 12000 kWh','sample':'interval-start actual q3 states, Feb1-Dec31','min':float(soc.min()),'max':float(soc.max()),'lower_bound_fraction':float(np.mean(np.isclose(states,1200,atol=1e-6,rtol=0))),'upper_bound_fraction':float(np.mean(np.isclose(states,10800,atol=1e-6,rtol=0))),'bound_tolerance_kwh':1e-6}
(ROOT/'results/soc_heatmap_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
