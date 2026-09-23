"""Real evidence figure. Native width 5.8 in; paper width 136 mm.
Scale=0.923; tick 9.2 pt displays at 8.49 pt. No invented uncertainty.
"""
from pathlib import Path
import os, sys, json
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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def finish(fig,name):
    PS.save_bundle(fig,name,meta={'sources':['results/capacity_surface.json','data/inputs.npz'],'structure':'255点容量-功率网格与等高线','notes':'每格点重解日LP，星号为原设备'})
# Adapted from recipe:competition.contour; independently solve every grid point.
import importlib.util
spec=importlib.util.spec_from_file_location('model_depth',ROOT/'code/model_depth.py')
md=importlib.util.module_from_spec(spec);spec.loader.exec_module(md)
a=np.load(ROOT/'data/inputs.npz')['base'];net=(a[:,1]-a[:,2])/6;price=a[:,0]
capacities=np.arange(8000.,16001.,500.)
powers=np.arange(2000.,9001.,500.)
cost=np.empty((len(powers),len(capacities)))
checks={k:0. for k in ['primal_residual','dual_stationarity','dual_sign_violation','duality_gap','complementarity']}
for i,power in enumerate(powers):
    for j,capacity in enumerate(capacities):
        sol,cert,_,_=md.solve_day(net,price,capacity=capacity,power=power)
        cost[i,j]=sol.fun
        for k,v in cert.items():checks[k]=max(checks[k],float(v))
assert np.isfinite(cost).all() and max(checks.values())<1e-6
base=float(cost[np.where(powers==5000)[0][0],np.where(capacities==12000)[0][0]])
assert abs(base-float(np.load(ROOT/'results/q1.npz')['objective']))<1e-6
np.savez_compressed(ROOT/'results/capacity_surface.npz',capacities_kwh=capacities,powers_kw=powers,cost_yuan=cost)
report={'grid_shape':list(cost.shape),'solves':int(cost.size),'capacity_range_kwh':[float(capacities.min()),float(capacities.max())],'power_range_kw':[float(powers.min()),float(powers.max())],'base_cost_yuan':base,'cost_range_yuan':[float(cost.min()),float(cost.max())],'certificate_maxima':checks,'capacity_adjacent_max_cost_increase':float(np.diff(cost,axis=1).max()),'power_adjacent_max_cost_increase':float(np.diff(cost,axis=0).max()),'scope':'Deterministic single-day operating cost; all states begin/end at 6000 kWh; no investment objective. Each grid point solved, contour between grid points is visual interpolation.'}
(ROOT/'results/capacity_surface.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
X,Y=np.meshgrid(capacities/1000,powers/1000);Z=cost/10000
fig,ax=plt.subplots(figsize=PS.figsize(5.8,3.65))
cmap=LinearSegmentedColormap.from_list('macaron_cost',[pu._lighten(pu.PALETTE[2],.70),pu.PALETTE[0],pu.PALETTE[3]],N=256)
cf=ax.contourf(X,Y,Z,levels=np.linspace(Z.min(),Z.max(),151),cmap=cmap,antialiased=False)
# Rasterize only the dense fill; keep contour labels and axes as vectors.
try:
    cf.set_zorder(-1)
except AttributeError:
    for _c in getattr(cf,'collections',[]):
        _c.set_zorder(-1)
ax.set_rasterization_zorder(0)
levels=np.arange(np.ceil(Z.min()*10)/10,np.floor(Z.max()*10)/10+.001,.1)
cs=ax.contour(X,Y,Z,levels=levels[1:-1],colors='white',linewidths=.9,alpha=.95)
ax.clabel(cs,inline=True,fontsize=9.2,fmt='%.1f',manual=[(14.4,5.2),(12.5,4.5),(10.8,3.7),(9.5,3.2),(9.1,2.5)])
ax.scatter([12],[5],s=85,marker='*',color=pu.COLORS['text'],edgecolor='white',linewidth=1,zorder=6)
ax.annotate('基准设备',xy=(12,5),xytext=(13.35,6.4),fontsize=9.5,arrowprops=dict(arrowstyle='-',color=pu.COLORS['text'],lw=1),bbox=dict(boxstyle='round,pad=.23',facecolor='white',edgecolor=pu.COLORS['grid'],alpha=.95))
cbar=fig.colorbar(cf,ax=ax,pad=.025,fraction=.045,aspect=22)
cbar.solids.set_edgecolor('face')
cbar.set_ticks([3.3,3.4,3.5,3.6,3.7,3.8,3.9]);cbar.set_ticklabels(['3.3','3.4','3.5','3.6','3.7','3.8','3.9']);cbar.set_label('最优日电费 / 万元');cbar.ax.tick_params(labelsize=9.2)
ax.set_xlabel('名义容量 / MWh');ax.set_ylabel('充放电功率上限 / MW')
ax.set_xticks([8,10,12,14,16]);ax.set_yticks([2,3,4,5,6,7,8,9]);ax.set_xlim(8,16);ax.set_ylim(2,9)
fig.subplots_adjust(left=.13,right=.87,bottom=.18,top=.96)
finish(fig,'fig_capacity_surface');print(json.dumps(report,ensure_ascii=False))
