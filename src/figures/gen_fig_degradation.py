"""图4。三项指标并列的注数色阶矩阵，完整保留15个方案的45个指标值。
各指标独立归一化，色条给出单位和范围，颜色深浅不跨指标比较。
"""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
if str(ROOT) not in sys.path:
 sys.path.insert(0,str(ROOT))
from _utils import plot_style as PS
PS.apply(9.2)
INK=PS.INK
def save(fig,name):
 PS.save_bundle(fig,name,meta={'sources':['results/degradation_study.json'],'structure':'三项指标归一化色阶矩阵','notes':'15组控制尺度与吞吐系数组合'})

records=json.loads((ROOT/'results/degradation_study.json').read_text())['records']
rhos=[0,.02,.05,.1,.2];minutes=[10,30,60]
metrics=[('purchase_cost_yuan',1e4,'购电费 / 万元',3,PS.BLUE),
 ('efc',1,'等效循环 / 次',3,PS.GREEN),
 ('mode_switches',1,'状态切换 / 次',0,PS.RED)]
matrices=[]
for key,scale,_,_,_ in metrics:
 matrix=np.array([[next(r[key] for r in records if r['control_minutes']==m and r['rho_yuan_per_cell_kwh']==q)/scale for m in minutes] for q in rhos])
 matrices.append(matrix)
assert len(records)==15 and all(z.shape==(5,3) for z in matrices)
fig=plt.figure(figsize=PS.figsize(5.8,3.55))
axes=[];images=[]
lefts=[.15,.443,.736];panel_width=.24
for j,((key,scale,title,digits,color),data,left) in enumerate(zip(metrics,matrices,lefts)):
 ax=fig.add_axes([left,.285,panel_width,.50]);axes.append(ax)
 cmap=LinearSegmentedColormap.from_list('metric_'+key,['#F5F8FA',color],256)
 norm=Normalize(vmin=data.min(),vmax=data.max())
 im=ax.imshow(data,cmap=cmap,norm=norm,aspect='auto',interpolation='nearest');images.append(im)
 for row in range(5):
  for col in range(3):
   val=data[row,col];shade=norm(val)
   ax.text(col,row,f'{val:.{digits}f}',ha='center',va='center',fontsize=9.8,
      color='white' if shade>.63 else INK)
 ax.set_xticks(range(3),['10','30','60']);ax.xaxis.tick_top();ax.tick_params(axis='x',length=0,pad=7)
 ax.set_yticks(range(5),[f'{q:.2f}' for q in rhos] if j==0 else [])
 ax.tick_params(axis='y',length=0,pad=9)
 ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True)
 ax.grid(which='minor',color='white',lw=1.3);ax.tick_params(which='minor',length=0)
 for spine in ax.spines.values():spine.set_visible(False)
 if j==0:ax.set_ylabel('循环成本系数 κ / (元/kWh)',labelpad=9)
 fig.text(left,.934,chr(97+j).upper(),fontsize=9,fontweight='bold',color=color)
 fig.text(left+panel_width/2,.868,'控制尺度 / 分钟',fontsize=8.2,ha='center',color=PS.GRAY)
 cax=fig.add_axes([left,.167,panel_width,.026])
 cbar=fig.colorbar(im,cax=cax,orientation='horizontal');cbar.outline.set_visible(False)
 ticks=[3.52,3.68,3.84] if j==0 else ([.9,1.2,1.6] if j==1 else [8,16,25])
 cbar.set_ticks(ticks);cbar.ax.tick_params(length=2,width=.5,pad=3,labelsize=8)
fig.text(.55,.047,'深色表示该指标数值较大',ha='center',fontsize=8.5,color=PS.GRAY)
save(fig,'fig_degradation')
