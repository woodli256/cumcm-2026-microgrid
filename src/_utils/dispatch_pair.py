"""以问题一正式数据绘制同排的日调度与竖向电价色带。"""
from pathlib import Path
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
from matplotlib.colors import Normalize
from _utils import plot_style as PS

ROOT = Path(__file__).resolve().parents[1]


def dispatch():
    PS.apply(10.0)
    a = np.load(ROOT / 'data/inputs.npz')
    q = np.load(ROOT / 'results/q1.npz')
    edges = np.arange(145) / 6
    centers = (edges[:-1] + edges[1:]) / 2
    load = a['base'][:, 1] / 1000
    pv = a['base'][:, 2] / 1000
    buy, charge, discharge = [q[k] * 6 / 1000 for k in ('g', 'c', 'd')]
    storage = q['s'] / 1000
    price = a['base'][:, 0]

    fig = plt.figure(figsize=(PS.FIG_WIDTH_IN, 3.15))
    ax = fig.add_axes([.075, .18, .60, .55])
    state = ax.twinx()
    ax.bar(centers, charge, width=1/6, color=PS.GREEN, alpha=.9,
           linewidth=0, zorder=2)
    ax.bar(centers, -discharge, width=1/6, color=PS.ORANGE, alpha=.7,
           linewidth=0, zorder=2)
    ax.stairs(load, edges, baseline=None, color=PS.BLUE_D, lw=1.15, zorder=4)
    ax.stairs(pv, edges, baseline=None, color=PS.ORANGE, lw=1.25, zorder=4)
    ax.stairs(buy, edges, baseline=None, color=PS.BLUE_D, ls=':', lw=1.05, zorder=3)
    state.plot(edges, storage, color=PS.PURPLE, lw=1.6)
    ax.set(xlim=(0, 24), ylim=(-5.5, 10.5), xticks=range(0, 25, 4),
           yticks=[-5, 0, 5, 10], xlabel='时刻 / h')
    state.set(ylim=(0, 12), yticks=[1.2, 6, 10.8], yticklabels=['1.2', '6.0', '10.8'])
    state.spines['right'].set_visible(True)
    state.spines['right'].set_color(PS.AXIS)
    state.spines['top'].set_visible(False)
    state.tick_params(axis='y', colors=PS.INK, pad=3)
    ax.text(0, 1.045, '功率 / MW', transform=ax.transAxes, ha='left', va='bottom')
    ax.text(1, 1.045, '储量 / MWh', transform=ax.transAxes, ha='right', va='bottom')
    ax.axhline(0, color=PS.AXIS, lw=.65, zorder=1)
    PS.clean(ax)
    for lo, hi, label, color in [(0, 4, '低价充电', PS.BLUE_D),
                                  (18, 21, '高价放电', PS.ORANGE)]:
        ax.add_patch(Rectangle((lo, -5.5), hi-lo, 16,
                              facecolor=color+'09', edgecolor=color,
                              lw=.65, ls=(0, (4, 3)), zorder=0))
        ax.text((lo+hi)/2, 9.4, label, ha='center', va='center',
                fontsize=9.5, color=PS.INK)

    handles = [Line2D([], [], color=PS.BLUE_D, label='小区负载'),
               Line2D([], [], color=PS.BLUE_D, ls=':', label='计划购电'),
               Line2D([], [], color=PS.ORANGE, label='光伏发电'),
               Patch(facecolor=PS.GREEN, label='储能充电'),
               Line2D([], [], color=PS.PURPLE, label='储能储量'),
               Patch(facecolor=PS.ORANGE, alpha=.7, label='储能放电')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.38, .995),
               ncol=3, fontsize=9.5, handlelength=1.6,
               columnspacing=1.25, handletextpad=.5, labelspacing=.5)

    priceax = fig.add_axes([.835, .18, .045, .55])
    priceax.imshow(price[:, None], extent=[0, 1, 0, 24], origin='lower',
                   cmap='YlOrBr', vmin=0, vmax=1.4, aspect='auto',
                   interpolation='nearest')
    priceax.set(xticks=[], yticks=range(0, 25, 4), ylim=(0, 24))
    priceax.tick_params(axis='y', pad=3)
    priceax.spines['right'].set_visible(True)
    priceax.spines['top'].set_visible(True)
    priceax.text(.5, -.17, '时刻 / h', transform=priceax.transAxes,
                 ha='center', va='top')
    cax = fig.add_axes([.91, .18, .017, .55])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1.4), cmap='YlOrBr'),
                      cax=cax, ticks=[0, .7, 1.4])
    cb.ax.tick_params(pad=3)
    fig.text(.884, .80, '电价 / (元/kWh)', ha='center', va='bottom')

    fig.canvas.draw()
    for artist in fig.findobj(Text):
        artist.set_fontsize(max(9.5, artist.get_fontsize()))
    sources = ['data/inputs.npz', 'results/q1.npz']
    PS.save_bundle(fig, 'fig_dispatch', arrays={
        'time_edges_h': edges, 'load_mw': load, 'pv_mw': pv,
        'charge_mw': charge, 'discharge_mw': discharge, 'purchase_mw': buy,
        'storage_mwh': storage, 'price_yuan_kwh': price,
        'highlight_hours': [[0, 4], [18, 21]],
    }, meta={
        'data_status': 'real', 'sources': sources,
        'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        'structure': '左侧双纵轴日调度曲线，右侧同日分时电价竖向色带，同排等高',
        'visual_reference': '用户本轮提供的两张图，数值取正式附件与问题一结果',
        'notes': '功率为区间常值，不平滑；储量为145个区间边界点，单位MWh；电价144段按0至24时自下向上排列，色阶0至1.4元/kWh。',
        'paper_width_mm': 160.0, 'actual_min_font_pt': 9.5,
        'actual_min_effective_font_pt': round(9.5 * 160 / (PS.FIG_WIDTH_IN * 25.4), 3),
    })

