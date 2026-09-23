# -*- coding: utf-8 -*-
"""论文统一绘图规范（唯一视觉配置入口）。

本模块集中管理字体、配色、字号、线宽、marker、网格、边距与保存参数。
其它绘图脚本只允许从本模块取值，不得另行定义一套视觉规范。

语义配色沿用 CLAUDE.md 中的 MH_DATA_FIG_* 自定义调色板（低饱和、可印刷），
并按“语义固定”原则分配角色；同时为每个系列固定线型与 marker，保证黑白打印可区分。
"""
from pathlib import Path
import json
import re
import shutil
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- 尺寸与输出
FIG_WIDTH_IN = 6.6          # 全文统一图宽 167.6 mm
HEIGHTS = {'wide': 3.2, 'pair': 3.9, 'standard': 4.8, 'tall': 5.3, 'dense': 6.0}
DPI = 350                   # 屏幕预览与 PNG 导出分辨率（≥300）
MIN_PT = 9.0                # 图内最小字号（导出前兜底放大）
FINAL_WIDTH_MM = 152.0      # 正文实际排版宽度

# ---------------------------------------------------------------- 字体
CJK_CANDIDATES = [
    'C:/Windows/Fonts/simsun.ttc', 'C:/Windows/Fonts/simsunb.ttf',
    '/System/Library/Fonts/Songti.ttc', '/System/Library/Fonts/Supplemental/Songti.ttc',
    '/Library/Fonts/SourceHanSerifSC-Regular.otf',
    'C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf',
]
LATIN_CANDIDATES = ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'SimSun', 'DejaVu Serif']


def cjk_font_path():
    for p in CJK_CANDIDATES:
        if Path(p).exists():
            try:
                font_manager.fontManager.addfont(p)
                return p
            except Exception:
                continue
    return None


def cjk_font_name():
    p = cjk_font_path()
    return font_manager.FontProperties(fname=p).get_name() if p else 'SimSun'


def latin_font_name():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in LATIN_CANDIDATES:
        if name in installed:
            return name
    return 'DejaVu Serif'


# ---------------------------------------------------------------- 语义配色
def _palette_from_claude():
    """读取 CLAUDE.md 中用户指定的调色板，作为系列色底座。"""
    default = ['#7EAEC5', '#DF9C9B', '#8FBEAA', '#B7A4D0', '#DEBD86', '#A7C8D8', '#C4B5A4']
    f = ROOT / 'CLAUDE.md'
    if not f.exists():
        return default
    m = re.search(r'MH_DATA_FIG_COLORS=([#0-9A-Fa-f,\s]+)', f.read_text(encoding='utf-8', errors='ignore'))
    if not m:
        return default
    cols = [c.strip() for c in m.group(1).split(',') if c.strip().startswith('#')]
    return cols or default


_PALETTE = _palette_from_claude()


def _darken(hex_color, amount=0.45):
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    f = 1 - amount
    return '#%02X%02X%02X' % (int(r * f), int(g * f), int(b * f))


# 角色固定：历史/原始=灰，基准=蓝，本文方法=深蓝，改进/最优=绿，对照=暖金，
# 风险/紧急=降饱和红，第四类方案=紫，区间填充=浅蓝，辅助带=暖灰。
INK = '#293C4A'
GRID = '#E4EAEE'
AXIS = '#9BABB6'
GRAY = '#8A949C'
GRAY_L = _PALETTE[6] if len(_PALETTE) > 6 else '#C4B5A4'
BLUE = _PALETTE[0]
BLUE_L = _PALETTE[5] if len(_PALETTE) > 5 else '#A7C8D8'
BLUE_D = _darken(_PALETTE[0], .42)
GREEN = _PALETTE[2]
ORANGE = _PALETTE[4]
RED = _PALETTE[1]
PURPLE = _PALETTE[3]

ROLE_COLORS = {
    'history': GRAY, 'baseline': BLUE, 'ours': BLUE_D, 'improved': GREEN,
    'alternative': ORANGE, 'risk': RED, 'fourth': PURPLE, 'band': BLUE_L,
}
CYCLE = [BLUE_D, ORANGE, GREEN, PURPLE, RED, GRAY, BLUE]
LINESTYLES = ['-', '--', '-.', ':', (0, (5, 1, 1, 1))]
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']
CMAP_DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list(
    'paper_signed', [PURPLE, '#FAFAFA', ORANGE])
CMAP_SEQUENTIAL = matplotlib.colors.LinearSegmentedColormap.from_list(
    'paper_seq', ['#F5F8FA', BLUE_L, BLUE, BLUE_D])


def figsize(w, h):
    """把原图尺寸换算为统一图宽（167.6 mm），高度按原比例缩放，保持正文版式不变。"""
    return (FIG_WIDTH_IN, round(float(h) * FIG_WIDTH_IN / float(w), 3))


def series(i, role=None):
    """返回第 i 个系列的 (颜色, 线型, marker)，保证任意两个系列至少两项不同。"""
    c = ROLE_COLORS.get(role) if role else None
    return {'color': c or CYCLE[i % len(CYCLE)],
            'ls': LINESTYLES[i % len(LINESTYLES)],
            'marker': MARKERS[i % len(MARKERS)]}


def gray_luminance(hex_color):
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return round(.2126 * r + .7152 * g + .0722 * b, 3)


# ---------------------------------------------------------------- 全局样式
def apply(base_pt=9.2):
    """应用统一视觉规范；所有绘图脚本在开始画图前调用一次。"""
    cjk, latin = cjk_font_name(), latin_font_name()
    # font.family 显式给出拉丁+中文两级回退，保证中文一定有字形来源，
    # 且拉丁字母与数字仍由 Times New Roman 渲染（避免缺字警告与豆腐块）。
    plt.rcParams.update({
        'font.family': [latin, cjk],
        'font.sans-serif': [latin, cjk, 'DejaVu Sans'],
        'font.serif': [latin, cjk, 'DejaVu Serif'],
        'font.monospace': ['Consolas', 'Menlo', 'DejaVu Sans Mono'],
        'axes.unicode_minus': False,
        'mathtext.fontset': 'stix',          # 数学符号与 Times 衬线协调
        'font.size': base_pt,
        'axes.labelsize': base_pt,
        'axes.titlesize': base_pt + .6,
        'xtick.labelsize': base_pt - .2,
        'ytick.labelsize': base_pt - .2,
        'legend.fontsize': base_pt - .2,
        'legend.frameon': False,
        'legend.handlelength': 1.6,
        'legend.columnspacing': 1.2,
        'legend.labelspacing': .35,
        'axes.linewidth': .6,
        'axes.edgecolor': AXIS,
        'axes.labelcolor': INK,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'text.color': INK,
        'xtick.color': INK,
        'ytick.color': INK,
        'xtick.major.width': .6,
        'ytick.major.width': .6,
        'xtick.major.size': 2.6,
        'ytick.major.size': 2.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.axisbelow': True,
        'lines.linewidth': 1.15,
        'lines.markersize': 4.4,
        'lines.markerfacecolor': 'white',
        'lines.markeredgewidth': .9,
        'patch.edgecolor': 'white',
        'patch.linewidth': .7,
        'errorbar.capsize': 2,
        'grid.color': GRID,
        'grid.linewidth': .5,
        'grid.alpha': 1.0,
        'figure.dpi': 120,
        'savefig.dpi': DPI,
        'savefig.bbox': 'standard',
        'savefig.pad_inches': .08,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })
    plt.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=CYCLE)
    return {'cjk': cjk, 'latin': latin, 'base_pt': base_pt}


def clean(ax, grid='y'):
    """统一样式：浅色辅助网格 + 细刻度，不抢视觉注意力。"""
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=.5)
    ax.tick_params(length=2.6, width=.6, pad=3, colors=INK)
    for side in ('left', 'bottom'):
        if side in ax.spines:
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(.6)
    return ax


def panel(ax, label, loc='left'):
    """多面板标识 (a)/(b)/(c)。图内不写整句标题，含义交由正文图注。"""
    ax.set_title(label, loc=loc, pad=6, fontsize=plt.rcParams['axes.titlesize'])
    return ax


def mark_point(ax, x, y, text=None, color=None, dx=6, dy=6, ha='left'):
    """关键点强调：最优参数、极值、转折点。标注保持简短。"""
    color = color or RED
    ax.plot([x], [y], marker='o', ms=5.0, mfc='white', mec=color, mew=1.2, zorder=5)
    if text:
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords='offset points',
                    ha=ha, color=color, fontsize=plt.rcParams['legend.fontsize'])
    return ax


def reference(ax, value, axis='x', text=None, color=None):
    """参考线：目标值、零点、覆盖率名义水平等。"""
    color = color or GRAY
    if axis == 'x':
        ax.axvline(value, color=color, ls=':', lw=.9)
    else:
        ax.axhline(value, color=color, ls=':', lw=.9)
    if text:
        if axis == 'x':
            ax.annotate(text, (value, 1.0), xytext=(3, -10), textcoords='offset points',
                        color=color, fontsize=plt.rcParams['legend.fontsize'], va='top')
        else:
            ax.annotate(text, (0, value), xytext=(4, 3), textcoords='offset points',
                        color=color, fontsize=plt.rcParams['legend.fontsize'])
    return ax


def save_bundle(fig, stem, arrays=None, meta=None, preview_dir=None):
    """统一导出：PDF（正文用矢量）+ 350 dpi PNG + SVG，并登记源数据哈希。"""
    fig.canvas.draw()
    for artist in fig.findobj(Text):
        if artist.get_text():
            artist.set_fontsize(max(MIN_PT, artist.get_fontsize()))
    fig.canvas.draw()
    out = ROOT / 'figures'
    out.mkdir(exist_ok=True)
    written = []
    for ext, kw in (('pdf', {}), ('png', {'dpi': DPI}), ('svg', {})):
        fig.savefig(out / f'{stem}.{ext}', **kw)
        written.append(ext)
    info = {'figure': stem, 'outputs': written, 'png_dpi': DPI,
            'fig_width_in': round(fig.get_figwidth(), 3),
            'min_effective_font_pt': round(
                MIN_PT * FINAL_WIDTH_MM / (fig.get_figwidth() * 25.4), 3)}
    if arrays:
        serial = {k: np.asarray(v) for k, v in arrays.items()}
        assert all(np.isfinite(v).all() for v in serial.values()), stem
        data = out / 'source_data'
        data.mkdir(exist_ok=True)
        np.savez_compressed(data / f'{stem}.npz', **serial)
        info['shapes'] = {k: list(v.shape) for k, v in serial.items()}
    if meta is not None:
        data = out / 'source_data'
        data.mkdir(exist_ok=True)
        payload = dict(meta)
        payload['style'] = style_card()
        payload.update({k: info[k] for k in ('outputs', 'png_dpi', 'min_effective_font_pt')})
        if 'shapes' in info:
            payload['shapes'] = info['shapes']
        (data / f'{stem}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
                                           encoding='utf-8')
    if preview_dir:
        p = Path(preview_dir)
        p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p / f'{stem}.png', dpi=DPI)
        fig.savefig(p / f'{stem}.svg')
    plt.close(fig)
    return info


def style_card():
    """导出规范摘要，供检查记录与图注核对。"""
    return {'figure_width_in': FIG_WIDTH_IN, 'heights': HEIGHTS, 'dpi': DPI,
            'min_font_pt': MIN_PT, 'final_width_mm': FINAL_WIDTH_MM,
            'cjk_font': cjk_font_name(), 'latin_font': latin_font_name(),
            'mathtext': 'stix', 'palette_source': 'CLAUDE.md MH_DATA_FIG_COLORS',
            'roles': ROLE_COLORS, 'cycle': CYCLE, 'linestyles': LINESTYLES, 'markers': MARKERS,
            'gray_luminance': {k: gray_luminance(v) for k, v in ROLE_COLORS.items()}}


if __name__ == '__main__':
    print(json.dumps(style_card(), ensure_ascii=False, indent=2))
