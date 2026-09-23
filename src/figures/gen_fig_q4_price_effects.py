"""问题四同价价格预测对照。共享日期轴的累计效果与逐月费用分解。"""
from pathlib import Path
from datetime import date, timedelta
import sys
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from _utils import plot_style as PS
PS.apply(10.5)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

NAME = 'fig_q4_price_effects'
START = 31
MONTH = np.array([(date(2025, 1, 1) + timedelta(days=i)).month for i in range(365)])
PATHS = ['data/inputs.npz', 'results/q2.npz', 'results/q3.npz',
         'results/q4_2.npz', 'results/q4_3.npz', 'results/q4_price_effects/analysis.json']
REPORT = json.loads((ROOT / PATHS[-1]).read_text())
PRICE = np.load(ROOT / PATHS[0])['price']


def components(name):
    with np.load(ROOT / f'results/{name}.npz') as z:
        up = np.maximum(z['a'] - z['g'], 0)
        down = np.maximum(z['g'] - z['a'], 0)
        out = np.stack([np.sum(PRICE * z['g'], axis=1),
                        np.sum(5 * PRICE * z['e'], axis=1),
                        np.sum(1.5 * PRICE * up, axis=1),
                        np.sum(0.5 * PRICE * down, axis=1)], axis=1)
    assert np.isfinite(out).all()
    return out


DIFF = np.stack([components('q2') - components('q4_2'),
                 components('q3') - components('q4_3')])
DAILY = DIFF[:, START:].sum(axis=2)
CUM = np.concatenate([np.zeros((2, 1)), np.cumsum(DAILY, axis=1)], axis=1)
MONTHLY = np.stack([DIFF[:, MONTH == m].sum(axis=1) for m in range(2, 13)], axis=1)
assert np.max(np.abs(MONTHLY[:, :, 3])) < 1e-6
checks = []
for i, key in enumerate(['q4_2_vs_q2', 'q4_3_vs_q3']):
    c = REPORT['comparisons'][key]
    assert abs(CUM[i, -1] - c['saving_yuan']) < 1e-6
    assert np.max(np.abs(MONTHLY[i].sum(axis=0) - c['component_saving_yuan'])) < 1e-6
    assert np.max(np.abs(MONTHLY[i].sum(axis=1) - [v['saving_yuan'] for v in c['monthly']])) < 1e-6
    checks.append({'key': key, 'total_saving_yuan': float(CUM[i, -1]),
                   'monthly_positive': int(np.sum(MONTHLY[i].sum(axis=1) > 1e-6)),
                   'monthly_negative': int(np.sum(MONTHLY[i].sum(axis=1) < -1e-6)),
                   'component_saving_yuan': MONTHLY[i].sum(axis=0).tolist()})

# 历年优秀论文中共享坐标、局部变化与总量对应的结构，在这里用于同一费用差的时间尺度分解。
fig, axes = plt.subplots(3, 1, figsize=(6.6, 6.3), sharex=True,
                         gridspec_kw={'height_ratios': [1.45, 1, 1]})
fig.subplots_adjust(left=.12, right=.96, bottom=.17, top=.925, hspace=.47)
edges = np.array([(date(2025, m, 1) - date(2025, 2, 1)).days for m in range(2, 13)] + [334])
centers = (edges[:-1] + edges[1:]) / 2
names = ['不调整（q=0.8）', '日内调整（q=0.65）']
line_colors = [PS.BLUE_D, PS.PURPLE]
for ax in axes:
    PS.clean(ax, grid='y')
    ax.set_xlim(0, 334)
    ax.axhline(0, color=PS.INK, lw=.75, zorder=2)
    for x in edges[1:-1]:
        ax.axvline(x, color=PS.GRID, lw=.4, zorder=0)

ax = axes[0]
ax.set_title('(a) 累计费用变化', loc='left', fontsize=10.5, pad=9)
ax.text(1, 1.075, '累计节省 / 万元', transform=ax.transAxes, ha='right', fontsize=10.5)
for i, (color, style) in enumerate(zip(line_colors, ['-', '--'])):
    ax.plot(np.arange(335), CUM[i] / 1e4, color=color, ls=style, lw=1.25,
            label=names[i])
    ax.plot(334, CUM[i, -1] / 1e4, 'o', ms=4.5, mfc=color, mec='white', clip_on=False)
    ax.annotate(f'{CUM[i, -1] / 1e4:+.3f}', (334, CUM[i, -1] / 1e4),
                xytext=(-5, 8 if i == 0 else -16), textcoords='offset points',
                ha='right', color=color, fontsize=10.5)
lo = np.min(CUM) / 1e4; hi = np.max(CUM) / 1e4
ax.set_ylim(lo - .30, hi + .45)
ax.legend(loc='upper left', ncol=1, fontsize=10, handlelength=2.3)

colors = [PS.BLUE, PS.ORANGE, PS.GREEN]
labels = ['计划费', '紧急费', '增购费']
low = MONTHLY[:, :, :3].clip(max=0).sum(axis=2).min() / 1e4
high = MONTHLY[:, :, :3].clip(min=0).sum(axis=2).max() / 1e4
lim = max(abs(low), abs(high)) * 1.18
for i, ax in enumerate(axes[1:]):
    ax.set_title(f'({chr(98+i)}) {names[i]}的逐月分解', loc='left', fontsize=10.5, pad=9)
    ax.text(1, 1.085, '节省 / 万元', transform=ax.transAxes, ha='right', fontsize=10.5)
    positive = np.zeros(11); negative = np.zeros(11)
    for j, color in enumerate(colors):
        v = MONTHLY[i, :, j] / 1e4
        base = np.where(v >= 0, positive, negative)
        ax.bar(centers, v, width=np.diff(edges) * .55, bottom=base,
               color=color, edgecolor='white', lw=.5, zorder=3)
        positive += np.clip(v, 0, None)
        negative += np.clip(v, None, 0)
    # 空心菱形显示净额，避免把正负分项的总跨度误认成总节省。
    ax.plot(centers, MONTHLY[i].sum(axis=1) / 1e4, ls='none', marker='D',
            ms=3.7, mfc='white', mec=PS.INK, mew=.8, zorder=4)
    ax.set_ylim(-lim, lim)
    ax.set_yticks([-1, 0, 1])
axes[-1].set_xticks(centers, [f'{m}月' for m in range(2, 13)])
axes[-1].set_xlabel('2025年月度', labelpad=6)
fig.legend([Patch(facecolor=c, edgecolor='white') for c in colors] +
           [plt.Line2D([], [], ls='none', marker='D', mfc='white', mec=PS.INK, ms=4)],
           labels + ['净节省'], ncol=4, loc='lower center', bbox_to_anchor=(.54, .015), fontsize=10)

meta = {'data_status': 'real', 'sources': PATHS,
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in PATHS},
        'figure_structure': '同一日期轴连接每日累计差额与两类策略的逐月费用构成；月内条形由原始日差额精确求和',
        'units': '绘图万元，源数据元', 'saving_definition': '固定价生成动作在实际价格下的费用减去历史预测价生成动作的实际费用',
        'evaluation': REPORT['evaluation'], 'checks': checks,
        'notes': '正值为降费、负值为增费；净节省菱形为各分项之和；减购费用差为0；仅在各自相同q和信息规则内比较',
        'paper_width_mm': 160.0, 'min_font_pt': 10,
        'reference_structure': 'cumcm-paper-figures references/examples.md 中2021 A偏差/绝对偏差共享坐标与2022 A同对象跨面板对应；complex-figures.md多视图共享数据要求'}
PS.save_bundle(fig, NAME, arrays={'day_boundary': np.arange(335), 'daily_saving_yuan': DAILY,
    'cumulative_saving_yuan': CUM, 'months': np.arange(2, 13), 'month_edges': edges,
    'monthly_component_saving_yuan': MONTHLY}, meta=meta)
audit = ROOT / 'audit/问题四补图'; audit.mkdir(parents=True, exist_ok=True)
(audit / '同价对照验证.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
print(json.dumps(checks, ensure_ascii=False))
