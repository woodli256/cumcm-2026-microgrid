"""价格加权余量匹配对照复合图。仅读取已验证结果，不重新求解模型。"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from _utils import plot_style as PS

PS.apply(10.0)
NAME = 'fig_q4_weight_attribution'
SOURCES = ['results/q4_weight_attribution_legacy/report.json',
           'results/q4_price_effects/analysis.json']
legacy, current = [json.loads((ROOT / p).read_text()) for p in SOURCES]
assert legacy['status'] == current['status'] == 'VERIFIED'
keys = [
    ('旧参数不调整', 0.65, legacy['comparisons']['definition'],
     legacy['comparisons']['weights'], legacy['comparisons']['whole_replacement']),
    ('正式不调整', 0.80, current['comparisons']['uniform_empirical_q4_2_vs_q4_2'],
     current['comparisons']['weighted_q4_2_vs_uniform_empirical_q4_2'],
     current['comparisons']['weighted_q4_2_vs_q4_2']),
    ('正式日内调整', 0.65, current['comparisons']['uniform_empirical_q4_3_vs_q4_3'],
     current['comparisons']['weighted_q4_3_vs_uniform_empirical_q4_3'],
     current['comparisons']['weighted_q4_3_vs_q4_3']),
]
row_labels = [f'{label}\nq={q:.2f}' for label, q, *_ in keys]
effects_yuan = np.array([[a['saving_yuan'], w['saving_yuan'], t['saving_yuan']]
                         for _, _, a, w, t in keys])
inventory_yuan = np.array([w['inventory_adjusted_saving_yuan'] for _, _, _, w, _ in keys])
assert np.max(abs(effects_yuan[:, 0] + effects_yuan[:, 1] - effects_yuan[:, 2])) < 1e-7
months = np.array([m['month'] for m in keys[1][3]['monthly']])
assert np.array_equal(months, np.arange(2, 13))
monthly_yuan = np.array([[m['saving_yuan'] for m in row[3]['monthly']] for row in keys[1:]])
assert np.max(abs(monthly_yuan.sum(axis=1) - effects_yuan[1:, 1])) < 1e-7
counts = np.array([[(v > 1e-6).sum(), (v < -1e-6).sum()] for v in monthly_yuan])
assert np.array_equal(counts, [[4, 7], [4, 7]])
assert legacy['inventory']['value_yuan_per_cell_kwh'] == current['inventory']['value_yuan_per_cell_kwh']
value = current['inventory']['value_yuan_per_cell_kwh']
effects = effects_yuan / 10000
monthly = monthly_yuan / 10000
inventory = inventory_yuan / 10000

def signed(v, digits=4):
    # Only suppress sub-cent numerical noise in display. Preserve raw arrays.
    if abs(v) < 0.5 * 10**(-digits):
        v = 0.
    return f'{v:+.{digits}f}' if v else f'{v:.{digits}f}'

fig = plt.figure(figsize=(6.6, 5.9), facecolor='white')
# Shared configuration rows join decomposition and bookkeeping check.
ax = fig.add_axes([.224, .56, .428, .303])
bx = fig.add_axes([.723, .56, .26, .303], sharey=ax)
ax.set_xlim(-1.7, 9.6)
ax.set_ylim(2.5, -.56)
ax.set_yticks([0, 1, 2], row_labels)
ax.tick_params(axis='y', length=0, pad=8)
ax.set_xticks([-1, 0, 3, 6, 9])
ax.set_xlabel('节省费用 / 万元', labelpad=4)
ax.grid(axis='x', color=PS.GRID)
ax.axvline(0, color=PS.GRAY, lw=.75)
ax.spines['left'].set_visible(False)
for i, (algorithm, weight, total) in enumerate(effects):
    # Sequential floating segments. A negative segment extends to the left.
    ax.barh(i+.13, algorithm, left=0, height=.23, color=PS.ORANGE, zorder=3)
    ax.barh(i+.13, weight, left=algorithm, height=.23, color=PS.GREEN, zorder=3)
    ax.plot(total, i+.13, marker='D', color=PS.INK, markersize=3.8, zorder=4)
    # Exact decomposition is readable even when a segment is small at shared scale.
    ax.text(.02, i-.23, f'{algorithm:.3f}'.replace('-0.000', '0.000') + ('  +  ' if weight >= 0 else '  −  ') + f'{abs(weight):.3f}  =  {total:.3f}',
            transform=ax.get_yaxis_transform(), fontsize=9.2, va='center')
    if i < 2:
        ax.axhline(i+.56, color=PS.GRID, lw=.65)
fig.text(.023, .96, '(a) 费用分解', fontsize=11)
fig.text(.69, .96, '(b) 库存修正', fontsize=11)
handles = [Patch(facecolor=PS.ORANGE, label='分位数算法'),
           Patch(facecolor=PS.GREEN, label='价格权重'),
           Line2D([0], [0], ls='none', marker='D', color=PS.INK,
                  markersize=4, label='合计')]
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.204, .931),
           ncol=3, columnspacing=.8, handlelength=1.05, handletextpad=.4,
           fontsize=9.2)
# Numeric companion avoids visually magnifying the very small inventory adjustment.
bx.set_axis_off()
bx.set_xlim(0, 1)
bx.text(.21, -.69, '修正前', ha='center', va='bottom', fontsize=9.5)
bx.text(.77, -.69, '修正后', ha='center', va='bottom', fontsize=9.5)
for i in range(3):
    bx.text(.21, i+.05, signed(effects[i, 1]), ha='center', va='center', fontsize=10)
    bx.text(.77, i+.05, signed(inventory[i]), ha='center', va='center', fontsize=10)
    bx.plot([.425, .565], [i+.05, i+.05], color=PS.GRAY, lw=.8)
    if i < 2:
        bx.axhline(i+.56, color=PS.GRID, lw=.65)
fig.text(.852, .492, '价格权重节省 / 万元', ha='center', fontsize=9.4)
fig.text(.024, .432, '(c) 正式策略逐月权重收益', fontsize=11)
fig.text(.98, .432, '正值节省，负值增费', ha='right', fontsize=9.2)
# Formal configurations stay in the same order as rows 2 and 3 above.
cx = fig.add_axes([.224, .239, .672, .14])
cmap = LinearSegmentedColormap.from_list('weighted_matched_signed', [PS.RED, '#FFFFFF', PS.GREEN])
lim = .70
im = cx.imshow(monthly, cmap=cmap, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
               aspect='auto', interpolation='none')
cx.set_xticks(np.arange(11), [f'{m}月' for m in months])
cx.xaxis.tick_top()
cx.tick_params(axis='both', length=0, pad=6)
cx.set_yticks([0, 1], row_labels[1:])
for spine in cx.spines.values():
    spine.set_visible(False)
for j in range(12):
    cx.axvline(j-.5, color='white', lw=.9)
cx.axhline(.5, color='white', lw=1.0)
for i in range(2):
    for j in range(11):
        cx.text(j, i, f'{monthly[i,j]:+.2f}', ha='center', va='center', fontsize=9.0)
    cx.text(11.15, i, f'{counts[i,0]}降\n{counts[i,1]}增', ha='center', va='center',
            fontsize=9.2, clip_on=False)
cbax = fig.add_axes([.44, .158, .27, .015])
cb = fig.colorbar(im, cax=cbax, orientation='horizontal', ticks=[-.7, 0, .7])
cb.ax.tick_params(labelsize=9.2, length=2.3, pad=3)
fig.text(.745, .152, '节省 / 万元', va='center', fontsize=9.2)
fig.text(.023, .079, '分位数算法为等权线性插值改为等权阶梯；价格权重在同一阶梯算法下比较。',
         fontsize=9.1)
fig.text(.023, .038, f'统计 2—12 月。库存单价为 {value:.5f} 元/kWh，仅用于统一记账。',
         fontsize=9.1)

meta = {
    'data_status': 'real',
    'sources': SOURCES,
    'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES},
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'configuration_order': [{'label': x[0], 'q': x[1]} for x in keys],
    'period': '2月至12月，334天，全部使用附件4同一实际电价结算',
    'effect_definition': ['linear_equal_cost minus staircase_equal_cost',
                          'staircase_equal_cost minus staircase_price_weighted_cost',
                          'linear_equal_cost minus staircase_price_weighted_cost'],
    'effect_columns': ['分位数算法', '价格权重', '合计'],
    'effects_yuan': effects_yuan.tolist(),
    'inventory_adjusted_weight_saving_yuan': inventory_yuan.tolist(),
    'monthly_weight_saving_yuan': monthly_yuan.tolist(),
    'monthly_saving_higher_cost_count': counts.tolist(),
    'inventory_formula': 'C_adj=C+v*(S_start-S_end)',
    'inventory_value_yuan_per_cell_kwh': value,
    'visual_structure': '共享三行配置的浮动分项条与库存修正对照，接正式两类策略逐月匹配收益矩阵',
    'skill_structure_reference': 'complex-figures中的跨面板对象对应、共享顺序和主图辅助图结构；未转录历史论文数值',
    'units': '图中万元；源数组保存元',
    'limits': ['旧参数只用于复核原8.75万元归因，不替换正式q。',
               '价格权重收益仅代表对应q及完整在线规则的样本期差异。',
               '库存修正是统一记账，不是统一首末储量重新优化。',
               '月度4降7增不构成跨年统计显著性证据。'],
    'checks': {'decomposition_additive': True, 'monthly_matches_annual': True,
               'both_formal_rows_4_saving_7_higher_cost': True},
    'requested_paper_width_mm': 160,
    'minimum_effective_font_pt_at_160mm': 9 * 160 / (6.6 * 25.4)
}
PS.save_bundle(fig, NAME, arrays={'q': np.array([x[1] for x in keys]),
    'effects_yuan': effects_yuan, 'inventory_adjusted_weight_saving_yuan': inventory_yuan,
    'month': months, 'monthly_weight_saving_yuan': monthly_yuan,
    'monthly_saving_higher_cost_count': counts}, meta=meta)
metadata_path = ROOT / 'figures/source_data' / f'{NAME}.json'
metadata = json.loads(metadata_path.read_text())
metadata['paper_width_mm'] = 160
metadata['min_effective_font_pt'] = round(9 * 160 / (6.6 * 25.4), 3)
metadata['style']['final_width_mm'] = 160
metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
audit = ROOT / 'audit/问题四补图'
audit.mkdir(parents=True, exist_ok=True)
(audit / '加权对照说明.md').write_text('''# 价格加权匹配对照图

建议置于问题四第8.7节，匹配对照表及其解释之后，作为分项归因与逐月稳定性的合并证据。

图名建议为“价格加权的匹配对照”。通栏160 mm插入，最小有效字号8.59 pt。

图(a)共享三行配置，以同一零线展示先改变分位数算法、再改变价格权重的费用变化。图(b)保持相同行序，比较纯价格权重收益在原费用和统一库存记账后的值，不夸大细小差额。图(c)只展示正式两种策略的同阶梯、同参数纯权重逐月收益，正为节省，负为增费。

真实数据来自 `results/q4_weight_attribution_legacy/report.json` 和 `results/q4_price_effects/analysis.json`，未重新运行优化。文件SHA-256、绘图数组、单位换算和逐月数值均记录在 `figures/source_data/fig_q4_weight_attribution.json`，原始单位为元。旧参数不调整的8.7455万元整体收益中，分位数算法贡献0.4463万元、价格权重贡献8.2991万元。正式不调整纯权重节省0.8676万元，正式日内调整纯权重增加0.6677万元，两类正式策略均为4个月节省、7个月增费。

图中库存单价为0.6895775元/kWh，按 C_adj=C+v*(S_start-S_end)记账，不代表售电收入，也不等同于统一首末状态重优化。具体布局借鉴绘图Skill的跨面板对象对应与共享行序，所有数值使用本题已核验结果，未转录历史论文数值或新增演示数据。

复现命令 `python3 figures/gen_fig_q4_weight_attribution.py`。

自动验证已通过，分项和等于总变化，逐月和等于测试期纯权重收益，正式两行均为4降7增。最终PNG已进行视觉检查。
''', encoding='utf-8')
print(json.dumps({'effects_yuan': effects_yuan.tolist(), 'inventory_yuan': inventory_yuan.tolist(),
                  'month_counts': counts.tolist(), 'output': NAME}, ensure_ascii=False))
