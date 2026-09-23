"""问题四价格结构组合图，使用实际价格事后拟合，不运行调度求解。

运行 python3 figures/gen_fig_q4_price_structure.py
全年热图及右侧比例共用日期轴，三个左侧视图共用日内时刻轴。
"""
from datetime import date
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from _utils import plot_style as PS


def main():
    source = ROOT / 'data/inputs.npz'
    records = ROOT / 'results/improve/improve.json'
    data = np.load(source)
    price = data['price']
    base = data['base'][:, 0]
    assert price.shape == (365, 144) and base.shape == (144,)
    assert np.isfinite(price).all() and np.isfinite(base).all()
    # 与 code/improve_experiments.py::price_structure 完全相同。
    scale = (price @ base) / (base @ base)
    fitted = scale[:, None] * base[None, :]
    residual = price - fitted
    r2 = 1 - np.sum(residual ** 2) / np.sum((price - price.mean()) ** 2)
    summary = {
        'shape_scale_r2': float(r2),
        'residual_std': float(residual.std()),
        'price_std': float(price.std()),
        'mean_per_day_corr': float(np.mean([
            np.corrcoef(price[d], base)[0, 1] for d in range(365)])),
        'daily_scale_min': float(scale.min()),
        'daily_scale_max': float(scale.max()),
        'price_mean': float(price.mean()),
    }
    old = json.loads(records.read_text())['price_structure']
    discrepancies = {key: abs(value - old[key]) for key, value in summary.items()}
    assert max(discrepancies.values()) < 1e-12
    assert np.max(np.abs(residual + fitted - price)) < 1e-12
    # 365个日期在相同时刻的经验分位数，不是置信区间或预测带。
    quantiles = np.quantile(price, [.1, .5, .9], axis=0, method='linear')
    residual_quantiles = np.quantile(residual, [.1, .5, .9], axis=0, method='linear')
    assert quantiles.min() >= 0 and quantiles.max() < 1.80
    assert residual_quantiles.min() > -.24 and residual_quantiles.max() < .24
    x = np.arange(145) / 6
    ends = lambda v: np.r_[v, v[-1]]
    day = np.arange(365) + .5
    dates = [date(2025, m, 1) for m in [1, 3, 5, 7, 9, 11]]
    day_ticks = [(dt - date(2025, 1, 1)).days + .5 for dt in dates]

    PS.apply(10.5)
    fig = plt.figure(figsize=(6.6, 6.4), facecolor='white')
    heat = fig.add_axes([.11, .535, .61, .365])
    level = fig.add_axes([.79, .535, .17, .365], sharey=heat)
    profile = fig.add_axes([.11, .292, .61, .15], sharex=heat)
    resid = fig.add_axes([.11, .09, .61, .12], sharex=heat)
    cax = fig.add_axes([.405, .936, .315, .013])
    fig.text(.11, .975, '(a) 年内价格', va='top', fontsize=11)
    fig.text(.405, .975, '价格 / (元/kWh)', va='top', fontsize=10.5)
    # 数学字体不承担中文，避免混排时缺少中文字形。
    fig.text(.79, .925, '日水平', ha='left', va='bottom', fontsize=10.5)
    fig.text(.867, .925, '$k_j$', ha='left', va='bottom', fontsize=10.5)

    image = heat.imshow(price, origin='upper', interpolation='nearest',
                        aspect='auto', extent=[0, 24, 365, 0],
                        cmap=PS.CMAP_SEQUENTIAL, vmin=0, vmax=1.8,
                        rasterized=True)
    heat.set_yticks(day_ticks, [f'{dt.month}月' for dt in dates])
    heat.set_xticks([0, 4, 8, 12, 16, 20, 24])
    heat.tick_params(axis='x', labelbottom=False, bottom=False)
    heat.set_ylim(365, 0)
    heat.set_xlim(0, 24)
    for dt in dates[1:]:
        yy = (dt - date(2025, 1, 1)).days
        heat.axhline(yy, color='white', lw=.6, alpha=.65)
        level.axhline(yy, color=PS.GRID, lw=.55, zorder=0)
    cb = fig.colorbar(image, cax=cax, orientation='horizontal', ticks=[0, .6, 1.2, 1.8])
    cb.ax.tick_params(length=2, pad=2, labelsize=10.3)
    cb.outline.set_linewidth(.5)

    level.fill_betweenx(day, 1, scale, color=PS.BLUE_L, alpha=.35)
    level.plot(scale, day, color=PS.BLUE_D, lw=.9)
    level.axvline(1, color=PS.GRAY, lw=.75, ls='--')
    level.set_xlim(.70, 1.23)
    level.set_xticks([.8, 1., 1.2])
    level.tick_params(axis='y', left=False, labelleft=False)
    level.set_xlabel('事后拟合', labelpad=5)
    level.spines['left'].set_visible(False)

    fig.text(.11, .478, '(b) 日内形状', va='top', fontsize=11)
    fig.text(.72, .478, '价格 / (元/kWh)', va='top', ha='right', fontsize=10.5)
    profile.fill_between(x, ends(quantiles[0]), ends(quantiles[2]), step='post',
                         color=PS.BLUE_L, alpha=.45, linewidth=0)
    profile.step(x, ends(quantiles[1]), where='post', color=PS.BLUE_D, lw=1.25)
    profile.step(x, ends(base), where='post', color=PS.ORANGE, lw=1.05, ls='--')
    profile.set_ylim(0, 1.8)
    profile.set_yticks([0, .5, 1.0, 1.5])
    profile.tick_params(axis='x', bottom=False, labelbottom=False)
    PS.clean(profile)
    fig.legend(handles=[
        Line2D([], [], color=PS.BLUE_D, lw=1.25, label='全年中位数'),
        Patch(facecolor=PS.BLUE_L, alpha=.45, edgecolor='none', label='10%—90%'),
        Line2D([], [], color=PS.ORANGE, lw=1.05, ls='--', label='基准电价'),
    ], loc='upper left', bbox_to_anchor=(.775, .455), borderaxespad=0,
       handlelength=1.4, handletextpad=.55, labelspacing=.75, fontsize=10.3)

    fig.text(.11, .247, '(c) 拟合残差', va='top', fontsize=11)
    fig.text(.72, .247, '残差 / (元/kWh)', va='top', ha='right', fontsize=10.5)
    resid.fill_between(x, ends(residual_quantiles[0]), ends(residual_quantiles[2]),
                       step='post', color=PS.BLUE_L, alpha=.45, linewidth=0)
    resid.step(x, ends(residual_quantiles[1]), where='post', color=PS.BLUE_D, lw=1.15)
    resid.axhline(0, color=PS.GRAY, lw=.8, ls='--')
    resid.set_ylim(-.24, .24)
    resid.set_yticks([-.2, 0, .2])
    resid.set_xlabel('日内时刻 / h', labelpad=5)
    PS.clean(resid)
    fig.text(.79, .218, f'$R^2={r2:.4f}$', va='top', fontsize=10.5)
    fig.text(.79, .162, '残差标准差\n' + f'{residual.std():.4f} 元/kWh',
             va='top', fontsize=10.5, linespacing=1.45)

    for ax in [heat, level, profile, resid]:
        ax.tick_params(labelsize=10.3)
    for artist in fig.findobj(Text):
        if artist.get_text():
            artist.set_fontsize(max(10.3, artist.get_fontsize()))
    fig.canvas.draw()
    frame = fig.bbox
    clipped = []
    for artist in fig.findobj(Text):
        if not artist.get_text() or not artist.get_visible():
            continue
        bb = artist.get_window_extent(fig.canvas.get_renderer())
        if bb.width and bb.height and (bb.x0 < frame.x0-1 or bb.y0 < frame.y0-1
                                      or bb.x1 > frame.x1+1 or bb.y1 > frame.y1+1):
            clipped.append(artist.get_text())
    assert not clipped, clipped
    assert heat.get_position().x0 == profile.get_position().x0 == resid.get_position().x0
    assert heat.get_position().width == profile.get_position().width == resid.get_position().width
    assert heat.get_ylim() == level.get_ylim()

    stem = 'fig_q4_price_structure'
    meta = {
        'data_status': 'real',
        'sources': ['data/inputs.npz', 'results/improve/improve.json'],
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [source, records]},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_fields': {'price': '实际交付电价，365天×144个十分钟区间，元/kWh',
                          'base[:,0]': '附件1基准电价，144个十分钟区间，元/kWh'},
        'date_range': ['2025-01-01', '2025-12-31'],
        'fit_definition': 'k_j = (price_j @ base)/(base @ base), residual = price - k_j*base',
        'r2_definition': '1 - sum(residual**2)/sum((price-price.mean())**2)',
        'quantiles': '各时刻365个日期的10%、50%、90%经验分位数，线性插值；不是置信区间',
        'information_scope': '全年实现价的事后描述，不是当天决策输入或预测准确率',
        'structure': '热图与右侧日水平共日期轴；热图、日内形状、残差共时刻轴和宽度',
        'paper_width_mm': 160.0,
        'actual_min_font_pt': 10.3,
        'actual_min_effective_font_pt': 10.3 * 160/(6.6*25.4),
        'summary': summary,
        'verification': {'max_metric_difference_vs_existing': max(discrepancies.values()),
                         'missing_values': 0, 'sample_count': int(price.size),
                         'labels_outside_canvas': clipped,
                         'shared_time_axis': True, 'shared_date_axis': True},
    }
    PS.save_bundle(fig, stem, arrays={
        'price_yuan_per_kwh': price, 'baseline_yuan_per_kwh': base,
        'day_index': np.arange(365), 'interval_start_hour': np.arange(144)/6,
        'daily_scale': scale, 'fitted_price_yuan_per_kwh': fitted,
        'residual_yuan_per_kwh': residual, 'price_quantiles': quantiles,
        'residual_quantiles': residual_quantiles,
    }, meta=meta)
    audit = ROOT / 'audit/问题四补图'
    audit.mkdir(parents=True, exist_ok=True)
    text = f'''# 问题四价格结构图

本图使用 data/inputs.npz 中全部365天的实际电价和附件1基准电价，不运行优化求解，也没有重构或演示数据。

复合结构沿用 cumcm-paper-figures 的多视图对应原则。全年热图和右侧日水平曲线共用日期轴。下方日内形状和残差与热图共用0至24时的坐标和绘图区宽度。10%至90%带是同一时刻跨365天的经验分布范围，不是预测带或置信区间。电价和残差中位数采用同色，便于核对分解前后的时段差异。

逐日比例拟合严格使用 code/improve_experiments.py::price_structure 的定义。R²为{r2:.15f}，残差标准差为{residual.std():.15f}元/kWh。七项结构统计量与 results/improve/improve.json 的最大差为{max(discrepancies.values()):.3g}。实际样本共52560个，没有缺失或非有限值。日期覆盖全年，图中比例仅为事后描述，不作为预测输入。

建议插在第8.2节“价格结构与调度作用”第一段之后。正文可补“图\\ref{{fig:q4-price-structure}}按相同日期与时刻展示实际电价、逐日比例及残差，日内主要峰谷位置较稳定，全年变化较多体现为价格水平的变化。”

运行命令为 `python3 figures/gen_fig_q4_price_structure.py`。PDF、PNG和SVG保存在 figures/fig_q4_price_structure 对应格式，完整绘图输入与来源哈希位于 figures/source_data/fig_q4_price_structure.npz 和同名JSON。脚本检查源统计一致、坐标对应和文字不越界，已打开最终PNG检查。已使用纯白画布及统一配色、宋体与Times字体，按160毫米插图宽度，最小有效字号约{10.3*160/(6.6*25.4):.2f}pt。
'''
    (audit / '价格结构说明.md').write_text(text, encoding='utf-8')
    print(json.dumps({'figure': stem, 'summary': summary,
                      'max_existing_metric_difference': max(discrepancies.values())},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
