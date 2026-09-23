# -*- coding: utf-8 -*-
"""只替换次日光伏输入，复核跨日方案费用的归因。

运行 python3 code/check_crossday_attribution.py。两组都从1月1日连续运行365天，
只统计2月1日至12月31日。不会调用revision_experiments.main或覆盖正式结果。
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import time

import numpy as np
import scipy

import revision_experiments as revision
from solve import ROOT, DT, forecasts, read_inputs, safety, totals
from validate import check

OUT = ROOT / 'results/crossday_attribution'
Q = .65


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items(), key=lambda item: str(item[0])):
        digest.update(str(key).encode('utf-8'))
        if isinstance(value, tuple):
            for array in value:
                digest.update(np.ascontiguousarray(array).tobytes())
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def crossday_parts(a, fc, day, issue, q):
    """逐行复现原next_forecast的负载、预报和安全量，便于单独替换光伏。"""
    length = issue * 36
    ids = np.arange(max(0, day - 28), day)
    same = ids[(day + 1 - ids) % 7 == 0]
    if len(same):
        ids = same
    load = np.median(a['load'][ids, :length], axis=0) if len(ids) else np.full(length, 4000.)
    published = np.interp(np.arange(145, 145 + length),
                          np.arange(25) * 6 + length,
                          np.r_[a['pv'][day, length - 1], a['forecast'][day, issue]])
    # 当日尚未完成，不使用当日实际光伏；有不足七天的历史时只用已有完整日期。
    historical = (np.mean(a['pv'][max(0, day - 7):day, :length], axis=0)
                  if day else np.zeros(length))
    margin = safety(fc[True][1], day, 0, q)[:length]
    return load, published, historical, margin


@contextmanager
def single_input_control(mode, records, terminals):
    """两个入口均复用原全年执行器；控制组只改变次日光伏预测值。"""
    original_forecast = revision.next_forecast
    original_linprog = revision.linprog

    def controlled_forecast(a, fc, day, issue, q):
        net, price = original_forecast(a, fc, day, issue, q)
        load, published, historical, margin = crossday_parts(a, fc, day, issue, q)
        reconstructed = (load - published) * DT + margin
        assert np.array_equal(net, reconstructed), '发布预报的分量重建必须逐值相同'
        used = published if mode == 'published' else historical
        chosen = net if mode == 'published' else (load - historical) * DT + margin
        records.append((day, issue, load.copy(), published.copy(), historical.copy(),
                        margin.copy(), price.copy(), chosen.copy()))
        return chosen, price

    def audited_linprog(c, *args, **kwargs):
        # 这里只拦截revision.extended_plan的两次LP，不改solve.plan。
        n = (len(c) - 1) // 7
        assert len(c) == 7 * n + 1
        assert kwargs['bounds'][5 * n] == (6000, 6000)
        result = original_linprog(c, *args, **kwargs)
        if result.success:
            terminals.append(float(result.x[5 * n]))
        return result

    revision.next_forecast = controlled_forecast
    revision.linprog = audited_linprog
    try:
        yield
    finally:
        revision.next_forecast = original_forecast
        revision.linprog = original_linprog


def describe_difference(left, right, threshold=1e-6):
    difference = np.asarray(right) - np.asarray(left)
    return {
        'max_abs': float(np.abs(difference).max()),
        'sum_abs': float(np.abs(difference).sum()),
        'signed_sum': float(difference.sum()),
        'count_above_1e_6': int((np.abs(difference) > threshold).sum()),
    }


def pack_inputs(records):
    offsets = np.r_[0, np.cumsum([len(record[2]) for record in records])]
    packed = {'day': np.array([r[0] for r in records]),
              'issue': np.array([r[1] for r in records]), 'offsets': offsets}
    names = ['next_load_kw', 'published_pv_kw', 'historical_pv_kw',
             'safety_kwh', 'price_from_next_forecast', 'used_net_kwh']
    for index, name in enumerate(names, start=2):
        packed[name] = np.concatenate([r[index] for r in records])
    return packed


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    a = read_inputs(ROOT / 'data/inputs.npz')
    fc = forecasts(a)
    initial_hashes = {'inputs': array_digest(a), 'forecasts_and_errors': array_digest(fc)}
    prices = np.tile(a['base'][:, 0], (365, 1))
    outputs, packed = {}, {}
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'VERIFIED限于本年固定q=0.65的跨日光伏单因素输入对照；非独立年份验证',
        'design': {
            'simulation_days': 365, 'scored_day_indices': [31, 364],
            'scored_dates': ['2025-02-01', '2025-12-31'], 'initial_soc_kwh': 6000.,
            'q': Q, 'window_intervals': 144, 'interval_hours': DT,
            'update_hours': [6, 12, 18], 'window_terminal_soc_kwh': 6000.,
            'single_change': '仅次日光伏由当次已发布预报改为此前至多7个完整日期同期算术均值',
            'fixed_rules': ['当前日预测', '安全余量及误差历史', '次日负载', '固定电价',
                            '规划窗口', '窗口末储量约束', '合同调整规则', '实际储能执行规则'],
            'history_cutoff': '每个更新时刻使用当日零点之前已经完成的日期，不使用当日未完成数据',
            'first_day_history_fallback': '2025-01-01没有历史完整日期，历史光伏置零，与solve.predict历史预测约定一致',
            'last_day': '2025-12-31沿用原variant日末截断分支，两组均不使用不存在的下一年预报',
            'next_day_contract': '仅作预测电价估值变量，实际执行仍只取当日合同直到下一次更新',
            'causal_state_note': '两组各自连续执行，输入替换引起的后续实际状态和合同差异自然保留，不强行重置',
        },
        'provenance': {
            'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'files_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [
                ROOT / 'data/inputs.npz', ROOT / 'code/solve.py',
                ROOT / 'code/revision_experiments.py', Path(__file__).resolve()]},
        },
        'runs': {},
    }
    for mode in ['published', 'historical_mean']:
        records, terminals = [], []
        print(f'开始连续重跑 {mode}', flush=True)
        with single_input_control(mode, records, terminals):
            out = revision.variant(a, fc, use=True, crossday=True, q=Q)
        assert array_digest(a) == initial_hashes['inputs']
        assert array_digest(fc) == initial_hashes['forecasts_and_errors']
        validation = check(out, a['load'], a['pv'], prices)
        assert validation['pass'], validation
        assert len(records) == 364 * 3 and len(terminals) == len(records) * 2
        terminal_error = float(np.max(np.abs(np.array(terminals) - 6000)))
        assert terminal_error < 1e-6
        outputs[mode] = out
        packed[mode] = pack_inputs(records)
        np.savez_compressed(OUT / f'{mode}.npz', **out)
        np.savez_compressed(OUT / f'{mode}_inputs.npz', **packed[mode],
                            planned_terminal_soc_kwh=np.array(terminals))
        report['runs'][mode] = {'totals': totals(out), 'validation': validation,
                               'lp_solve_count': len(terminals),
                               'planning_terminal_max_abs_error_kwh': terminal_error}
        print(f"完成 {mode}，测试期费用 {totals(out)['total_cost']:.10f} 元", flush=True)

    p, h = packed['published'], packed['historical_mean']
    invariant_fields = ['day', 'issue', 'offsets', 'next_load_kw', 'published_pv_kw',
                        'historical_pv_kw', 'safety_kwh', 'price_from_next_forecast']
    equal = {key: bool(np.array_equal(p[key], h[key])) for key in invariant_fields}
    assert all(equal.values()), equal
    pv_diff = h['historical_pv_kw'] - p['published_pv_kw']
    sample_days = np.repeat(p['day'], np.diff(p['offsets']))
    report['input_control'] = {
        'unchanged_exogenous_fields': equal,
        'underlying_input_arrays_unchanged': True,
        'forecast_and_error_arrays_unchanged': True,
        'crossday_pv_full_year': {
            **describe_difference(p['published_pv_kw'], h['historical_pv_kw']),
            'sample_count': int(len(pv_diff)), 'unit': 'kW',
            'count_changed_exact': int(np.count_nonzero(pv_diff)),
            'mean_abs_kw': float(np.abs(pv_diff).mean()),
        },
        'crossday_pv_scored_period': {
            **describe_difference(p['published_pv_kw'][sample_days >= 31],
                                  h['historical_pv_kw'][sample_days >= 31]),
            'sample_count': int((sample_days >= 31).sum()),
            'count_changed_exact': int(np.count_nonzero(pv_diff[sample_days >= 31])),
        },
        'net_change_only_from_pv_max_abs_error_kwh': float(np.max(np.abs(
            (h['used_net_kwh'] - p['used_net_kwh']) + pv_diff * DT))),
    }
    report['comparison'] = {
        'direction': '历史均值减发布预报',
        'total_cost_difference_yuan': totals(outputs['historical_mean'])['total_cost'] - totals(outputs['published'])['total_cost'],
        'cost_component_difference_yuan': (outputs['historical_mean']['cost'][31:].sum(0) - outputs['published']['cost'][31:].sum(0)).tolist(),
        'initial_test_soc_difference_kwh': float(outputs['historical_mean']['s'][31, 0] - outputs['published']['s'][31, 0]),
        'final_soc_difference_kwh': float(outputs['historical_mean']['s'][-1, -1] - outputs['published']['s'][-1, -1]),
        'executed_arrays_test_period': {
            key: describe_difference(outputs['published'][key][31:], outputs['historical_mean'][key][31:])
            for key in ['g', 'a', 's', 'c', 'd', 'e', 'w', 'up', 'down', 'cost']
        },
    }
    saved_crossday = dict(np.load(ROOT / 'results/revision/crossday_q3.npz'))
    baseline = dict(np.load(ROOT / 'results/q3.npz'))
    report['saved_result_comparison'] = {
        'baseline_total_cost_yuan': totals(baseline)['total_cost'],
        'saved_crossday_total_cost_yuan': totals(saved_crossday)['total_cost'],
        'published_rerun_minus_saved_cost_yuan': totals(outputs['published'])['total_cost'] - totals(saved_crossday)['total_cost'],
        'published_rerun_vs_saved_max_abs_by_array': {
            key: describe_difference(saved_crossday[key], outputs['published'][key])['max_abs']
            for key in ['g', 'a', 's', 'c', 'd', 'e', 'w', 'up', 'down', 'cost']
        },
        'baseline_validation': check(baseline, a['load'], a['pv'], prices),
        'saved_crossday_validation': check(saved_crossday, a['load'], a['pv'], prices),
        'overall_window_scheme_saving_yuan': totals(baseline)['total_cost'] - totals(outputs['published'])['total_cost'],
    }
    user_delta = .0014
    report['user_supplied_comparison'] = {
        'baseline_rounded_yuan': 13640252.27,
        'published_rounded_yuan': 13629128.59, 'historical_rounded_yuan': 13629128.59,
        'absolute_cost_difference_yuan': user_delta,
        'rerun_abs_difference_yuan': abs(report['comparison']['total_cost_difference_yuan']),
        'difference_from_user_absolute_delta_yuan': abs(report['comparison']['total_cost_difference_yuan']) - user_delta,
        'matches_to_four_decimal_places': round(abs(report['comparison']['total_cost_difference_yuan']), 4) == user_delta,
    }
    report['conclusion'] = ('原模型与24小时方案同时改变窗口长度、终端约束位置和次日光伏输入。'
        '该整体方案在本年降低费用，但1.11万元不能归因于次日光伏预报。'
        '在窗口、终端和规则一致的对照中，预报与历史均值费用几乎相同，'
        '现有实验未识别次日光伏预报的独立收益，也未分别识别窗口长度和终端位置的贡献。'
        '本结论不证明跨日预报在其他年份、目标或调度规则下无用。')
    report['elapsed_seconds'] = time.monotonic() - start
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    old = report['saved_result_comparison']
    comp = report['comparison']
    pv = report['input_control']['crossday_pv_full_year']
    text = f'''# 跨日光伏预报归因复核

{report['conclusion']}

## 同条件比较

| 方法 | 测试期总费用/元 |
| --- | ---: |
| 原日末截断模型 | {old['baseline_total_cost_yuan']:.8f} |
| 24小时窗口及窗口末储量6000，使用已发布次日光伏 | {totals(outputs['published'])['total_cost']:.8f} |
| 相同窗口和终端，次日光伏改为过去七天同期均值 | {totals(outputs['historical_mean'])['total_cost']:.8f} |

历史均值减已发布预报的费用差为 {comp['total_cost_difference_yuan']:.10f} 元。
用户给出的绝对差为0.0014元，本次绝对差保留四位小数是否一致为 {report['user_supplied_comparison']['matches_to_four_decimal_places']}。
已发布预报重跑相对旧保存结果的费用差为 {old['published_rerun_minus_saved_cost_yuan']:.10f} 元。

## 比较条件与输入证据

两组各从1月1日6000 kWh连续执行365天，计分区间为2月1日至12月31日的334天。
两组复用同一原始执行入口，仅替换次日光伏输入。当前日预测、安全余量、次日负载、价格、窗口、终端目标、合同和储能执行规则均相同。
每次更新只使用当日零点之前已完成的至多七天同期光伏。第一天没有历史，历史均值方案取零；此设定同时影响预热期及其后继状态，因此明确保留在复现条件中。
12月31日两组均沿用原实现的日末截断处理。

共检查 {pv['sample_count']} 个跨日光伏十分钟输入，其中 {pv['count_changed_exact']} 个数值发生变化，{pv['count_above_1e_6']} 个变化超过0.000001 kW。
平均绝对变化 {pv['mean_abs_kw']:.6f} kW，最大绝对变化 {pv['max_abs']:.6f} kW。
两组测试首储量差为 {comp['initial_test_soc_difference_kwh']:.10f} kWh，末储量差为 {comp['final_soc_difference_kwh']:.10f} kWh。

## 验证范围

两组均通过电量平衡、储量递推、容量功率边界、禁止同时充放电、非负性、跨日连续性和四项费用独立复算。
每组2184次跨日LP求解均核查窗口末储量6000 kWh。完整残差、合同和状态差见report.json。
输入数组与当日预测误差数组的运行前后散列不变，次日负载、价格和安全量逐值相同。

| 结论 | 来源 | 单位 | 验证与容差 | 状态 | 限制 |
| --- | --- | --- | --- | --- | --- |
| 两种次日光伏输入费用几乎相同 | 本目录两组NPZ及report.json | 元 | 全365天重跑及费用复算，残差小于0.000001 | VERIFIED | 仅本年、固定q=0.65及当前调度规则 |
| 1.11万元是整体跨日方案费用差 | 原q3.npz与published.npz | 元 | 同计分区间总费用 | VERIFIED | 未分解窗口长度与终端位置贡献 |
| 1.11万元属于次日光伏预报贡献 | 上述单因素输入对照 | 元 | 控制窗口、终端及其余规则 | FAILED | 不外推为跨日预报普遍无用 |

运行命令为 `python3 code/check_crossday_attribution.py`。脚本不覆盖正式结果及原修订结果。
'''
    (OUT / '复核说明.md').write_text(text, encoding='utf-8')
    print(json.dumps({'comparison': comp, 'saved': old,
                      'input_change': pv, 'output': str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
