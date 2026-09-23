# -*- coding: utf-8 -*-
"""本轮改进的三项新增实验。

1) 完美信息下界：在相同初末储量下，允许预知全部实际负载、光伏与电价时重解LP，
   给出因果策略与下界的差距，补上原稿缺少的最优性间隙。
2) 波动电价的结构诊断：把附件4逐日价格分解为“当日水平×固定日内形状”，
   说明价格预测的调度价值受结构限制，并给出三种预测器的共同目标MAE。
3) 问题三增加决策时刻的价值：在0、6、12、18点之外增加3、9、15、21点的
   状态刷新（不引入新预报），并给出“同位置换成完美光伏”的上界。

输出保存在 results/improve/，不覆盖任何正式结果。
"""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from solve import (ROOT, DT, ETA, POWER, LO, HI, QS, Q_MAIN, read_inputs, forecasts,
                   predict, safety, plan, step, totals)
from validate import check

OUT = ROOT / 'results/improve'
OUT.mkdir(exist_ok=True)
STEPS_PUBLISHED = (36, 72, 108)
STEPS_REFRESH = (18, 36, 54, 72, 90, 108, 126)


def perfect_bound(a, start, end, s_start, s_end, dynamic):
    """完美信息下界：预知实际净需求与结算电价时的最小购电费。"""
    days = np.arange(start, end)
    T = len(days) * 144
    net = ((a['load'][days] - a['pv'][days]) / 6).ravel()
    price = (a['price'][days] if dynamic else np.tile(a['base'][:, 0], (len(days), 1))).ravel()
    N = 5 * T + 1
    A = lil_matrix((2 * T + 2, N))
    b = np.zeros(2 * T + 2)
    for t in range(T):
        A[t, t] = 1; A[t, T + t] = -1; A[t, 2 * T + t] = 1; A[t, 3 * T + t] = -1
        b[t] = net[t]
        A[T + t, 4 * T + t] = -1; A[T + t, 4 * T + t + 1] = 1
        A[T + t, T + t] = -ETA; A[T + t, 2 * T + t] = 1 / ETA
    A[2 * T, 4 * T] = 1; b[2 * T] = s_start
    A[2 * T + 1, 5 * T] = 1; b[2 * T + 1] = s_end
    c = np.concatenate([price, np.zeros(N - T)])
    lb = np.concatenate([np.zeros(4 * T), np.full(T + 1, LO)])
    ub = np.concatenate([np.full(T, np.inf), np.full(T, POWER * DT), np.full(T, POWER * DT),
                         np.full(T, np.inf), np.full(T + 1, HI)])
    res = linprog(c, A_eq=A.tocsr(), b_eq=b, bounds=list(zip(lb, ub)), method='highs')
    if not res.success:
        raise RuntimeError(res.message)
    return {'cost': float(res.fun), 'purchase': float(res.x[:T].sum()),
            'charge': float(res.x[T:2 * T].sum()), 'discharge': float(res.x[2 * T:3 * T].sum()),
            'waste': float(res.x[3 * T:4 * T].sum())}


def price_structure(a):
    base = a['base'][:, 0]
    price = a['price']
    scale = (price @ base) / (base @ base)
    resid = price - scale[:, None] * base[None, :]
    fit = 1 - (resid ** 2).sum() / ((price - price.mean()) ** 2).sum()
    per_day_corr = float(np.mean([np.corrcoef(price[d], base)[0, 1] for d in range(365)]))
    tests = slice(31, 365)
    fc_med = np.zeros_like(price)
    fc_persist = np.zeros_like(price)
    fc_shape = np.zeros_like(price)
    for d in range(365):
        hist = price[max(0, d - 7):d]
        fc_med[d] = np.median(hist, axis=0) if len(hist) else base
        fc_persist[d] = price[d - 1] if d else base
        ks = scale[max(0, d - 7):d]
        fc_shape[d] = (np.median(ks) if len(ks) else scale[d]) * base
    mae = lambda x: float(np.abs(x[tests] - price[tests]).mean())
    return {'shape_scale_r2': float(fit), 'residual_std': float(resid.std()),
            'price_std': float(price.std()), 'mean_per_day_corr': per_day_corr,
            'daily_scale_min': float(scale.min()), 'daily_scale_max': float(scale.max()),
            'price_mean': float(price.mean()),
            'mae_same_time_median_7d': mae(fc_med), 'mae_persistence_1d': mae(fc_persist),
            'mae_shape_times_median_scale_7d': mae(fc_shape)}


def run_refresh(a, fc, q, steps, perfect=False):
    """在给定决策时刻重解剩余合同；perfect=True 时用实际光伏替换当前时段的预报。"""
    pred, err = fc[True]
    out = {k: np.zeros((365, 144)) for k in ['g', 'a', 'c', 'd', 'e', 'w', 'up', 'down']}
    out['s'] = np.zeros((365, 145)); out['cost'] = np.zeros((365, 4))
    state = 6000.
    for day in range(365):
        price = a['base'][:, 0]
        out['s'][day, 0] = state
        g = plan(pred[day, 0] + safety(err, day, 0, q), price, state)['g']
        out['g'][day] = g
        active = g.copy()
        for t in range(144):
            if t in steps and t > 0:
                issue = min(t // 36, 3)
                off = 36 * issue
                net = pred[day, issue, t:] + safety(err, day, issue, q)[t - off:]
                if perfect:
                    net = (a['load'][day, t:] - a['pv'][day, t:]) * DT + safety(err, day, issue, q)[t - off:]
                active[t:] = plan(net, price[t:], state, original=g[t:])['g']
            actual = (a['load'][day, t] - a['pv'][day, t]) * DT
            ch, di, em, w, state = step(active[t], actual, state)
            for key, val in [('a', active[t]), ('c', ch), ('d', di), ('e', em), ('w', w)]:
                out[key][day, t] = val
            out['s'][day, t + 1] = state
        out['up'][day] = np.maximum(active - g, 0)
        out['down'][day] = np.maximum(g - active, 0)
        out['cost'][day] = [price @ g, 5 * price @ out['e'][day],
                            1.5 * price @ out['up'][day], .5 * price @ out['down'][day]]
    out['days'] = np.arange(365)
    return out


def quantile_refine(a, fc):
    """调价类问题在解析区间内的细分点，用于说明平坦区。"""
    out = {}
    for name, use, dynamic in [('q3', True, False), ('q4_3', True, True)]:
        curve = {}
        for q in (1 / 3., 0.4, 0.5, 0.55):
            o = run(a, fc, use, dynamic, q)
            t = totals(o)
            curve[f'{q:.4f}'] = {'total_cost': t['total_cost'], 'e': t['e'], 'up': t['up'], 'w': t['w']}
            print('refine', name, q, round(t['total_cost'], 2), flush=True)
        out[name] = curve
    return out


def main():
    a = read_inputs(ROOT / 'data/inputs.npz')
    fc = forecasts(a)
    report = json.loads((OUT / 'improve.json').read_text(encoding='utf-8')) if \
        (OUT / 'improve.json').exists() else {}
    if '--refine' in sys.argv:
        report['quantile_refine'] = quantile_refine(a, fc)
        (OUT / 'improve.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('已更新 results/improve/improve.json 的分位数细分点')
        return

    # 1) 完美信息下界与最优性间隙
    bounds = {}
    for name, dynamic in [('q2', False), ('q3', False), ('q4_2', True), ('q4_3', True)]:
        o = dict(np.load(ROOT / f'results/{name}.npz'))
        keep = o['days'] >= 31
        s_start = float(o['s'][31, 0]); s_end = float(o['s'][-1, -1])
        res = perfect_bound(a, 31, 365, s_start, s_end, dynamic)
        causal = totals(o)['total_cost']
        bounds[name] = {**res, 'causal_cost': causal,
                        'gap': causal - res['cost'], 'gap_ratio': (causal - res['cost']) / causal,
                        's_start': s_start, 's_end': s_end,
                        'charge_causal': totals(o)['c'], 'discharge_causal': totals(o)['d']}
        print('bound', name, round(res['cost'], 2), 'causal', round(causal, 2), flush=True)
    report['perfect_bound'] = bounds

    # 2) 波动电价结构诊断
    report['price_structure'] = price_structure(a)
    print('price structure', report['price_structure'], flush=True)

    # 3) 增加决策时刻的价值
    q3 = Q_MAIN['q3']
    refresh = run_refresh(a, fc, q3, STEPS_REFRESH, perfect=False)
    perfect = run_refresh(a, fc, q3, STEPS_REFRESH, perfect=True)
    price = np.tile(a['base'][:, 0], (365, 1))
    for tag, o in [('refresh', refresh), ('refresh_perfect', perfect)]:
        v = check(o, a['load'], a['pv'], price)
        assert v['pass'], (tag, v)
        np.savez_compressed(ROOT / f'results/improve/{tag}.npz', **o)
        print(tag, round(totals(o)['total_cost'], 2), flush=True)
    base = totals(dict(np.load(ROOT / 'results/q3.npz')))
    report['extra_steps'] = {
        'published_steps': list(STEPS_PUBLISHED), 'refresh_steps': list(STEPS_REFRESH),
        'published': base, 'refresh': totals(refresh), 'refresh_perfect': totals(perfect),
        'saving_refresh': base['total_cost'] - totals(refresh)['total_cost'],
        'saving_refresh_perfect': base['total_cost'] - totals(perfect)['total_cost'],
        'up_refresh': totals(refresh)['up'], 'up_perfect': totals(perfect)['up']}
    print('extra steps', report['extra_steps']['saving_refresh'],
          report['extra_steps']['saving_refresh_perfect'], flush=True)

    (OUT / 'improve.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                      encoding='utf-8')
    print('已写出 results/improve/improve.json')


if __name__ == '__main__':
    main()
