# -*- coding: utf-8 -*-
"""核对问题三数学表述与正式算法，只写核验报告，不改正式结果。

独立重建预测、经验分位数和合同 LP。验证当前合同区间时允许
线性规划存在多个最优解，不把不同最优轨迹误判为数据错误。
"""
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix

from solve import (ROOT, DT, ETA, HI, LO, POWER, Q_MAIN, forecasts,
                   plan, predict, read_inputs, safety, step, structure)
from validate import check


OUTPUT = ROOT / 'audit/建模数学增强/问题三公式实现核验.json'
DATES = ('2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21')
TOL = 1e-6


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def historical_load(a, day):
    ids = [h for h in range(max(0, day - 28), day) if (day - h) % 7 == 0]
    if not ids:
        ids = list(range(max(0, day - 7), day))
    return np.median(a['load'][ids], axis=0) if ids else np.full(144, 4000.)


def reference_prediction(a, day, issue):
    """按右端点定义手工线性插值，并单独写出负载水平修正。"""
    k = 36 * issue
    base = historical_load(a, day)
    raw_ratio = 1.
    if k:
        raw_ratio = float(np.median(a['load'][day, k-6:k] /
                                    np.maximum(base[k-6:k], 1.)))
    ratio = min(1.3, max(.7, raw_ratio))
    endpoint = np.arange(k + 1, 145) - k
    left_index = (endpoint - 1) // 6
    fraction = (endpoint - 6 * left_index) / 6.
    left_pv = float(a['pv'][day, k-1]) if k else 0.
    nodes = np.r_[left_pv, a['forecast'][day, issue]]
    pv = nodes[left_index] + fraction * (nodes[left_index + 1] - nodes[left_index])
    net = (ratio * base[k:] - pv) * DT
    prices = (np.median(a['price'][max(0, day-7):day], axis=0)
              if day else a['base'][:, 0].copy())
    return net, prices[k:], raw_ratio, ratio


def linear_quantile(values, q):
    values = np.sort(np.asarray(values, dtype=float))
    position = (len(values) - 1) * q
    lo, hi = int(np.floor(position)), int(np.ceil(position))
    return float(values[lo] + (position - lo) * (values[hi] - values[lo]))


def reference_margin(err, day, issue, q):
    k = 36 * issue
    result = np.zeros(144-k)
    pool_sizes = []
    if day < 7:
        return result, pool_sizes
    for lo in range(k, 144, 12):
        values = [float(err[h, issue, t])
                  for h in range(max(1, day-28), day)
                  for t in range(lo, lo+12) if np.isfinite(err[h, issue, t])]
        pool_sizes.append(len(values))
        if values:
            result[lo-k:lo-k+12] = linear_quantile(values, q)
    return result, pool_sizes


def independent_problem(net, price, initial, original=None):
    """变量依次为合同、充电、放电、未利用量、储量、增购、减购。"""
    n = len(net)
    adjustment = original is not None
    variables = 5*n + 1 + (2*n if adjustment else 0)
    rows = 3*n if adjustment else 2*n
    matrix = lil_matrix((rows, variables))
    for t in range(n):
        matrix[t, t] = 1
        matrix[t, n+t] = -1
        matrix[t, 2*n+t] = 1
        matrix[t, 3*n+t] = -1
        matrix[n+t, 4*n+t] = -1
        matrix[n+t, 4*n+t+1] = 1
        matrix[n+t, n+t] = -ETA
        matrix[n+t, 2*n+t] = 1/ETA
        if adjustment:
            matrix[2*n+t, t] = 1
            matrix[2*n+t, 5*n+1+t] = -1
            matrix[2*n+t, 6*n+1+t] = 1
    cost = np.zeros(variables)
    if adjustment:
        cost[5*n+1:6*n+1] = 1.5*price
        cost[6*n+1:] = .5*price
    else:
        cost[:n] = price
    bounds = ([(0, None)]*n + [(0, POWER*DT)]*(2*n) +
              [(0, None)]*n + [(LO, HI)]*(n+1))
    bounds[4*n] = (initial, initial)
    bounds[5*n] = (6000., 6000.)
    if adjustment:
        bounds += [(0, None)]*(2*n)
    rhs = np.r_[net, np.zeros(n), original if adjustment else []]
    return matrix.tocsr(), rhs, cost, bounds


def solve_independent(problem):
    matrix, rhs, cost, bounds = problem
    primary = linprog(cost, A_eq=matrix, b_eq=rhs, bounds=bounds, method='highs')
    assert primary.success, primary.message
    # Infer n directly from the fixed initial-state bound rather than stage status.
    initial_indices = [i for i, (lower, upper) in enumerate(bounds)
                       if upper is not None and lower == upper]
    n = initial_indices[0]//4
    secondary_cost = np.zeros(len(cost))
    secondary_cost[n:3*n] = 1
    secondary = linprog(secondary_cost, A_eq=matrix, b_eq=rhs, bounds=bounds,
                        A_ub=csr_matrix(cost.reshape(1, -1)),
                        b_ub=[float(primary.fun)+1e-7], method='highs')
    assert secondary.success, secondary.message
    return primary, secondary


def bound_residual(x, bounds):
    violations = []
    for value, (lower, upper) in zip(x, bounds):
        if lower is not None:
            violations.append(lower-value)
        if upper is not None:
            violations.append(value-upper)
    return max(0., max(violations))


def main():
    started = datetime.now(timezone.utc).isoformat()
    protected = [ROOT/'data/inputs.npz'] + sorted((ROOT/'results').rglob('*.npz'))
    protected += sorted((ROOT/'results').rglob('*.xlsx'))
    before = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    a = read_inputs(ROOT/'data/inputs.npz')
    official = dict(np.load(ROOT/'results/q3.npz'))
    pred, err = forecasts(a)[True]
    q = Q_MAIN['q3']
    prediction_error = price_error = margin_error = mutation_error = 0.
    ratio_min, ratio_max = np.inf, -np.inf
    clipped_low = clipped_high = negative_values = history_pools = 0
    minimum_margin, maximum_margin = np.inf, -np.inf
    count = 0
    checked_dates = {key: [] for key in DATES}
    independent_err = np.full_like(err, np.nan)

    for day in range(365):
        for issue in range(4):
            k = issue*36
            net, prices, raw, ratio = reference_prediction(a, day, issue)
            actual_net = (a['load'][day, k:]-a['pv'][day, k:])*DT
            independent_err[day, issue, k:] = actual_net-net
            observed, observed_prices = predict(a, day, issue, True)
            prediction_error = max(prediction_error, float(np.abs(net-observed).max()))
            price_error = max(price_error, float(np.abs(prices-observed_prices).max()))
            assert np.allclose(net, pred[day, issue, k:], rtol=0, atol=1e-9)
            ratio_min, ratio_max = min(ratio_min, ratio), max(ratio_max, ratio)
            clipped_low += int(raw < .7)
            clipped_high += int(raw > 1.3)
            # Mutate all unavailable actual data and not-yet-issued forecasts.
            altered = {key: values.copy() for key, values in a.items()}
            altered['load'][day, k:] += 20000.
            altered['pv'][day, k:] += 10000.
            altered['load'][day+1:] += 17000.
            altered['pv'][day+1:] += 13000.
            altered['price'][day:] += 9.
            altered['forecast'][day, issue+1:] += 9000.
            altered['forecast'][day+1:] += 9000.
            changed = predict(altered, day, issue, True)
            mutation_error = max(mutation_error,
                                 *(float(np.abs(x-y).max())
                                   for x, y in zip((observed, observed_prices), changed)))
            count += 1

    safety_mutation_error = 0.
    for day in range(365):
        for issue in range(4):
            expected, sizes = reference_margin(independent_err, day, issue, q)
            actual = safety(err, day, issue, q)
            margin_error = max(margin_error, float(np.abs(expected-actual).max()))
            minimum_margin = min(minimum_margin, float(actual.min()))
            maximum_margin = max(maximum_margin, float(actual.max()))
            negative_values += int((actual < 0).sum())
            history_pools += len(sizes)
            changed_errors = err.copy()
            changed_errors[day:] = 1e8
            mutated = safety(changed_errors, day, issue, q)
            safety_mutation_error = max(safety_mutation_error,
                                        float(np.abs(actual-mutated).max()))
            # Every pooled target belongs to an earlier complete day.
            if sizes:
                latest_history_target = day*144-1
                current_release = day*144+issue*36
                assert latest_history_target < current_release

    assert count == 1460
    assert max(prediction_error, price_error, margin_error) < 1e-9
    assert mutation_error == 0 and safety_mutation_error == 0
    assert negative_values > 0, '经验余量应保留负值，下截断并非当前算法。'

    lp_rows = []
    for day_string in DATES:
        day = (date.fromisoformat(day_string)-date(2025, 1, 1)).days
        for issue in range(4):
            k = issue*36
            net = pred[day, issue, k:] + safety(err, day, issue, q)
            price = a['base'][k:, 0]
            initial = float(official['s'][day, k])
            original = official['g'][day, k:] if issue else None
            problem = independent_problem(net, price, initial, original)
            matrix, rhs, cost, bounds = problem
            original_matrix, original_count = structure(len(net), ETA, bool(issue))
            matrix_delta = matrix-original_matrix
            matrix_error = float(np.abs(matrix_delta.data).max()) if matrix_delta.nnz else 0.
            assert matrix.shape[1] == original_count and matrix_error == 0
            primary, secondary = solve_independent(problem)
            repeated = plan(net, price, initial, original=original)
            primary_error = abs(float(primary.fun)-repeated['primary'])
            objective_error = abs(float(cost@secondary.x)-repeated['objective'])
            equality_error = float(np.abs(matrix@secondary.x-rhs).max())
            bounds_error = float(bound_residual(secondary.x, bounds))
            n = len(net)
            dual_report = None
            if issue:
                shadow = primary.eqlin.marginals[:n]
                increase = primary.x[5*n+1:6*n+1]
                lower_violation = float(max(0., -shadow.min()))
                upper_violation = float(max(0., np.max(shadow-1.5*price)))
                complementary = float(np.max(np.abs(increase*(1.5*price-shadow))))
                assert max(lower_violation, upper_violation, complementary) < TOL
                dual_report = {'影子价最小值': float(shadow.min()),
                               '影子价最大值': float(shadow.max()),
                               '非负下界违反量': lower_violation,
                               '1_5倍价格上界违反量': upper_violation,
                               '增购与影子价互补残差': complementary,
                               '来源': '主费用LP的eqlin.marginals，非第二阶段对偶',
                               '核验通过': True}
            horizon_end = min(144, k+36)
            saved = official['a'][day, k:horizon_end]
            active_error = float(np.abs(secondary.x[:len(saved)]-saved).max())
            # Saved contracts after the next update are different decisions.
            # Verify only the current committed interval, with the future free.
            fixed = list(bounds)
            for t, value in enumerate(saved):
                fixed[t] = (float(value), float(value))
            if issue == 0:
                for t, value in enumerate(official['g'][day]):
                    fixed[t] = (float(value), float(value))
            fixed_solution = linprog(cost, A_eq=matrix, b_eq=rhs,
                                     bounds=fixed, method='highs')
            assert fixed_solution.success, (day_string, issue, fixed_solution.message)
            fixed_gap = float(fixed_solution.fun-primary.fun)
            assert max(primary_error, objective_error, equality_error, bounds_error) < TOL
            assert abs(fixed_gap) < TOL, (day_string, issue, fixed_gap)
            row = {'日期': day_string, '发布时刻_h': issue*6,
                   '剩余区间数': n, '变量数': matrix.shape[1], '等式约束数': matrix.shape[0],
                   '实际初始储量_kWh': initial, '主目标_元': float(primary.fun),
                   '独立矩阵最大差': matrix_error, '主目标重算差_元': primary_error,
                   '第二阶段目标重算差_元': objective_error,
                   '平衡等式残差': equality_error, '变量边界残差': bounds_error,
                   '已保存活动合同最大差_kWh': active_error,
                   '固定已保存合同区间后的主目标差_元': fixed_gap,
                   '主费用LP影子价核验': dual_report,
                   '核验通过': True}
            lp_rows.append(row)
            checked_dates[day_string].append(issue*6)

    reachability = []
    for k in (0, 36, 72, 108):
        n = 144-k
        hardest_lower = max(LO, HI-n*POWER*DT/ETA)
        hardest_upper = min(HI, LO+n*ETA*POWER*DT)
        assert hardest_lower <= 6000 <= hardest_upper
        reachability.append({'发布时刻_h': k//6, '剩余区间数': n,
                             '全允许初态下最严可达下界_kWh': hardest_lower,
                             '全允许初态下最严可达上界_kWh': hardest_upper,
                             '目标储量_kWh': 6000., '核验通过': True})

    boundary_cases = []
    # e = [epsilon - (a - predicted_n) - min(P*Delta, eta*(s-LO))]_+.
    for state in (LO, 6000., HI):
        available = min(POWER*DT, ETA*(state-LO))
        for deficit in (-10., 0., available, available+1., available+1000.):
            predicted = 500.
            contract = 600.
            epsilon = deficit+contract-predicted
            actual_net = predicted+epsilon
            reference_e = max(0., epsilon-(contract-predicted)-available)
            execution = step(contract, actual_net, state)
            assert abs(execution[2]-reference_e) < 1e-10
            boundary_cases.append({'储量_kWh': state, '净缺口_kWh': deficit,
                                   '可用放电量_kWh': available,
                                   '紧急购电_kWh': reference_e, '核验通过': True})

    price = np.tile(a['base'][:, 0], (365, 1))
    physical = check(official, a['load'], a['pv'], price)
    assert physical['pass'], physical
    after = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    assert after == before, '核验不得改变正式数据、工作簿或保存结果。'
    report = {
        '运行时间_UTC': started,
        '范围': '数学表述与现有正式算法的一致性，不是新策略性能实验。',
        '公式口径': {'分位数': q, '分位数算法': '线性插值',
                   '历史池': '此前最多28天，同发布时刻、同两小时块；前7天余量取零；排除第0天误差。',
                   '余量允许负值': True,
                   '优化目标': '确定性代理LP的合同调整费，不含情景紧急购电目标。',
                   '实际执行': '仅当前更新区间合同生效，充放电按实际观测重新执行。'},
        '全部发布点': {'数量': count, '预测最大差_kWh': prediction_error,
                    '价格预测最大差': price_error, '余量公式最大差_kWh': margin_error,
                    '未来信息扰动后预测最大差': mutation_error,
                    '未来误差扰动后余量最大差': safety_mutation_error,
                    '负载修正比例范围': [ratio_min, ratio_max],
                    '比例下截断次数': clipped_low, '比例上截断次数': clipped_high,
                    '误差池数量': history_pools, '负余量区间数量': negative_values,
                    '余量范围_kWh': [minimum_margin, maximum_margin], '核验通过': True},
        '四指定日期发布时刻': checked_dates,
        '独立LP核验': lp_rows,
        '多解处理': '既核对同一矩阵与两阶段目标，也固定保存合同的当前执行区间重解，检验其目标最优性。零点另固定全部原计划。',
        '终端可达性': reachability,
        '可达性限定': '购电和未利用电量无上限的本题假设；不等于稳定性或全年最优。',
        '问题二缺口与储能边界手算': boundary_cases,
        '全年实际执行独立核验': physical,
        '受保护文件': {'数量': len(before), '全部保持原哈希': True,
                    '输入哈希': before['data/inputs.npz'],
                    '正式问题三结果哈希': before['results/q3.npz']},
        '代码哈希': sha256(Path(__file__).resolve()),
        '全部通过': True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'report': str(OUTPUT), 'publication_points': count,
                      'independent_LPs': len(lp_rows), 'all_passed': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
