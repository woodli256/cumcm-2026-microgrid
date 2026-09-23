# -*- coding: utf-8 -*-
"""在原第四问 LP 上增加预算价格不确定性，正式结果不覆盖。

价格中心、净需求预测、风险分位数、结算及实际执行均沿用 solve.py。
半径为过去最多 28 个完整日期同十分钟区间的因果价格预测绝对残差
80% 线性插值经验分位数。前 7 天半径为 0。Gamma = rho * 剩余区间数。
所有候选从 1 月 1 日连续运行，选择只使用 1 月 15 至 31 日费用。
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, lil_matrix, vstack

from solve import (ROOT, DT, ETA, LO, HI, POWER, Q_MAIN, read_inputs, forecasts,
                   predict, safety, plan, step, structure)
from validate import check

OUT = ROOT / "results/robust_price"
AUDIT = ROOT / "audit/建模数学增强"
RHOS = (0., .1, .25, 1.)
RADIUS_Q = .8
SOLVE_TOL = 1e-6


def support_sorted(exposure, gamma):
    """非负暴露的预算集支持函数，用于逐个 LP 独立核算目标。"""
    ordered = np.sort(np.asarray(exposure, dtype=float))[::-1]
    if not len(ordered):
        return 0.
    gamma = float(np.clip(gamma, 0., len(ordered)))
    count = int(np.floor(gamma))
    return float(ordered[:count].sum() +
                 ((gamma - count) * ordered[count] if count < len(ordered) else 0.))


def price_information(a):
    """历史各日预测也逐日因果形成；当前及未来真值不进入当前半径。"""
    center = np.stack([predict(a, day, 0, False)[1] for day in range(365)])
    errors = a["price"] - center
    radius = np.zeros_like(center)
    for day in range(7, 365):
        radius[day] = np.quantile(np.abs(errors[max(0, day - 28):day]),
                                  RADIUS_Q, axis=0)
    return center, radius, errors


def robust_plan(net, price, radius, initial, rho, original=None, target=6000.,
                method="highs"):
    """精确预算对偶。日内暴露包含固定原合同，不能从最坏费用中删去。

    原变量 g,c,d,w,s,[u,v] 后加入 z,r[0:n]。日内的 g 变量代表新合同 a，
    原合同 original 是常数。风险暴露为 original + 1.5u + 0.5v。
    """
    net, price, radius = [np.asarray(v, dtype=float) for v in (net, price, radius)]
    n = len(net)
    assert price.shape == radius.shape == net.shape and n > 0
    assert np.all(radius >= 0) and np.all(price > 0) and 0 <= rho <= 1
    if rho == 0 or not np.any(radius):
        result = plan(net, price, initial, original=original, target=target, method=method)
        const = 0. if original is None else float(price @ original)
        result.update({"robust_objective": result["objective"] + const,
                       "primary_robust_objective": result["primary"] + const,
                       "robust_premium": 0., "gamma": rho * n,
                       "diagnostics": {"equality_residual": 0., "inequality_violation": 0.,
                                       "bound_violation": 0., "support_dual_gap": 0.,
                                       "secondary_primary_increase": 0.}})
        return result

    adjusted = original is not None
    core_eq, core_n = structure(n, ETA, adjusted)
    size = core_n + n + 1
    eq = hstack([core_eq, csr_matrix((core_eq.shape[0], n + 1))], format="csr")
    rhs = np.r_[net, np.zeros(n), original if adjusted else []]
    obj = np.zeros(size)
    if adjusted:
        obj[5*n+1:6*n+1] = 1.5 * price
        obj[6*n+1:7*n+1] = .5 * price
    else:
        obj[:n] = price
    gamma = rho * n
    obj[core_n] = gamma
    obj[core_n+1:] = 1.
    bounds = ([(0, None)] * n + [(0, POWER * DT)] * (2*n) +
              [(0, None)] * n + [(LO, HI)] * (n+1))
    bounds[4*n] = (initial, initial)
    bounds[5*n] = (target, target)
    if adjusted:
        bounds += [(0, None)] * (2*n)
    bounds += [(0, None)] * (n+1)
    ub = lil_matrix((n, size))
    ubrhs = np.zeros(n)
    for t in range(n):
        if adjusted:
            ub[t, 5*n+1+t] = 1.5 * radius[t]
            ub[t, 6*n+1+t] = .5 * radius[t]
            ubrhs[t] = -radius[t] * original[t]
        else:
            ub[t, t] = radius[t]
        ub[t, core_n] = -1.
        ub[t, core_n+1+t] = -1.
    ub = ub.tocsr()
    primary = linprog(obj, A_eq=eq, b_eq=rhs, A_ub=ub, b_ub=ubrhs,
                      bounds=bounds, method=method)
    if not primary.success:
        raise RuntimeError(primary.message)
    secondary_obj = np.zeros(size)
    secondary_obj[n:3*n] = 1.
    ub2 = vstack([ub, csr_matrix(obj.reshape(1, -1))], format="csr")
    secondary = linprog(secondary_obj, A_eq=eq, b_eq=rhs, A_ub=ub2,
                        b_ub=np.r_[ubrhs, primary.fun + 1e-7],
                        bounds=bounds, method=method)
    if not secondary.success:
        raise RuntimeError(secondary.message)
    x = secondary.x
    exposure = (np.asarray(original) + 1.5*x[5*n+1:6*n+1] + .5*x[6*n+1:7*n+1]
                if adjusted else x[:n])
    support = support_sorted(radius * exposure, gamma)
    dual_value = gamma*x[core_n] + x[core_n+1:].sum()
    bound_error = 0.
    for value, (lower, upper) in zip(x, bounds):
        if lower is not None:
            bound_error = max(bound_error, lower-value)
        if upper is not None:
            bound_error = max(bound_error, value-upper)
    diagnostics = {
        "equality_residual": float(np.max(np.abs(eq @ x - rhs))),
        "inequality_violation": float(max(0., np.max(ub @ x - ubrhs))),
        "bound_violation": float(bound_error),
        "support_dual_gap": float(abs(support - dual_value)),
        "secondary_primary_increase": float(max(0., obj @ x-primary.fun)),
    }
    assert max(diagnostics.values()) < SOLVE_TOL, diagnostics
    nominal = float(price @ exposure)
    const = float(price @ original) if adjusted else 0.
    return {"g": x[:n], "c": x[n:2*n], "d": x[2*n:3*n],
            "w": x[3*n:4*n], "s": x[4*n:5*n+1],
            "objective": float(obj @ x), "primary": float(primary.fun),
            "robust_objective": nominal + support,
            "primary_robust_objective": float(primary.fun)+const,
            "robust_premium": support, "gamma": gamma,
            "diagnostics": diagnostics}


def run(a, fc, centers, radii, name, rho, end=365):
    use = name == "q4_3"
    q = Q_MAIN[name]
    pred, errors = fc[use]
    out = {key: np.zeros((end, 144)) for key in ["g", "a", "c", "d", "e", "w", "up", "down"]}
    out["s"] = np.zeros((end, 145))
    out["cost"] = np.zeros((end, 4))
    out["days"] = np.arange(end)
    out["q"] = np.full(end, q)
    out["rho"] = np.full(end, rho)
    planned = np.full((end, 4, 3), np.nan)
    worst = {key: 0. for key in ["equality_residual", "inequality_violation", "bound_violation",
                                "support_dual_gap", "secondary_primary_increase"]}
    state = 6000.
    calls = 0
    for day in range(end):
        out["s"][day, 0] = state
        def current_plan(issue, original=None):
            nonlocal calls
            k = issue*36
            net = pred[day, issue, k:] + safety(errors, day, issue, q)
            result = robust_plan(net, centers[day, k:], radii[day, k:], state, rho, original)
            for key, value in result["diagnostics"].items():
                worst[key] = max(worst[key], value)
            planned[day, issue] = [result["robust_objective"], result["robust_premium"], result["gamma"]]
            calls += 1
            return result["g"]
        original = current_plan(0)
        out["g"][day] = original
        active = original.copy()
        for t in range(144):
            if use and t in (36, 72, 108):
                active[t:] = current_plan(t//36, original[t:])
            actual = (a["load"][day, t]-a["pv"][day, t])*DT
            charge, discharge, emergency, unused, state = step(active[t], actual, state)
            for key, value in [("a", active[t]), ("c", charge), ("d", discharge),
                               ("e", emergency), ("w", unused)]:
                out[key][day, t] = value
            out["s"][day, t+1] = state
        out["up"][day] = np.maximum(active-original, 0.)
        out["down"][day] = np.maximum(original-active, 0.)
        p = a["price"][day]
        out["cost"][day] = [p @ original, 5*p @ out["e"][day],
                              1.5*p @ out["up"][day], .5*p @ out["down"][day]]
    return out, {"lp_calls": calls, "max_residuals": worst, "planning_values": planned}


def summarize(out, value, first=31, end=None):
    stop = len(out["days"]) if end is None else end
    daily = out["cost"][first:stop].sum(axis=1)
    count = max(1, int(np.ceil(.05*len(daily))))
    start_state = float(out["s"][first, 0])
    end_state = float(out["s"][stop-1, -1])
    total = float(daily.sum())
    return {"days": len(daily), "total_cost_yuan": total,
            "cost_components_yuan": out["cost"][first:stop].sum(axis=0).tolist(),
            "emergency_kwh": float(out["e"][first:stop].sum()),
            "emergency_cost_yuan": float(out["cost"][first:stop, 1].sum()),
            "daily_cost_p95_yuan": float(np.quantile(daily, .95)),
            "tail_day_count": count,
            "worst_5_percent_daily_mean_yuan": float(np.sort(daily)[-count:].mean()),
            "initial_soc_kwh": start_state, "terminal_soc_kwh": end_state,
            "inventory_adjustment_yuan": value*(start_state-end_state),
            "inventory_adjusted_cost_yuan": total+value*(start_state-end_state)}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_all():
    OUT.mkdir(parents=True, exist_ok=True)
    a = read_inputs(ROOT/"data/inputs.npz")
    fc = forecasts(a)
    center, radius, errors = price_information(a)
    np.savez_compressed(OUT/"price_information.npz", center=center, radius=radius, errors=errors)
    value = float(ETA*np.mean(a["base"][:, 0]))
    source_paths = [Path(__file__), ROOT/"code/solve.py", ROOT/"code/validate.py",
                    ROOT/"data/inputs.npz", ROOT/"results/q4_2.npz", ROOT/"results/q4_3.npz"]
    report = {
        "scope": "只改变规划目标的预算价格鲁棒项，正式NPZ与工作簿不覆盖",
        "parameters": {"rho_candidates": list(RHOS), "radius_quantile": RADIUS_Q,
                       "past_complete_days": 28, "radius_zero_first_days": 7,
                       "gamma": "rho * remaining_ten_minute_intervals",
                       "radius_quantile_algorithm": "np.quantile 默认线性插值，沿日期轴、同十分钟区间",
                       "inventory_value_yuan_per_cell_kwh": value},
        "training": {"from": "2025-01-15", "to": "2025-01-31", "days": 17,
                     "selection_rule": "按原结算总费用最小选择；费用差不超过1e-6元时优先较小rho"},
        "evaluation": {"from": "2025-02-01", "to": "2025-12-31", "days": 334,
                       "designation": "回顾性方法修订中的顺序回测，全年敏感性不用于再选参数"},
        "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in source_paths},
        "strategies": {},
    }
    for name in ("q4_2", "q4_3"):
        records = []
        original = dict(np.load(ROOT/f"results/{name}.npz"))
        for rho in RHOS:
            started = time.monotonic()
            print(f"开始 {name}, rho={rho:g}", flush=True)
            out, diagnostics = run(a, fc, center, radius, name, rho)
            elapsed = time.monotonic()-started
            validation = check(out, a["load"], a["pv"], a["price"])
            assert validation["pass"], validation
            regression = None
            if rho == 0:
                regression = {key: float(np.max(np.abs(out[key]-original[key])))
                              for key in ("g", "a", "c", "d", "e", "w", "up", "down", "s", "cost")}
                assert max(regression.values()) < SOLVE_TOL, regression
            filename = f"{name}_rho_{rho:g}.npz"
            np.savez_compressed(OUT/filename, **out)
            np.savez_compressed(OUT/f"{name}_rho_{rho:g}_planning.npz",
                                planning_values=diagnostics.pop("planning_values"))
            record = {"rho": rho, "q": Q_MAIN[name], "runtime_seconds": elapsed,
                      "training": summarize(out, value, first=14, end=31),
                      "test": summarize(out, value), "validation": validation,
                      "planning_validation": diagnostics,
                      "zero_budget_regression": regression,
                      "file": filename, "sha256": file_hash(OUT/filename)}
            records.append(record)
            print(f"完成 {name}, rho={rho:g}, 训练 {record['training']['total_cost_yuan']:.2f}, "
                  f"测试 {record['test']['total_cost_yuan']:.2f}, {elapsed:.1f} 秒", flush=True)
            report["strategies"][name] = {"candidates": records}
            (OUT/"experiment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        best_value = min(rec["training"]["total_cost_yuan"] for rec in records)
        chosen = min((rec for rec in records if rec["training"]["total_cost_yuan"] <= best_value+1e-6),
                     key=lambda rec: rec["rho"])
        report["strategies"][name].update({"selected_rho": chosen["rho"],
                                            "selected_by": "January 15-31 cost only",
                                            "formal_result_changed": False})
        (OUT/"experiment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    # Verify all protected inputs and formal trajectories survived the experiment.
    assert all(file_hash(ROOT/path) == digest for path, digest in report["source_sha256"].items())
    report["status"] = "COMPLETE_PENDING_INDEPENDENT_CHECK"
    (OUT/"experiment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="执行八组全年敏感性，结果仅写入独立目录")
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return
    run_all()


if __name__ == "__main__":
    main()
