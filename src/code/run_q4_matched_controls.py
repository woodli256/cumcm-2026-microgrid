# -*- coding: utf-8 -*-
"""固定经验逆分布定义，重跑等权历史误差对照，不覆盖正式结果。

运行 python code/run_q4_matched_controls.py 后，再运行
python code/analyze_q4_price_effects.py 生成与价格加权结果的匹配比较。
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import json
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/q4_price_effects"


def uniform_empirical_margin(a, err, day, issue, q):
    """样本池、缺失处理和前7日回退与weighted_margin完全相同，仅全部权重为1。"""
    from solve import safety
    k = issue * 36
    result = safety(err, day, issue, q)
    if day < 7:
        return result
    for lo in range(k, 144, 12):
        values = err[max(1, day - 28):day, issue, lo:lo + 12].ravel()
        values = np.sort(values[np.isfinite(values)])
        # inf{z: empirical_cdf(z)>=q}。整数边界与价格权重实现相同，
        # 使用searchsorted而非线性插值；每个有效样本的权重固定为1。
        index = min(np.searchsorted(np.arange(1, len(values) + 1), q * len(values)), len(values) - 1)
        result[lo - k:lo - k + 12] = values[index]
    return result


def run_one(name, use, q):
    import revision_experiments as rev
    from solve import read_inputs, forecasts, totals
    started = time.monotonic()
    print(f"开始 {name}，q={q}", flush=True)
    a = read_inputs(ROOT / "data/inputs.npz")
    fc = forecasts(a)
    old_weighted_margin = rev.weighted_margin
    # 确认实现确实等于原加权函数的全部权重设为1，避免引入样本池变化。
    unit_prices = dict(a)
    unit_prices["price"] = np.ones_like(a["price"])
    margin_error = 0.0
    for day in [0, 6, 7, 31, 90, 180, 270, 364]:
        for issue in range(4 if use else 1):
            direct = uniform_empirical_margin(a, fc[use][1], day, issue, q)
            oracle = old_weighted_margin(unit_prices, fc[use][1], day, issue, q)
            margin_error = max(margin_error, float(np.max(np.abs(direct - oracle))))
    assert margin_error == 0.0
    rev.weighted_margin = uniform_empirical_margin
    try:
        # 与原加权实验同入口，保留全部模型、实际执行、预测与初末条件。
        o = rev.variant(a, fc, dynamic=True, weighted=True, use=use, q=q)
    finally:
        rev.weighted_margin = old_weighted_margin
    target = OUT / (name + ".npz")
    np.savez_compressed(target, **o)
    result = {"name": name, "q": q, "use_forecast_and_adjustment": use,
              "elapsed_seconds": time.monotonic() - started,
              "uniform_margin_vs_original_function_with_unit_weights_max_error_kwh": margin_error,
              "totals": totals(o), "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    print(f"完成 {name}，费用 {result['totals']['total_cost']:.2f} 元", flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [("uniform_empirical_q4_2", False, 0.8), ("uniform_empirical_q4_3", True, 0.65)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_one, *task) for task in tasks]
        results = [f.result() for f in futures]
    report = {"scope": "从1月1日6000 kWh连续重新运行两份等权经验逆分布对照，2月至12月评价",
              "change": "只把同一经验逆分布下的历史实际价格权重替换为单位权重",
              "fixed": "相同过去28个完整日期、同2h误差池、相同day0排除、相同前7日回退、相同q=.8/.65、同预测及执行规则",
              "weighted_candidates": "使用已保存的results/revision/weighted_q4_2.npz与weighted_q4_3.npz，并独立复核，不重跑候选",
              "results": results,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                  Path(__file__).resolve(), ROOT / "code/revision_experiments.py", ROOT / "code/solve.py", ROOT / "data/inputs.npz"]}}
    (OUT / "matched_control_run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
