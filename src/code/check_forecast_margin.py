# -*- coding: utf-8 -*-
"""按正式历史余量复核单区间覆盖率，不选参、不重跑调度、不改正式结果。

Q2 使用零点预测的全天 144 段；Q3 每次发布仅使用后续 36 段，
每个实际十分钟区间只计一次。这里的覆盖不是跨期同时安全概率，
也不是储能执行后免于紧急购电的概率。
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import sys

import numpy as np

sys.dont_write_bytecode = True
from solve import ROOT, Q_MAIN, forecasts, predict, read_inputs, safety

OUTPUT = ROOT / "results/model_refinement/forecast_margin.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(error, margin):
    error, margin = np.asarray(error), np.asarray(margin)
    assert error.shape == margin.shape and error.size
    assert np.isfinite(error).all() and np.isfinite(margin).all()
    covered = int(np.count_nonzero(error <= margin))
    count = int(error.size)
    return {"interval_count": count, "covered_count": covered,
            "coverage_fraction": covered / count,
            "coverage_percent": 100 * covered / count}


def independent_margin(err, day, issue, q):
    """排序后手工线性插值，独立核对 safety 的样本范围与分位数。"""
    start = issue * 36
    result = np.zeros(144 - start)
    if day < 7:
        return result
    for first in range(start, 144, 12):
        values = []
        for previous in range(max(1, day - 28), day):
            for target in range(first, first + 12):
                value = err[previous, issue, target]
                if np.isfinite(value):
                    values.append(float(value))
        if not values:
            continue
        values.sort()
        location = (len(values) - 1) * q
        lower = int(np.floor(location))
        upper = int(np.ceil(location))
        value = values[lower] + (location - lower) * (values[upper] - values[lower])
        result[first - start:first - start + 12] = value
    return result


def main():
    source_paths = [ROOT / "data/inputs.npz", ROOT / "code/solve.py",
                    ROOT / "results/forecasts.npz", ROOT / "results/q2.npz",
                    ROOT / "results/q3.npz"]
    before = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    inputs = read_inputs(ROOT / "data/inputs.npz")
    prepared = forecasts(inputs)
    saved = np.load(ROOT / "results/forecasts.npz")
    forecast_difference = {}
    for use, key in [(False, "p2"), (True, "p3")]:
        current = prepared[use][0]
        assert current.shape == saved[key].shape
        assert np.array_equal(np.isnan(current), np.isnan(saved[key]))
        valid = np.isfinite(current)
        forecast_difference[key] = float(np.max(np.abs(current[valid] - saved[key][valid])))
        assert forecast_difference[key] < 1e-10
    assert Q_MAIN["q2"] == .8 and Q_MAIN["q3"] == .65
    days = np.arange(31, 365)
    dates = [dt.date(2025, 1, 1) + dt.timedelta(days=int(day)) for day in days]
    months = np.array([date.month for date in dates])
    cases = {}
    independent_max = 0.
    independent_count = 0
    for name, use, issues, length in [("q2", False, [0], 144),
                                      ("q3", True, [0, 1, 2, 3], 36)]:
        q = Q_MAIN[name]
        _, error = prepared[use]
        issue_results, all_errors, all_margins = {}, [], []
        for issue in issues:
            start = issue * 36
            selected_error = error[days, issue, start:start + length]
            selected_margin = np.stack([safety(error, int(day), issue, q)[:length]
                                        for day in days])
            for day in (31, 171, 364):
                direct = independent_margin(error, day, issue, q)
                actual = safety(error, day, issue, q)
                independent_max = max(independent_max, float(np.max(np.abs(direct - actual))))
                independent_count += len(actual) // 12
            all_errors.append(selected_error)
            all_margins.append(selected_margin)
            block_results = []
            for first in range(0, length, 12):
                block_error = selected_error[:, first:first + 12]
                block_margin = selected_margin[:, first:first + 12]
                block_results.append({
                    "lead_start_minutes": first * 10,
                    "lead_end_minutes": (first + 12) * 10,
                    "interval_definition": "发布后该两小时内的十二个十分钟区间",
                    "overall": stats(block_error, block_margin),
                    "by_month": {f"2025-{month:02d}": stats(block_error[months == month],
                                                            block_margin[months == month])
                                 for month in range(2, 13)}})
            issue_results[f"{issue * 6:02d}:00"] = {
                "applied_target_clock_start": f"{issue * 6:02d}:00",
                "applied_target_clock_end": "24:00" if name == "q2" else f"{(issue + 1) * 6:02d}:00",
                "overall": stats(selected_error, selected_margin),
                "by_month": {f"2025-{month:02d}": stats(selected_error[months == month],
                                                        selected_margin[months == month])
                             for month in range(2, 13)},
                "by_two_hour_lead_block": block_results}
        joined_error = np.concatenate(all_errors, axis=1)
        joined_margin = np.concatenate(all_margins, axis=1)
        assert joined_error.shape == (334, 144)
        case = {"q": q, "actual_interval_counting": "每个实际十分钟区间恰好一次",
                "overall": stats(joined_error, joined_margin),
                "by_month": {f"2025-{month:02d}": stats(joined_error[months == month],
                                                        joined_margin[months == month])
                             for month in range(2, 13)},
                "by_issue": issue_results}
        # 两套独立分组的计数应恢复同一个年度分母与分子。
        for field in ("interval_count", "covered_count"):
            assert sum(row[field] for row in case["by_month"].values()) == case["overall"][field]
            assert sum(row["overall"][field] for row in issue_results.values()) == case["overall"][field]
        cases[name] = case
    assert independent_max < 1e-10
    mutation_checks = []
    for use, issues in [(False, [0]), (True, [0, 1, 2, 3])]:
        pred, err = prepared[use]
        for issue in issues:
            day = 171
            start = issue * 36
            modified = {key: value.copy() for key, value in inputs.items()}
            modified["load"][day, start:] += 20000
            modified["pv"][day, start:] += 10000
            modified["load"][day + 1:] += 17000
            modified["pv"][day + 1:] += 13000
            modified["price"][day:] += 9
            modified["forecast"][day, issue + 1:] += 9000
            modified["forecast"][day + 1:] += 9000
            future_err = err.copy()
            future_err[day:] += 12345
            q = .65 if use else .8
            pred_difference = float(np.max(np.abs(predict(modified, day, issue, use)[0]
                                                  - pred[day, issue, start:])))
            margin_difference = float(np.max(np.abs(safety(future_err, day, issue, q)
                                                    - safety(err, day, issue, q))))
            assert pred_difference == 0 and margin_difference == 0
            mutation_checks.append({"q": q, "issue_hour": issue * 6,
                                    "prediction_difference_kwh": pred_difference,
                                    "margin_difference_kwh": margin_difference})
    after = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    assert before == after
    report = {
        "status": "VERIFIED", "scope": "正式Q2/Q3历史余量的十分钟单区间事后覆盖诊断",
        "period": {"start": "2025-02-01", "end": "2025-12-31", "days": 334},
        "definition": "实际净需求减对应发布预测的误差 epsilon 不大于当时历史余量 b 的区间比例",
        "history_rule": "过去最多28个完整日期，同发布时刻、同两小时块；排除1月1日启动误差；线性插值分位数",
        "selection": "Q2固定q=0.8，Q3固定q=0.65，本核验不重新选参、不改变任何正式调度",
        "limitations": ["逐区间比例不是全天或全年同时安全概率",
                        "误差被合同余量覆盖与储能执行后是否发生紧急购电不同",
                        "结果为同一年固定规则的事后诊断，不给出新年份概率保证",
                        "Q3仅计每次发布后实际采用的六小时，不重复统计更远期预测"],
        "cases": cases,
        "checks": {"saved_forecast_max_difference_kwh": forecast_difference,
                   "independent_linear_quantile_blocks": independent_count,
                   "independent_margin_max_difference_kwh": independent_max,
                   "future_information_mutations": mutation_checks,
                   "source_files_unchanged": before == after},
        "source_sha256": before,
        "script_sha256": digest(Path(__file__)),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: item["overall"] for name, item in cases.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
