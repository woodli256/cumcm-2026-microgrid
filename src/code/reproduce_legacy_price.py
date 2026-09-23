# -*- coding: utf-8 -*-
"""复核旧q=0.65不调整实验的分位数算法与价格权重贡献。

三组均从1月1日连续重新求解，全部输出限定在 results/q4_weight_attribution_legacy 目录。
运行 python code/reproduce_legacy_price.py。
"""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/q4_weight_attribution_legacy"
sys.path.insert(0, str(ROOT / "code"))
import revision_experiments as rev
from solve import read_inputs, forecasts, predict, safety
from run_q4_matched_controls import uniform_empirical_margin
from analyze_q4_price_effects import validate, cost_components, summarize, MONTH

Q = 0.65
TOL = 1e-6
NAMES = ["linear_quantile", "uniform_empirical", "price_weighted_empirical"]
USER_COSTS = dict(zip(NAMES, [15235929.10, 15231465.99, 15148474.57]))
USER_SAVINGS = {"definition": 4463.11, "weights": 82991.42}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison(reference, candidate, summaries, monthly, value):
    a, b = summaries[reference], summaries[candidate]
    saving = a["total_cost_yuan"] - b["total_cost_yuan"]
    inventory_slope = a["net_inventory_use_kwh"] - b["net_inventory_use_kwh"]
    threshold = None if abs(inventory_slope) < 1e-12 else -saving / inventory_slope
    month_rows = [{"month": m,
        "saving_yuan": monthly[reference][str(m)]["total_cost_yuan"] - monthly[candidate][str(m)]["total_cost_yuan"],
        "inventory_adjusted_saving_yuan": monthly[reference][str(m)]["inventory_adjusted_cost_yuan"] - monthly[candidate][str(m)]["inventory_adjusted_cost_yuan"]}
        for m in range(2, 13)]
    adjusted = saving + value * inventory_slope
    assert abs(sum(r["saving_yuan"] for r in month_rows) - saving) < TOL
    assert abs(sum(r["inventory_adjusted_saving_yuan"] for r in month_rows) - adjusted) < TOL
    return {"reference": reference, "candidate": candidate,
        "saving_yuan": saving, "saving_percent": 100 * saving / a["total_cost_yuan"],
        "inventory_adjusted_saving_yuan": adjusted, "inventory_saving_slope_kwh": inventory_slope,
        "ranking_threshold_yuan_per_cell_kwh": threshold,
        "ranking_changes_at_value_0_to_1_6": threshold is not None and 0 <= threshold <= 1.6,
        "saving_at_value_0_yuan": saving, "saving_at_value_1_6_yuan": saving + 1.6 * inventory_slope,
        "component_saving_yuan": (np.array(a["cost_components_yuan"]) - b["cost_components_yuan"]).tolist(),
        "monthly": month_rows, "saving_month_count": sum(r["saving_yuan"] > TOL for r in month_rows)}


def check_information_boundary(a, fc, original_weighted):
    """未调整策略零点预测与三种余量不得依赖本日及以后实际数据。"""
    checks = []
    for day in [31, 100, 300]:
        changed = {k: v.copy() for k, v in a.items()}
        changed["load"][day:] += 5000
        changed["pv"][day:] += 2000
        changed["price"][day:] *= 3
        changed["forecast"][day:] += 3000
        changed_fc = forecasts(changed)
        forecast_diff = max(float(np.max(np.abs(x-y))) for x, y in zip(predict(a, day, 0, False), predict(changed, day, 0, False)))
        for name, margin in [("linear_quantile", lambda aa, err, dd: safety(err, dd, 0, Q)),
                             ("uniform_empirical", lambda aa, err, dd: uniform_empirical_margin(aa, err, dd, 0, Q)),
                             ("price_weighted_empirical", lambda aa, err, dd: original_weighted(aa, err, dd, 0, Q))]:
            diff = float(np.max(np.abs(margin(a, fc[False][1], day)-margin(changed, changed_fc[False][1], day))))
            assert diff == 0 and forecast_diff == 0
            checks.append({"day_index": day, "method": name, "prediction_max_difference": forecast_diff, "margin_max_difference_kwh": diff})
    return checks


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [ROOT / "results" / (n + ".npz") for n in ["q2", "q3", "q4_2", "q4_3", "q4_2_q0.65"]]
    protected += [ROOT / f"results/revision/weighted_q4_{i}.npz" for i in [2, 3]]
    protected_before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    a = read_inputs(ROOT / "data/inputs.npz")
    assert a["load"].shape == a["pv"].shape == a["price"].shape == (365, 144)
    assert a["base"].shape == (144, 3) and a["forecast"].shape == (365, 4, 24)
    assert all(np.all(np.isfinite(v)) for v in a.values())
    assert np.all(a["price"] > 0)
    fc = forecasts(a)
    original_weighted = rev.weighted_margin
    value = float(np.mean(a["base"][:, 0]) * 0.9)
    summaries, monthly, checks, outputs = {}, {}, {}, {}
    unit_prices = dict(a, price=np.ones_like(a["price"]))
    uniform_agreement = max(float(np.max(np.abs(uniform_empirical_margin(a, fc[False][1], day, 0, Q) - original_weighted(unit_prices, fc[False][1], day, 0, Q)))) for day in [0, 6, 7, 31, 100, 300, 364])
    assert uniform_agreement == 0
    for name in NAMES:
        print(f"开始重跑 {name}，不调整，q={Q}", flush=True)
        try:
            rev.weighted_margin = uniform_empirical_margin if name == "uniform_empirical" else original_weighted
            o = rev.variant(a, fc, use=False, dynamic=True, weighted=name != "linear_quantile", q=Q)
        finally:
            rev.weighted_margin = original_weighted
        assert np.array_equal(np.unique(o["q"]), [Q])
        assert np.max(np.abs(o["a"]-o["g"])) == 0
        target = OUT / (name + ".npz")
        np.savez_compressed(target, **o)
        # 对写入后的文件重新读回验证，覆盖保存完整性。
        with np.load(target) as z:
            saved = {k:z[k] for k in z.files}
        checks[name] = validate(saved, a, a["price"])
        costs = cost_components(saved, a["price"])
        summaries[name] = summarize(saved, costs, np.arange(365) >= 31, value)
        monthly[name] = {str(m): summarize(saved, costs, MONTH == m, value) for m in range(2, 13)}
        outputs[name] = {"path": str(target.relative_to(ROOT)), "sha256": digest(target), "q_values": np.unique(saved["q"]).tolist()}
        print(f"完成 {name}，334天费用 {summaries[name]['total_cost_yuan']:.8f} 元", flush=True)
    steps = {
        "definition": comparison(NAMES[0], NAMES[1], summaries, monthly, value),
        "weights": comparison(NAMES[1], NAMES[2], summaries, monthly, value),
        "whole_replacement": comparison(NAMES[0], NAMES[2], summaries, monthly, value),
    }
    assert abs(steps["definition"]["saving_yuan"] + steps["weights"]["saving_yuan"] - steps["whole_replacement"]["saving_yuan"]) < TOL
    assert abs(steps["definition"]["inventory_adjusted_saving_yuan"] + steps["weights"]["inventory_adjusted_saving_yuan"] - steps["whole_replacement"]["inventory_adjusted_saving_yuan"]) < TOL
    with np.load(ROOT / "results/q4_2_q0.65.npz") as old:
        old_cost = float(old["cost"][31:].sum())
        with np.load(OUT / "linear_quantile.npz") as rerun:
            max_trajectory_diff = {k: float(np.max(np.abs(old[k] - rerun[k]))) for k in old.files}
    assert max(max_trajectory_diff.values()) < TOL
    match_costs = {name: {"user_yuan": USER_COSTS[name], "rerun_yuan": summaries[name]["total_cost_yuan"],
                         "rounded_cents_match": round(summaries[name]["total_cost_yuan"], 2) == USER_COSTS[name]} for name in NAMES}
    match_steps = {name: {"user_yuan": USER_SAVINGS[name], "rerun_yuan": steps[name]["saving_yuan"],
                         "rounded_cents_match": round(steps[name]["saving_yuan"], 2) == USER_SAVINGS[name]} for name in USER_SAVINGS}
    matched = all(x["rounded_cents_match"] for x in [*match_costs.values(), *match_steps.values()])
    info_checks = check_information_boundary(a, fc, original_weighted)
    protected_after = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    assert protected_before == protected_after
    report = {
        "status": "VERIFIED", "scope": "重新运行旧q=0.65不调整的三组全年轨迹，独立复核2月至12月费用及两步归因",
        "configuration": {"q": Q, "adjustment": False, "forecast": "历史负载与历史光伏；不使用已发布光伏预报", "price_prediction": "过去7个完整日期同期价格中位数，1月1日用附件1",
            "period": "从1月1日6000 kWh连续运行365天，1月预热，2月至12月334天计分",
            "shared_margin_pool": "同零点发布类型，过去至多28个完整日期，同两小时内12个时段，排除1月1日启动误差，前7天余量为0",
            "methods": {"linear_quantile": "numpy.quantile默认线性插值，样本等权", "uniform_empirical": "单位权重的离散经验逆分布", "price_weighted_empirical": "同一离散经验逆分布，权重改为与误差配对的历史实际电价"},
            "settlement": "附件4相同实现电价；计划1倍、紧急5倍；不调整所以增减购均为0"},
        "legacy_parameter_evidence": {"source": "results/q4_2_q0.65.npz", "stored_test_cost_yuan": old_cost, "rerun_trace_max_difference_by_key": max_trajectory_diff,
            "formal_current_q": 0.8, "interpretation": "用户三组旧费用对应q=0.65；本次不修改当前正式q=0.8或当前加权保存文件"},
        "input_validation": {"shape_pass": True, "all_finite": True, "all_actual_prices_positive": True},
        "inventory": {"value_yuan_per_cell_kwh": value, "formula": "C_adj=C+v*(S_start-S_end)", "source": "附件1平均电价乘单向放电效率0.9", "sensitivity_range": [0, 1.6], "limit": "统一记账敏感性，不是可实现售电收入或统一首末状态重优化"},
        "totals": summaries, "monthly_totals": monthly, "comparisons": steps,
        "user_values_match": matched, "user_cost_match": match_costs, "user_step_match": match_steps,
        "validation": checks, "uniform_function_with_unit_price_weights_max_difference_kwh": uniform_agreement,
        "future_data_perturbation_checks": info_checks,
        "limits": "两步差额沿预先明确的线性分位数、等权经验逆分布、价格加权经验逆分布顺序分解，限该q和该年顺序回测；全部替换收益不能归为价格权重，不能与正式q=0.8结果混用。",
        "protected_result_files_unchanged": True, "protected_source_hashes": protected_before,
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__).resolve(), ROOT/"data/inputs.npz", ROOT/"code/solve.py", ROOT/"code/revision_experiments.py", ROOT/"code/run_q4_matched_controls.py", ROOT/"code/analyze_q4_price_effects.py"]},
        "outputs": outputs, "elapsed_seconds": time.monotonic()-started,
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# 旧分位数0.65的权重归因复核", "", f"三组均在不调整策略下固定 q=0.65，从1月1日6000 kWh连续重新运行，以2月至12月334天计费。{'三组费用及两步差额均在分精度上匹配用户给出的数值。' if matched else '本次结果未全部匹配用户数值，差异已列入JSON。'}", "", "| 规则 | 测试费用/元 | 初储量/kWh | 末储量/kWh | 库存修正费用/元 |", "| --- | ---: | ---: | ---: | ---: |"]
    cn = {"linear_quantile":"原线性插值分位数", "uniform_empirical":"等权阶梯分位数", "price_weighted_empirical":"价格加权阶梯分位数"}
    for n in NAMES:
        s=summaries[n]
        lines.append(f"| {cn[n]} | {s['total_cost_yuan']:.2f} | {s['initial_soc_kwh']:.4f} | {s['terminal_soc_kwh']:.4f} | {s['inventory_adjusted_cost_yuan']:.2f} |")
    lines += ["", "| 比较 | 节省/元 | 库存修正后节省/元 |", "| --- | ---: | ---: |"]
    for label, key in [("只改分位数定义", "definition"), ("定义相同，只改价格权重", "weights"), ("从原算法整体替换", "whole_replacement")]:
        c=steps[key]
        lines.append(f"| {label} | {c['saving_yuan']:.2f} | {c['inventory_adjusted_saving_yuan']:.2f} |")
    lines += ["", f"库存参考价值为 {value:.7f} 元/电芯侧kWh，记账公式为费用加 v 乘初末储量差。", "", "此处是旧 q=0.65 的核验，当前正式不调整策略采用 q=0.8。两套结果不可混用。原加权对照同时改变了分位数定义和价格权重，因此不能把整体替换的节省额全部归因于价格权重。", "", "三份轨迹均通过电量平衡、储量递推、边界、功率、合同不调整及费用复算。新线性轨迹与原 q=0.65 保存轨迹逐项在1e-6容差内一致。等权实现与原加权函数输入全1权重的抽样结果相同；扰动本日及以后实际数据，不改变已检查日期的零点预测与三组余量。正式及现有加权结果文件哈希保持不变。", "", "运行命令为 `python code/reproduce_legacy_price.py`。全部原精度费用、两步差额、月度结果、库存阈值、约束残差与源文件哈希保存在同目录 `report.json`。"]
    (OUT / "复核说明.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"matched_user_values": matched,"totals":summaries,"comparisons":{k:{kk:vv for kk,vv in v.items() if kk!='monthly'} for k,v in steps.items()},"elapsed_seconds":report['elapsed_seconds']}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
