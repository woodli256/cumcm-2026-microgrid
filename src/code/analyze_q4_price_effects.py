# -*- coding: utf-8 -*-
"""独立复核已保存的价格对照与加权余量结果，不重新运行全年调度。

输出采用相同实现电价、相同测试期及同一电芯侧库存参考价值。
原正式结果、工作簿、图和附录均不覆盖。
"""
from pathlib import Path
from datetime import date, timedelta
from itertools import combinations
import hashlib
import json

import numpy as np
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/q4_price_effects"
AUDIT = ROOT / "audit/归因纠正与章节调整"
DT, ETA, LO, HI, POWER = 1 / 6, 0.9, 1200.0, 10800.0, 5000.0
START = 31
TOL = 1e-6
FILES = {
    "q2": "results/q2.npz",
    "q3": "results/q3.npz",
    "q4_2": "results/q4_2.npz",
    "q4_3": "results/q4_3.npz",
    "weighted_q4_2": "results/revision/weighted_q4_2.npz",
    "weighted_q4_3": "results/revision/weighted_q4_3.npz",
}
Q = {"q2": 0.8, "q3": 0.65, "q4_2": 0.8, "q4_3": 0.65,
     "weighted_q4_2": 0.8, "weighted_q4_3": 0.65,
     "uniform_empirical_q4_2": 0.8, "uniform_empirical_q4_3": 0.65}
MONTH = np.array([(date(2025, 1, 1) + timedelta(days=i)).month for i in range(365)])


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def cost_components(o, price):
    """从最终动作独立恢复合同调整量并重算全部四类费用，单位元。"""
    increase = np.maximum(o["a"] - o["g"], 0)
    decrease = np.maximum(o["g"] - o["a"], 0)
    return np.column_stack([
        np.sum(price * o["g"], axis=1),
        np.sum(5 * price * o["e"], axis=1),
        np.sum(1.5 * price * increase, axis=1),
        np.sum(0.5 * price * decrease, axis=1),
    ])


def validate(o, a, source_price):
    assert np.array_equal(o["days"], np.arange(365))
    assert all(np.all(np.isfinite(v)) for v in o.values())
    net = (a["load"] - a["pv"]) * DT
    residuals = {
        "balance_kwh": float(np.max(np.abs(o["a"] + o["d"] + o["e"] - net - o["c"] - o["w"]))),
        "state_kwh": float(np.max(np.abs(np.diff(o["s"], axis=1) - ETA * o["c"] + o["d"] / ETA))),
        "crossday_kwh": float(np.max(np.abs(o["s"][1:, 0] - o["s"][:-1, -1]))),
        "charge_discharge_overlap_kwh": float(np.max(np.minimum(o["c"], o["d"]))),
        "up_kwh": float(np.max(np.abs(o["up"] - np.maximum(o["a"] - o["g"], 0)))),
        "down_kwh": float(np.max(np.abs(o["down"] - np.maximum(o["g"] - o["a"], 0)))),
        "stored_cost_yuan": float(np.max(np.abs(cost_components(o, source_price) - o["cost"]))),
        "initial_state_kwh": abs(float(o["s"][0, 0]) - 6000),
        "state_bounds_kwh": max(0., LO - float(o["s"].min()), float(o["s"].max()) - HI),
        "power_bounds_kwh": max(0., float(o["c"].max()) - POWER * DT, float(o["d"].max()) - POWER * DT),
        "nonnegative_kwh": max(0., -min(float(o[k].min()) for k in ["g", "a", "c", "d", "e", "w", "up", "down"])),
    }
    assert max(residuals.values()) < TOL, residuals
    return {"status": "VERIFIED", "scope": "保存轨迹的全年物理约束与原结算费用独立复算", "tolerance": TOL, **residuals}


def summarize(o, costs, keep, value):
    ix = np.flatnonzero(keep)
    c = costs[ix].sum(axis=0)
    initial, terminal = float(o["s"][ix[0], 0]), float(o["s"][ix[-1], -1])
    usage = initial - terminal
    return {
        "days": len(ix), "cost_components_yuan": c.tolist(), "total_cost_yuan": float(c.sum()),
        "initial_soc_kwh": initial, "terminal_soc_kwh": terminal, "net_inventory_use_kwh": usage,
        "inventory_adjustment_yuan": value * usage,
        "inventory_adjusted_cost_yuan": float(c.sum()) + value * usage,
        "emergency_kwh": float(o["e"][ix].sum()), "unused_kwh": float(o["w"][ix].sum()),
        "planned_kwh": float(o["g"][ix].sum()), "increase_kwh": float(o["up"][ix].sum()),
        "decrease_kwh": float(o["down"][ix].sum()),
    }


def compare(left, right, summaries, monthly, value, scope):
    """正节省额表示 right 比 left 费用更低。"""
    a, b = summaries[left], summaries[right]
    saving = a["total_cost_yuan"] - b["total_cost_yuan"]
    slope = a["net_inventory_use_kwh"] - b["net_inventory_use_kwh"]
    threshold = -saving / slope if abs(slope) > 1e-12 else None
    months = []
    for month in range(2, 13):
        ma, mb = monthly[left][str(month)], monthly[right][str(month)]
        months.append({
            "month": month,
            "saving_yuan": ma["total_cost_yuan"] - mb["total_cost_yuan"],
            "inventory_adjusted_saving_yuan": ma["inventory_adjusted_cost_yuan"] - mb["inventory_adjusted_cost_yuan"],
        })
    assert abs(sum(m["saving_yuan"] for m in months) - saving) < TOL
    adjusted = saving + value * slope
    assert abs(sum(m["inventory_adjusted_saving_yuan"] for m in months) - adjusted) < TOL
    return {
        "reference": left, "candidate": right, "scope": scope,
        "reference_q": Q[left], "candidate_q": Q[right], "same_q": Q[left] == Q[right],
        "saving_yuan": saving, "saving_percent": 100 * saving / a["total_cost_yuan"],
        "component_saving_yuan": (np.array(a["cost_components_yuan"]) - b["cost_components_yuan"]).tolist(),
        "inventory_saving_slope_kwh": slope, "inventory_adjusted_saving_yuan": adjusted,
        "inventory_ranking_threshold_yuan_per_cell_kwh": threshold,
        "ranking_changes_within_0_to_1_6": threshold is not None and 0 <= threshold <= 1.6,
        "saving_at_value_0_yuan": saving, "saving_at_value_1_6_yuan": saving + 1.6 * slope,
        "monthly": months, "saving_month_count": sum(m["saving_yuan"] > TOL for m in months),
        "higher_cost_month_count": sum(m["saving_yuan"] < -TOL for m in months),
    }


def check_scaling_and_ranking():
    """同排序不同动作的两期例子，同时校验全目标正缩放的不变性。"""
    # g0, g1, c, d；两期净需求分别为0和1 kWh，电芯初末均为0。
    mat = np.array([[1, 0, -1, 0], [0, 1, 0, 1], [0, 0, ETA, -1 / ETA]])
    def solve(price):
        s = linprog([*price, 0, 0], A_eq=mat, b_eq=[0, 1, 0],
                    bounds=[(0, None), (0, None), (0, 2), (0, 2)], method="highs")
        assert s.success and np.max(np.abs(mat @ s.x - [0, 1, 0])) < 1e-9
        return s.x
    low_spread, high_spread = solve([0.5, 0.6]), solve([0.5, 0.8])
    assert low_spread[2] < 1e-9 and high_spread[2] > 1
    checks = {str(k): float(np.max(np.abs(solve(np.array([0.5, 0.8]) * k) - high_spread))) for k in [0.7, 1.2]}
    assert max(checks.values()) < 1e-9
    return {
        "status": "VERIFIED", "scope": "固定可行域的精确LP最优解集合与两期可复算例子",
        "theorem": "若可行域不变且目标所有系数同乘正数k，则argmin c^T x与argmin k c^T x相同。最优解不唯一时，任意求解器未必输出同一条轨迹。",
        "application_conditions": "净需求、初末储量、容量、功率、合同和信息规则均不变。所有价格相关费用项同乘k，不能另留未缩放的寿命费或终端估值。",
        "implementation_limit": "正式plan的二阶段容差为绝对1e-7元，因此数值容差和多解选择可能产生细微动作差，不能声称浮点输出逐项严格相同。",
        "arbitrage_condition": "普通购电跨期转移需要 eta^2 * p_high > p_low；仅知道高低排序不足以判断节费。",
        "eta_squared": ETA ** 2,
        "same_order_counterexample": [
            {"prices": [0.5, 0.6], "charge_kwh": float(low_spread[2]), "action": low_spread.tolist()},
            {"prices": [0.5, 0.8], "charge_kwh": float(high_spread[2]), "action": high_spread.tolist()},
        ],
        "positive_scaling_max_action_difference": checks,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    matched_paths = {f"uniform_empirical_q4_{i}": f"results/q4_price_effects/uniform_empirical_q4_{i}.npz" for i in [2, 3]}
    matched_available = all((ROOT / p).exists() for p in matched_paths.values())
    if matched_available:
        FILES.update(matched_paths)
    a = load_npz(ROOT / "data/inputs.npz")
    originals = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))
    value = float(np.mean(a["base"][:, 0]) * ETA)
    keep = np.arange(365) >= START
    summaries, monthly, checks = {}, {}, {}
    for name, relpath in FILES.items():
        o = load_npz(ROOT / relpath)
        source_price = a["base"][:, 0] if name in ["q2", "q3"] else a["price"]
        checks[name] = validate(o, a, source_price)
        if "q" in o:
            assert np.array_equal(np.unique(o["q"]), [Q[name]])
            checks[name]["q_source"] = "npz内逐日q数组"
        else:
            assert originals["training"][name]["main_q"] == Q[name]
            checks[name]["q_source"] = "results/summary.json中正式运行元数据，原npz未保存q数组"
        checks[name]["q"] = Q[name]
        costs = cost_components(o, a["price"])
        summaries[name] = summarize(o, costs, keep, value)
        monthly[name] = {str(m): summarize(o, costs, MONTH == m, value) for m in range(2, 13)}
        assert abs(sum(v["total_cost_yuan"] for v in monthly[name].values()) - summaries[name]["total_cost_yuan"]) < TOL
    scopes = [
        ("q2", "q4_2", "不调整规则内只更换规划价格，q均为0.8，比较含状态自然演化的完整在线规则"),
        ("q3", "q4_3", "日内调整规则内只更换规划价格，q均为0.65，比较含状态自然演化的完整在线规则"),
        ("q2", "q3", "固定价生成动作下两类完整策略之差，光伏信息、调整权限及q同时不同，不能识别纯合同作用"),
        ("q4_2", "q4_3", "历史价格预测下两类完整策略之差，光伏信息、调整权限及q同时不同，不能识别纯合同作用"),
        ("q4_2", "weighted_q4_2", "不调整策略内同q=0.8比较普通余量和价格加权历史误差余量，回顾性设计"),
        ("q4_3", "weighted_q4_3", "日内调整策略内同q=0.65比较普通余量和价格加权历史误差余量，回顾性设计"),
    ]
    if matched_available:
        for i, q in [(2, 0.8), (3, 0.65)]:
            scopes.append((f"uniform_empirical_q4_{i}", f"weighted_q4_{i}",
                           f"匹配控制：同q={q}、同经验逆分布和样本池，仅比较单位权重与历史实际价格权重；新重跑等权对照，候选用已保存结果"))
            scopes.append((f"q4_{i}", f"uniform_empirical_q4_{i}",
                           f"插值定义对照：同q={q}与单位权重，仅由原线性插值分位数换为等权经验逆分布"))
    comparisons = {b + "_vs_" + aa: compare(aa, b, summaries, monthly, value, scope) for aa, b, scope in scopes}
    ranking = sorted(summaries, key=lambda n: summaries[n]["inventory_adjusted_cost_yuan"])
    all_thresholds = []
    for aa, b in combinations(summaries, 2):
        c = compare(aa, b, summaries, monthly, value, "仅检查所有保存策略的费用排序，不作因果解释")
        all_thresholds.append({k: c[k] for k in ["reference", "candidate", "inventory_ranking_threshold_yuan_per_cell_kwh", "ranking_changes_within_0_to_1_6"]})
    ranking_unchanged = not any(c["ranking_changes_within_0_to_1_6"] for c in all_thresholds)
    report = {
        "status": "VERIFIED", "scope": "本脚本独立复核保存结果；run_q4_matched_controls.py另外连续重跑两份匹配等权对照" if matched_available else "保存结果独立复核，不是重新运行全年优化",
        "evaluation": {"days": 334, "intervals": 48096, "months": "2月至12月", "calendar": "365天平年，以2025日历标记月份",
                       "settlement": "全部按附件4同一实际电价；原计划全额支付，增购1.5倍，减购另收0.5倍，紧急5倍",
                       "initialization": "全部原回测从1月1日6000 kWh连续运行，2月1日测试首态自然不同",
                       "inference_limit": "动作不使用未来实现电价；库存修正只是统一记账敏感性，不等于重新优化统一首末状态，也不构成跨年统计显著性证明"},
        "component_order": ["plan", "emergency", "increase", "decrease"],
        "inventory": {"formula": "C_adj(v)=C+v*(S_start-S_end)", "value_yuan_per_cell_kwh": value,
                      "reference": "附件1固定电价144时段均价乘单向放电效率0.9", "sensitivity_interval": [0, 1.6],
                      "interpretation": "电芯侧库存折成可输出电量的统一参考价值，不是实际售电收入"},
        "totals": summaries, "monthly_totals": monthly, "comparisons": comparisons,
        "validation": checks, "ranking_at_reference_value": ranking, "all_pair_inventory_thresholds": all_thresholds,
        "all_rankings_unchanged_within_inventory_interval": ranking_unchanged,
        "matched_controls_available": matched_available,
        "weighted_margin": {
            "formula": "b=inf{z: sum(p_i*1[error_i<=z])/sum(p_i)>=q}",
            "sample_rule": "每个发布时刻分别使用此前最多28个完整日期，按同一2小时时段的12个十分钟误差与对应真实电价配对。样本起点max(1,day-28)，终点day排除当前日；先剔除非有限误差，再按误差排序累积价格权重。day<7沿用普通余量。",
            "q": {"weighted_q4_2": 0.8, "weighted_q4_3": 0.65},
            "source": "code/revision_experiments.py::weighted_margin与variant，保存npz的q数组与正式summary交叉核对",
            "limitation": "普通余量采用np.quantile线性插值，加权余量采用离散经验逆分布，因此直接与正式策略比较还改变了插值约定。现新增同经验逆分布的等权对照用于识别权重替换；经验策略有储能及合同补救，单时段0.8解析式不证明其全年最优。" if matched_available else "普通余量采用np.quantile线性插值，加权余量采用离散经验逆分布，因此实现还改变了分位数插值约定；改善不能全部归为价格权重，若需严格识别权重本身，应增加等权经验逆分布对照。经验策略有储能及合同补救，单时段0.8解析式不证明其全年最优。",
            "stale_text": "正文所称加权实验采用另一分位数与当前保存结果不符；旧experiments.json的weighted设计文字仍写original q0.65，应以npz实际q及当前调用代码为准。",
        },
        "lp_price_scaling": check_scaling_and_ranking(),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ["data/inputs.npz", "results/summary.json", "code/solve.py", "code/revision_experiments.py", *FILES.values()]},
    }
    (OUT / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_note = "另从1月1日连续重跑两份等权经验逆分布对照。" if matched_available else "没有重新运行全年调度。"
    lines = ["# 问题四深化建议", "", f"本次独立复核保存轨迹，{run_note}没有修改正式工作簿、图或附录。全部{len(summaries)}份轨迹的物理约束与原结算费用通过独立复算，容差为1e-6。", "", "## 同一实际电价的四格对照", "", "固定价仅用于生成前两类策略动作。四格费用均按附件4实际电价结算，因此不会把结算价格差误当成价格预测收益。", "", "| 规划价格与策略 | q | 费用/元 | 统一库存修正费用/元 | 测试初储量/kWh | 年末储量/kWh |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    labels = {"q2": "固定价生成动作，不调整", "q3": "固定价生成动作，日内调整", "q4_2": "历史价格预测，不调整", "q4_3": "历史价格预测，日内调整", "weighted_q4_2": "价格加权余量，不调整", "weighted_q4_3": "价格加权余量，日内调整", "uniform_empirical_q4_2": "等权经验逆分布，不调整", "uniform_empirical_q4_3": "等权经验逆分布，日内调整"}
    for n, s in summaries.items():
        lines.append(f"| {labels[n]} | {Q[n]} | {s['total_cost_yuan']:.2f} | {s['inventory_adjusted_cost_yuan']:.2f} | {s['initial_soc_kwh']:.2f} | {s['terminal_soc_kwh']:.2f} |")
    rank_note = "全部方案费用排序均不变" if ranking_unchanged else "部分方案存在排序切换，具体阈值见JSON"
    lines += ["", f"库存参考价值取附件1平均电价乘0.9，得到 {value:.7f} 元/电芯侧kWh。修正费用为 C+v(S初-S末)。这只是统一记账，没有强行改变原动作或初末状态。在 v 从0到1.6的区间，{rank_note}。", "", "| 成对比较 | 候选方案节省/元 | 库存修正后节省/元 | 节省月份数 | 库存价值排序阈值 |", "| --- | ---: | ---: | ---: | ---: |"]
    for c in comparisons.values():
        threshold = c['inventory_ranking_threshold_yuan_per_cell_kwh']
        threshold_text = "无有限阈值" if threshold is None else f"{threshold:.4f}"
        lines.append(f"| {c['candidate']} 相对 {c['reference']} | {c['saving_yuan']:.2f} | {c['inventory_adjusted_saving_yuan']:.2f} | {c['saving_month_count']}/11 | {threshold_text} |")
    lines += ["", "负节省额表示增费。无调整两项价格比较均使用q=0.8，日内调整两项均使用q=0.65。不同合同列还同时改变光伏信息和风险参数，只能称两类完整策略之差。不要写成严格识别了合同调整的独立因果贡献。", "", "## 月度分布", "", "| 月份 | 价格预测，不调整节省/元 | 价格预测，日内调整节省/元 | 加权余量，不调整节省/元 | 加权余量，日内调整节省/元 |", "| --- | ---: | ---: | ---: | ---: |"]
    keys = ["q4_2_vs_q2", "q4_3_vs_q3", "weighted_q4_2_vs_q4_2", "weighted_q4_3_vs_q4_3"]
    for i, m in enumerate(range(2, 13)):
        lines.append("| " + " | ".join([str(m)] + [f"{comparisons[k]['monthly'][i]['saving_yuan']:.2f}" for k in keys]) + " |")
    lines += ["", "## 可直接写入论文的说明", "", "制定计划取决于时段间相对比价及设备约束，单有电价高低排序并不足够。若低价时充入1 kWh，之后最多回送0.81 kWh，因此普通购电转移需要满足0.81倍高价大于低价。固定可行域下，将目标所有系数同时乘以同一正数不会改变精确线性规划的最优解集合，所以近似固定日内形状主要限制了每日整体价格水平预测对动作的作用。残差仍会改变相对比价与约束边际，具体收益必须由同价结算对照决定。", "", "在相同实际电价和各自相同风险分位数下，历史价格中位数使不调整策略费用下降16631.40元，却使日内调整策略费用增加1183.77元。统一库存记账后分别节省16641.88元和增加1172.00元；参考价值从0变化到1.6元/电芯侧kWh时，两项排序均不变。这表明该价格预测规则的本年收益依赖完整策略条件，不支持仅凭较小预测误差推断调度一定改善。", "", "价格加权余量使用此前最多28个完整日期的同期误差和实际价格配对，按价格权重构造经验分位数。保存结果中的不调整与日内调整分位数分别为0.8和0.65，与对应正式策略一致。独立复算得费用14742562.58元与14410537.84元，相对原策略分别节省8676.39元和增加7866.48元；库存修正后分别节省8663.22元和增加7885.51元。费用变化说明加权余量与可用补救方式共同决定结果，不能把价格加权作为普遍改进。", "", "加权实现使用离散经验逆分布，普通实现使用线性插值分位数。因此这组实验检验的是当前加权余量规则的整体替换，若要进一步识别价格权重本身，应补充等权经验逆分布对照。该方法在查看全年结果后设计，虽然每一步只使用历史信息，仍属于回顾性改进检验，需要新年份验证。", "", "## 证据边界", "", "价格正缩放结论针对精确LP的最优解集合。最优解不唯一、绝对数值容差或另加未同比缩放的寿命成本与终端价值时，不能声称程序逐项动作必然不变。脚本已用两期例子验证，同样的低高价排序可以对应不充电或充电；正尺度0.7与1.2则在该例中保持动作。", "", "复现命令为 `python code/analyze_q4_price_effects.py`。完整费用、月度、库存阈值、物理残差与源文件哈希保存在 `results/q4_price_effects/analysis.json`。"]
    if matched_available:
        lines = [line.replace("因此这组实验检验的是当前加权余量规则的整体替换，若要进一步识别价格权重本身，应补充等权经验逆分布对照。", "因此直接相对正式策略的比较检验当前加权余量规则的整体替换。为单独识别价格权重，已补充同经验逆分布的等权对照，结果见下一节。") for line in lines]
        lines += ["", "## 经验逆分布匹配控制", "", "为隔离价格权重，另行使用同一加权variant入口，从1月1日6000 kWh连续重跑两类等权经验逆分布方案。唯一区别为全部历史权重取1。样本池、缺失处理、前7日回退、分位数及其他模型规则保持一致。抽样检查确认，新余量函数与原weighted_margin输入全1价格所得余量完全相同。", "", "| 策略 | 同定义等权费用/元 | 价格加权费用/元 | 仅换权重节省/元 | 库存修正后节省/元 | 节省月份数 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for i in [2, 3]:
            uniform = f"uniform_empirical_q4_{i}"
            weighted = f"weighted_q4_{i}"
            c = comparisons[weighted + "_vs_" + uniform]
            lines.append(f"| {'不调整' if i == 2 else '日内调整'} | {summaries[uniform]['total_cost_yuan']:.2f} | {summaries[weighted]['total_cost_yuan']:.2f} | {c['saving_yuan']:.2f} | {c['inventory_adjusted_saving_yuan']:.2f} | {c['saving_month_count']}/11 |")
            direct = comparisons[weighted + f"_vs_q4_{i}"]
            interpolation = comparisons[uniform + f"_vs_q4_{i}"]
            assert abs(direct['saving_yuan'] - interpolation['saving_yuan'] - c['saving_yuan']) < TOL
        lines += ["", "此新增控制消除了分位数插值定义的差别。其差额可解释为在该历史回测和既定经验逆分布规则中替换价格权重的影响，仍不保证其他年份、风险参数或调整机制下均有相同收益。", "", "| 月份 | 同定义价格加权，不调整节省/元 | 同定义价格加权，日内调整节省/元 |", "| --- | ---: | ---: |"]
        for j, month in enumerate(range(2, 13)):
            vals = [comparisons[f"weighted_q4_{i}_vs_uniform_empirical_q4_{i}"]["monthly"][j]["saving_yuan"] for i in [2, 3]]
            lines.append(f"| {month} | {vals[0]:.2f} | {vals[1]:.2f} |")
        lines += ["", "新增运行的命令为 `python code/run_q4_matched_controls.py`，随后用原分析脚本独立复核。运行元数据及全1权重检查保存在 `results/q4_price_effects/matched_control_run.json`。"]
    (AUDIT / "问题四深化建议.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"inventory_value": value, "totals": summaries, "comparisons": {k: {kk: vv for kk, vv in v.items() if kk != "monthly"} for k, v in comparisons.items()}, "status": "VERIFIED"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
