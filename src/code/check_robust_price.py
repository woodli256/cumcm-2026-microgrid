# -*- coding: utf-8 -*-
"""预算价格扩展独立检查，小例先行，全年轨迹、选择规则和信息边界后验。"""
from itertools import combinations
from pathlib import Path
import argparse
import json
import sys

sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

import robust_price as rp
from solve import ROOT, ETA, DT, LO, HI, POWER, Q_MAIN, read_inputs, forecasts, predict, safety, plan, step
from validate import check


def independent_budget_lp(coefficients, gamma):
    """直接在价格偏差预算集上最大化；不使用鲁棒求解器的对偶约束。"""
    n = len(coefficients)
    result = linprog(-np.asarray(coefficients), A_ub=np.ones((1, n)), b_ub=[gamma],
                     bounds=[(0., 1.)]*n, method="highs-ipm")
    assert result.success
    return float(-result.fun)


def independent_dual_lp(coefficients, gamma):
    n = len(coefficients)
    constraints = np.c_[-np.ones(n), -np.eye(n)]
    result = linprog(np.r_[gamma, np.ones(n)], A_ub=constraints,
                     b_ub=-np.asarray(coefficients), bounds=[(0., None)]*(n+1), method="highs-ipm")
    assert result.success
    return float(result.fun)


def budget_vertices(n, gamma):
    """单调非负暴露下只需预算耗尽的顶点，限小例枚举。"""
    if gamma == 0:
        return [np.zeros(n)]
    if gamma >= n:
        return [np.ones(n)]
    integral = int(np.floor(gamma))
    fraction = gamma-integral
    points = []
    for ids in combinations(range(n), integral):
        point = np.zeros(n)
        point[list(ids)] = 1.
        if fraction < 1e-12:
            points.append(point)
        else:
            for remaining in set(range(n))-set(ids):
                candidate = point.copy()
                candidate[remaining] = fraction
                points.append(candidate)
    return points


def independent_cellside_vertex_plan(net, price, radius, initial, rho, original=None):
    """电芯侧物理变量和全部小例价格顶点重建 min-max LP。"""
    n = len(net)
    adjust = original is not None
    core = 5*n+1+(2*n if adjust else 0)
    size = core+1
    eq = lil_matrix((2*n+(n if adjust else 0), size))
    rhs = np.r_[net, np.zeros(n), original if adjust else []]
    for t in range(n):
        eq[t, t] = 1.
        eq[t, n+t] = -1/ETA
        eq[t, 2*n+t] = ETA
        eq[t, 3*n+t] = -1.
        eq[n+t, 4*n+t+1] = 1.
        eq[n+t, 4*n+t] = -1.
        eq[n+t, n+t] = -1.
        eq[n+t, 2*n+t] = 1.
        if adjust:
            eq[2*n+t, t] = 1.
            eq[2*n+t, 5*n+1+t] = -1.
            eq[2*n+t, 6*n+1+t] = 1.
    bounds = ([(0., None)]*n + [(0., POWER*DT*ETA)]*n +
              [(0., POWER*DT/ETA)]*n + [(0., None)]*n + [(LO, HI)]*(n+1))
    bounds[4*n] = (initial, initial)
    bounds[5*n] = (6000., 6000.)
    if adjust:
        bounds += [(0., None)]*(2*n)
    bounds.append((None, None))
    vertices = budget_vertices(n, rho*n)
    ub = lil_matrix((len(vertices), size))
    brhs = np.zeros(len(vertices))
    for row, vertex in enumerate(vertices):
        scenario_price = price+radius*vertex
        if adjust:
            ub[row, 5*n+1:6*n+1] = 1.5*scenario_price
            ub[row, 6*n+1:7*n+1] = .5*scenario_price
            brhs[row] = -float(scenario_price @ original)
        else:
            ub[row, :n] = scenario_price
        ub[row, -1] = -1.
    objective = np.zeros(size)
    objective[-1] = 1.
    result = linprog(objective, A_eq=eq.tocsr(), b_eq=rhs, A_ub=ub.tocsr(), b_ub=brhs,
                     bounds=bounds, method="highs-ipm")
    assert result.success, result.message
    return {"objective": float(result.fun), "vertex_count": len(vertices),
            "equality_residual": float(np.max(abs(eq @ result.x-rhs))),
            "inequality_violation": float(max(0., np.max(ub @ result.x-brhs)))}


def core_checks():
    rng = np.random.default_rng(20260913)
    maximum = 0.
    count = 0
    for n in (1, 2, 3, 7, 20):
        for rho in (0., .1, .25, .5, 1.):
            for values in (rng.uniform(0, 3000, n), np.zeros(n), np.full(n, 14.)):
                direct = independent_budget_lp(values, rho*n)
                dual = independent_dual_lp(values, rho*n)
                ordered = rp.support_sorted(values, rho*n)
                maximum = max(maximum, abs(direct-dual), abs(direct-ordered))
                count += 1
    assert maximum < 1e-7
    cases = []
    for adjusted in (False, True):
        for rho in rp.RHOS:
            net = np.array([-200., 1000., 500.])
            price = np.array([.3, 1.1, .6])
            radius = np.array([.4, .15, .25])
            original = np.array([500., 650., 700.]) if adjusted else None
            direct = rp.robust_plan(net, price, radius, 6000., rho, original)
            independent = independent_cellside_vertex_plan(net, price, radius, 6000., rho, original)
            gap = abs(direct["robust_objective"]-independent["objective"])
            assert gap < 1e-6, (adjusted, rho, gap)
            regression = None
            if rho == 0:
                reference = plan(net, price, 6000., original)
                regression = max(float(np.max(abs(direct[key]-reference[key]))) for key in ("g", "c", "d", "w", "s"))
                assert regression == 0.
            cases.append({"adjusted": adjusted, "rho": rho, "objective": direct["robust_objective"],
                          "independent_vertex_lp": independent, "objective_gap": gap,
                          "zero_budget_max_action_gap": regression})
    a = read_inputs(ROOT/"data/inputs.npz")
    center, radius, errors = rp.price_information(a)
    radius_reconstruction = 0.
    for day in (0, 6, 7, 31, 172, 364):
        expected = (np.zeros(144) if day < 7 else
                    np.quantile(np.abs(errors[max(0, day-28):day]), .8, axis=0))
        radius_reconstruction = max(radius_reconstruction, float(np.max(abs(expected-radius[day]))))
    assert radius_reconstruction == 0.
    mutation = []
    for day, issue in ((0, 0), (6, 0), (7, 0), (31, 1), (172, 2), (364, 3)):
        changed = {key: value.copy() for key, value in a.items()}
        k = 36*issue
        changed["price"][day:] += 4.321
        changed["load"][day, k:] += 12345.
        changed["pv"][day, k:] += 54321.
        changed["load"][day+1:] += 45678.
        changed["pv"][day+1:] += 76543.
        changed["forecast"][day, issue+1:] += 1234.
        changed["forecast"][day+1:] += 4321.
        after_center, after_radius, _ = rp.price_information(changed)
        current_net, current_price = predict(a, day, issue, True)
        after_net, after_price = predict(changed, day, issue, True)
        differences = {"price_center": float(np.max(abs(center[day]-after_center[day]))),
                       "price_radius": float(np.max(abs(radius[day]-after_radius[day]))),
                       "point_net_prediction": float(np.max(abs(current_net-after_net))),
                       "point_price_prediction": float(np.max(abs(current_price-after_price)))}
        assert max(differences.values()) == 0., differences
        nominal = np.full(144-k, 500.) if issue else None
        before = rp.robust_plan(current_net, current_price, radius[day, k:], 6000., .25, nominal)
        after = rp.robust_plan(after_net, after_price, after_radius[day, k:], 6000., .25, nominal)
        differences["contract_action"] = float(np.max(abs(before["g"]-after["g"])))
        assert differences["contract_action"] == 0.
        mutation.append({"day_index": day, "issue": issue, "differences": differences})
    return {"status": "PASS", "budget_support_examples": count,
            "budget_primal_dual_sorted_max_gap": maximum, "independent_small_lps": cases,
            "radius_reconstruction_max_gap": radius_reconstruction, "future_truth_mutations": mutation}


def trajectory_checks():
    report = json.loads((rp.OUT/"experiment.json").read_text(encoding="utf-8"))
    a = read_inputs(ROOT/"data/inputs.npz")
    value = float(ETA*np.mean(a["base"][:, 0]))
    output = {}
    for name in ("q4_2", "q4_3"):
        records = report["strategies"][name]["candidates"]
        assert len(records) == 4 and [r["rho"] for r in records] == list(rp.RHOS)
        candidates = []
        original = dict(np.load(ROOT/f"results/{name}.npz"))
        for record in records:
            path = rp.OUT/record["file"]
            assert rp.file_hash(path) == record["sha256"]
            saved = dict(np.load(path))
            assert np.array_equal(saved["days"], np.arange(365))
            assert np.array_equal(saved["q"], np.full(365, Q_MAIN[name]))
            assert np.array_equal(saved["rho"], np.full(365, record["rho"]))
            validation = check(saved, a["load"], a["pv"], a["price"])
            assert validation["pass"]
            assert all(np.all(np.isfinite(item)) for item in saved.values())
            # 独立检查实际充放电确实逐段使用原即时执行，无额外留电规则。
            state = saved["s"][:, :-1]
            balance = saved["a"]-(a["load"]-a["pv"])*DT
            expected_c = np.where(balance >= 0, np.minimum(np.minimum(balance, POWER*DT),
                                  np.maximum(0., (HI-state)/ETA)), 0.)
            expected_d = np.where(balance < 0, np.minimum(np.minimum(-balance, POWER*DT),
                                  np.maximum(0., (state-LO)*ETA)), 0.)
            execution_gap = max(float(np.max(abs(saved["c"]-expected_c))),
                                float(np.max(abs(saved["d"]-expected_d))))
            assert execution_gap < 1e-6
            # 费用、尾部及库存统计直接从保存值重建，不依赖运行摘要函数。
            train_cost = float(saved["cost"][14:31].sum())
            days = saved["cost"][31:].sum(1)
            reconstructed = {"total_cost_yuan": float(days.sum()),
                             "emergency_kwh": float(saved["e"][31:].sum()),
                             "daily_cost_p95_yuan": float(np.percentile(days, 95)),
                             "worst_5_percent_daily_mean_yuan": float(np.sort(days)[-17:].mean()),
                             "initial_soc_kwh": float(saved["s"][31, 0]),
                             "terminal_soc_kwh": float(saved["s"][-1, -1]),
                             "inventory_adjusted_cost_yuan": float(days.sum()+value*(saved["s"][31, 0]-saved["s"][-1, -1]))}
            summary_gap = max(abs(record["test"][key]-v) for key, v in reconstructed.items())
            assert summary_gap < 1e-6
            assert abs(record["training"]["total_cost_yuan"]-train_cost) < 1e-6
            candidates.append((train_cost, record["rho"]))
            zero = None
            if record["rho"] == 0:
                zero = {key: float(np.max(abs(saved[key]-original[key])))
                        for key in ("g", "a", "c", "d", "e", "w", "up", "down", "s", "cost")}
                assert max(zero.values()) < 1e-6
            output[name+f"_rho_{record['rho']:g}"] = {"physics_and_cost": validation,
                       "execution_rule_gap": execution_gap, "summary_max_gap": summary_gap,
                       "zero_budget_regression": zero}
        low = min(cost for cost, rho in candidates)
        selected = min(rho for cost, rho in candidates if cost <= low+1e-6)
        assert report["strategies"][name]["selected_rho"] == selected
    unchanged = {path: rp.file_hash(ROOT/path) == sha for path, sha in report["source_sha256"].items()}
    assert all(unchanged.values()), unchanged
    return {"status": "PASS", "output_count": len(output), "outputs": output,
            "protected_source_and_formal_results_unchanged": unchanged,
            "selection_rule": "重新从八份NPZ中取1月15至31日费用，复核选中rho，不用测试期费用"}


def write_note():
    data = json.loads((rp.OUT/"experiment.json").read_text(encoding="utf-8"))
    lines = ["# 问题四预算价格鲁棒实验", "", "价格鲁棒规划作为独立扩展实验运行，结果用于正文推导与附录对照。原正式策略、已有图表数据和五份结果工作簿保持不变。", "",
             "## 模型与原框架的关系", "",
             "价格中心沿用过去七天同期中位数。区间半径由此前至多28个完整日期的同十分钟价格预测绝对残差的80%线性插值分位数给出。前7天半径为零。误差对应的历史预测也只使用各历史日之前的数据。", "",
             r"价格不确定集为 $p_t=\widehat p_t+\delta_t z_t$，$0\le z_t\le1$，$\sum_tz_t\le\Gamma$，其中 $\Gamma=\rho N$。因计费电量非负，采用对称区间时最坏价格也在上偏方向，因此此处直接给出其等价的上偏部分。", "",
             r"最坏价格费用为 $\widehat p^{\mathsf T}x+\max_{z}\sum_t\delta_tx_tz_t$。预算支持函数的对偶为 $\min_{\theta,r\ge0}\Gamma\theta+\sum_tr_t$，满足 $\theta+r_t\ge\delta_tx_t$。将其放入原LP即可获得精确线性形式。", "",
             r"零点的计费暴露是 $x_t=g_t$。日内的计费暴露是 $x_t=g_t+1.5u_t+0.5v_t$，并满足 $a_t-g_t=u_t-v_t$。固定原合同量在最坏价格算子内部不可删去。原预测余量已用于规划供需，本扩展没有另造规划紧急购电变量，因此不是对全部随机实际账单的联合最坏化。", "",
             "净需求预测、安全分位数、零点及6/12/18点的更新时刻、日末6000 kWh规划目标、实际即时执行和实现电价结算完全沿用正式模型。这项扩展同时保留供需误差余量并考虑价格不确定性，但未声称建立完整联合分布，也未给出样本外概率保证。", "",
             "## 参数与对照", "",
             "预算比例候选固定为0、0.10、0.25、1.00。每一候选均从1月1日6000 kWh开始连续运行，仅依据1月15至31日原结算总费用选参数。2月至12月的全部候选只作敏感性报告。由于本实验在已查看全年数据的论文修订中设计，仍属于回顾性检验。", "",
             "零预算直接调用原规划器。八组结果采用同一物理执行与计费规则，所有正式动作、费用和库存的零预算回归误差均小于1e-6。以原固定电价均值乘效率作为统一电芯侧库存价值，库存修正只是比较记账，不是实际售电收入。", "",
             "| 策略 | 预算比例 | 训练费用/元 | 测试费用/元 | 紧急购电/kWh | 日费用P95/元 | 最差17日日均费用/元 | 库存修正费用/元 |", 
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, label in (("q4_2", "不调整"), ("q4_3", "日内调整")):
        records = data["strategies"][name]["candidates"]
        for r in records:
            t = r["test"]
            lines.append(f"| {label} | {r['rho']:.2f} | {r['training']['total_cost_yuan']:.2f} | {t['total_cost_yuan']:.2f} | {t['emergency_kwh']:.2f} | {t['daily_cost_p95_yuan']:.2f} | {t['worst_5_percent_daily_mean_yuan']:.2f} | {t['inventory_adjusted_cost_yuan']:.2f} |")
    lines += ["", "## 可写入正文的结论", ""]
    for name, label in (("q4_2", "不调整"), ("q4_3", "日内调整")):
        records = data["strategies"][name]["candidates"]
        selected = data["strategies"][name]["selected_rho"]
        chosen = next(r for r in records if r["rho"] == selected)
        baseline = records[0]
        savings = baseline["test"]["total_cost_yuan"]-chosen["test"]["total_cost_yuan"]
        lines.append(f"{label}策略按1月费用选择的预算比例为{selected:.2f}，测试费用{chosen['test']['total_cost_yuan']:.2f}元，相对零预算节省{savings:.2f}元。" +
                     ("训练选择保留原策略，不能根据全年敏感性再改选参数。" if selected == 0 else "该选择来自训练期，后续正式替换仍需由主任务统一更新全部相关结果。"))
        lines.append("")
    lines += ["鲁棒目标在所设价格集合内控制最坏的规划计费暴露，不等于真实全年费用或实证尾部必然下降。应同时展示增加的备用成本、紧急购电与尾部费用，避免把模型名称当作改善证据。", "",
              "## 核验与复现", "",
              "小例用价格预算集上的直接最大化、独立对偶和顶端排序核对支持函数。另用电芯侧物理变量与全部小例价格顶点重建优化问题，核对包含固定原合同的日内最坏费用。全年逐段检查平衡、储量、功率、互斥、跨日连续、实际执行及账单。未来真值和后续预报扰动不影响当前价格中心、半径与决策。", "",
              "```bash", "python3 code/check_robust_price.py --core-only", "python3 code/robust_price.py --run", "python3 code/check_robust_price.py", "```", "",
              "完整模型代码在code/robust_price.py，独立核验在code/check_robust_price.py。八份轨迹、规划目标、实验汇总与核验结果都在results/robust_price/。", ""]
    rp.AUDIT.mkdir(parents=True, exist_ok=True)
    (rp.AUDIT/"问题四鲁棒实验说明.md").write_text("\n".join(lines), encoding="utf-8")


def write_detail_table():
    """只读取已核验报告生成独立附件明细，不重新求解全年模型。"""
    data = json.loads((rp.OUT/"experiment.json").read_text(encoding="utf-8"))
    assert data["status"] == "VERIFIED"
    assert data["independent_validation_sha256"] == rp.file_hash(rp.OUT/"validation.json")
    lines = [r"% 由 python3 code/check_robust_price.py --table-only 生成。",
             r"\begin{table}[H]\centering",
             r"\caption{预算价格鲁棒模型的顺序对照}\label{tab:robust-price-detail}",
             r"\fontsize{9.5}{12}\selectfont\setlength{\tabcolsep}{2.5pt}\renewcommand{\arraystretch}{1.18}",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrr@{}}\toprule",
             r"策略 & $\rho$ & \shortstack{1月评分\\/元} & \shortstack{测试费\\/万元} & \shortstack{紧急电量\\/kWh} & \shortstack{最差17日\\日均费/元} & \shortstack{库存修正费\\/万元} \\",
             r"\midrule"]
    rows = 0
    for name, label in (("q4_2", "问题四-2"), ("q4_3", "问题四-3")):
        item = data["strategies"][name]
        assert len(item["candidates"]) == 4
        assert item["selected_rho"] == 0.
        if rows:
            lines.append(r"\midrule")
        for record in item["candidates"]:
            assert record["sha256"] == rp.file_hash(rp.OUT/record["file"])
            test = record["test"]
            rho = f"{record['rho']:.2f}"
            if record["rho"] == item["selected_rho"]:
                rho = r"\textbf{" + rho + "}"
            cells = [label, rho, f"{record['training']['total_cost_yuan']:.2f}",
                     f"{test['total_cost_yuan']/10000:.2f}", f"{test['emergency_kwh']:.2f}",
                     f"{test['worst_5_percent_daily_mean_yuan']:.2f}",
                     f"{test['inventory_adjusted_cost_yuan']/10000:.2f}"]
            lines.append(" & ".join(cells)+r" \\")
            rows += 1
    assert rows == 8
    lines += [r"\bottomrule\end{tabular*}",
              r"\par\vspace{4pt}\begin{minipage}{\linewidth}\fontsize{9}{11.5}\selectfont\raggedright",
              r"1月评分为1月15至31日总费用，测试期为2月至12月。各方案均从1月1日6000 kWh连续运行，$\rho$仅按1月评分选择，加粗的零预算为两类策略的选择结果，$\rho>0$均为未采用的敏感性候选。",
              r"库存修正费按统一价值0.6895775元/电芯侧kWh计入测试首末储量差，仅作比较记账。",
              r"\end{minipage}", r"\end{table}", ""]
    target = ROOT/"tables/robust_price_detail.tex"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--table-only", action="store_true", help="只由已核验报告生成八行附件明细表，不重跑全年")
    args = parser.parse_args()
    if args.table_only:
        print(str(write_detail_table()))
        return
    rp.OUT.mkdir(parents=True, exist_ok=True)
    core = core_checks()
    if args.core_only:
        (rp.OUT/"core_checks.json").write_text(json.dumps(core, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"status": core["status"], "budget_examples": core["budget_support_examples"],
                          "independent_small_lps": len(core["independent_small_lps"])}, ensure_ascii=False))
        return
    report = {"status": "PASS", "core": core, "trajectories": trajectory_checks(),
              "checker_sha256": rp.file_hash(Path(__file__))}
    (rp.OUT/"validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    experiment_path = rp.OUT/"experiment.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    experiment["status"] = "VERIFIED"
    experiment["independent_validation_file"] = "validation.json"
    experiment["independent_validation_sha256"] = rp.file_hash(rp.OUT/"validation.json")
    experiment_path.write_text(json.dumps(experiment, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    write_note()
    print(json.dumps({"status": "PASS", "outputs": report["trajectories"]["output_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
