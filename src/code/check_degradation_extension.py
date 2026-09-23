# -*- coding: utf-8 -*-
"""复算15组问题一扩展，生成独立附件明细，不覆盖任何正式结果。"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

sys.dont_write_bytecode = True
from degradation_study import (ROOT, B, DT, ETA, MINUTES, RHOS,
                               independent_cell_model, solve_model)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = [ROOT / "data/inputs.npz", ROOT / "code/degradation_study.py",
             ROOT / "results/degradation_study.json", ROOT / "results/degradation_study.npz"]
    paths += [ROOT / f"results/{name}.npz" for name in ("q1", "q2", "q3", "q4_2", "q4_3")]
    paths += sorted((ROOT / "results").glob("result*.xlsx"))
    paths += sorted((ROOT / "output/提交结果").glob("result*.xlsx"))
    before = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    original = json.loads((ROOT / "results/degradation_study.json").read_text(encoding="utf-8"))
    saved_arrays = np.load(ROOT / "results/degradation_study.npz")
    base = np.load(ROOT / "data/inputs.npz")["base"]
    net = (base[:, 1] - base[:, 2]) * DT
    price = base[:, 0]
    records, saved_checks = [], []
    metric_difference = 0.
    metric_keys = ("grid_electricity_kwh", "purchase_cost_yuan", "cell_throughput_kwh",
                   "efc", "degradation_cost_yuan", "total_objective_yuan", "primary_objective_yuan")
    for minutes in MINUTES:
        for rho in RHOS:
            index = len(records)
            rec, _ = solve_model(net, price, float(rho), int(minutes // 10))
            old = original["records"][index]
            assert old["control_minutes"] == minutes and old["rho_yuan_per_cell_kwh"] == rho
            for key in metric_keys:
                difference = abs(rec[key] - old[key])
                metric_difference = max(metric_difference, difference)
                assert difference < 1e-6, (index, key, difference)
            assert rec["mode_switches"] == old["mode_switches"]
            assert rec["direct_charge_discharge_switches"] == old["direct_charge_discharge_switches"]
            g, c, d, w, s = [saved_arrays[key][index] for key in ("g", "c", "d", "w", "s")]
            throughput = float(ETA * c.sum() + d.sum() / ETA)
            checks = {"balance_kwh": float(np.max(np.abs(g + d - c - w - net))),
                      "state_kwh": float(np.max(np.abs(np.diff(s) - ETA * c + d / ETA))),
                      "purchase_cost_yuan": abs(float(price @ g) - rec["purchase_cost_yuan"]),
                      "cell_throughput_kwh": abs(throughput - rec["cell_throughput_kwh"]),
                      "efc": abs(throughput / (2 * B) - rec["efc"]),
                      "total_objective_yuan": abs(float(price @ g) + float(rho) * throughput - rec["total_objective_yuan"])}
            assert max(checks.values()) < 1e-6
            saved_checks.append({"kappa": float(rho), "control_minutes": int(minutes), "residuals": checks})
            records.append(rec)
    independent = []
    for index in (0, 8, 14):
        rec = records[index]
        value = independent_cell_model(net, price, rec["rho_yuan_per_cell_kwh"], rec["control_minutes"] // 10)
        difference = abs(value - rec["primary_objective_yuan"])
        assert difference < 1e-6
        independent.append({"kappa": rec["rho_yuan_per_cell_kwh"],
                            "control_minutes": rec["control_minutes"],
                            "independent_cell_ipm_objective_yuan": value,
                            "objective_difference_yuan": difference})
    objectives = np.array([r["total_objective_yuan"] for r in records]).reshape(3, 5)
    throughput = np.array([r["cell_throughput_kwh"] for r in records]).reshape(3, 5)
    assert np.all(np.diff(objectives, axis=1) >= -1e-5)
    assert np.all(np.diff(objectives, axis=0) >= -1e-5)
    assert np.all(np.diff(throughput, axis=1) <= 1e-4)
    formal_q1 = np.load(ROOT / "results/q1.npz")
    formal_cost_difference = abs(records[0]["purchase_cost_yuan"] - float(price @ formal_q1["g"]))
    assert formal_cost_difference < 1e-6
    report = {"status": "VERIFIED", "scope": "问题一确定性单日运行扩展，15组假设吞吐成本与动作保持尺度",
              "formal_case": {"kappa_yuan_per_cell_kwh": 0., "control_minutes": 10,
                              "purchase_cost_yuan": records[0]["purchase_cost_yuan"]},
              "definitions": {"cell_throughput": "sum(eta*c+d/eta)", "efc": "cell_throughput/(2*12000)",
                              "total_objective": "purchase_cost+kappa*cell_throughput",
                              "mode_switches": "相邻十分钟区间在充电、空闲、放电三状态间的变化次数"},
              "limitations": ["循环成本系数是假设参数，未标定真实设备寿命",
                              "总目标含假设循环成本，不替代题目规定的购电费用",
                              "负载、光伏、购电和结算均保持十分钟，只有充放电动作按块固定",
                              "状态切换次数不是求解时间，也不等于等效循环数"],
              "records": records, "saved_npz_checks": saved_checks,
              "max_existing_metric_difference": metric_difference,
              "independent_cross_checks": independent,
              "formal_q1_cost_difference_yuan": formal_cost_difference,
              "monotonicity_checks": {"objective_nondecreasing_in_kappa": True,
                                      "objective_nondecreasing_with_control_block": True,
                                      "throughput_nonincreasing_in_kappa": True},
              "source_sha256": before, "script_sha256": digest(Path(__file__))}
    table_lines = [r"% 由 code/check_degradation_extension.py 生成；15组已复算。",
                   r"\begin{table}[H]\centering\small",
                   r"\caption{循环成本与动作保持尺度的单日对照}\label{tab:degradation-detail}",
                   r"\setlength{\tabcolsep}{3.0pt}",
                   r"\begin{tabular}{rrrrrrr}\toprule",
                   r"$\kappa$ & 动作保持/min & 购电费/元 & 循环成本/元 & 总目标/元 & EFC/次 & 切换/次 \\\midrule"]
    for i, rec in enumerate(records):
        table_lines.append(f"{rec['rho_yuan_per_cell_kwh']:.2f} & {rec['control_minutes']} & "
                           f"{rec['purchase_cost_yuan']:.2f} & {rec['degradation_cost_yuan']:.2f} & "
                           f"{rec['total_objective_yuan']:.2f} & {rec['efc']:.4f} & {rec['mode_switches']} " + r"\\")
    table_lines += [r"\bottomrule\end{tabular}",
                    r"\par\smallskip\footnotesize $\kappa$的单位为元/电芯吞吐kWh，EFC为等效完整循环数。正式结果取$\kappa=0$、动作保持10分钟，只计购电费。负载、光伏、购电与结算均保持10分钟，循环成本为未标定寿命的运行情景假设。切换次数统计充电、空闲与放电状态的相邻变化，不表示求解时间。",
                    r"\end{table}"]
    table_path = ROOT / "tables/degradation_detail.tex"
    table_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    after = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    assert before == after
    report["source_files_unchanged"] = True
    report["table_sha256"] = digest(table_path)
    output = ROOT / "results/model_refinement/degradation_recheck.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    margin_path = ROOT / "results/model_refinement/forecast_margin.json"
    margin = json.loads(margin_path.read_text(encoding="utf-8"))
    audit = {"status": "VERIFIED", "scope": "问题一15组运行扩展复算及问题二三正式余量单区间覆盖诊断",
             "q1": {"report": str(output.relative_to(ROOT)), "record_count": len(records),
                    "max_existing_metric_difference": metric_difference,
                    "independent_cross_checks": independent, "formal_q1_cost_difference_yuan": formal_cost_difference,
                    "source_files_unchanged": True, "formal_case": report["formal_case"]},
             "forecast_margin": {"report": str(margin_path.relative_to(ROOT)),
                                 "checks": margin["checks"],
                                 "overall": {name: case["overall"] for name, case in margin["cases"].items()},
                                 "interpretation": "单区间事后覆盖，不是跨期同时安全概率，不调整正式参数"},
             "output_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (output, margin_path, table_path)},
             "protected_source_sha256": before}
    audit_path = ROOT / "audit/建模数学增强/问题一二核验.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit["q1"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
