---
name: cumcm-results-audit
description: Recompute and audit this 2026 C-problem model's physical constraints, costs, and five exported result workbooks.
---

# C 题结果核验

以官方附件和空模板重新运行，不以仓库 `results/` 作为输入。使用 `--stage core` 完成主结果、`validate.py` 与 `check_exports.py`；仅需重建五份表格时使用默认 `--stage tables`，并选择新的仓库外输出目录。

重点检查逐段电量平衡、储能递推及边界、日末衔接、购电费用分项、更新时刻的信息边界，以及五份工作簿的模板位置和逐格数值。用 `scripts/compare_workbooks.py --expected results --actual "<外部计算目录>/results"` 比较仓库已有结果，说明可复现的命令、依赖版本和差异范围。

核验失败时保留失败证据并追踪到输入、求解或导出环节；不要通过覆盖提交结果来掩盖差异。完整补充实验属于 `--stage all`，只有实际运行后才能声称通过。
