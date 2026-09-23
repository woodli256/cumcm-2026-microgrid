---
name: cumcm-data-check
description: Check official input workbooks and templates for this 2026 C-problem microgrid model before reproducing results or changing ingestion code.
---

# C 题附件检查

使用 `src/code/ingest.py` 与 `src/code/export_five.py` 作为当前输入结构的依据。先确认附件目录包含 `附件1.xlsx` 至 `附件4.xlsx`，以及 `附件5/` 下五份原始空模板；仓库 `results/` 内的是已生成的结果，不是模板。

检查表名、时间粒度、缺失值和维度。现有读取程序期望负载、光伏、电价各为 `(365, 144)`，基础参数为 `(144, 3)`，预报为 `(1460, 24)`。若题目附件版本不同，应先说明差异再修改读取逻辑，不要悄悄补值或改变时间对齐。

保持原始工作簿只读。需要转换数据时，将生成的输入与检查记录放到仓库外的专用计算目录，并记录附件版本或哈希，方便后续结果追溯。
