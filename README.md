# 2026 数学建模竞赛 C 题：微网购电与储能调度

本仓库保存《光伏与电价波动下的微网购电及储能调度》的复现代码、五份结果表和论文 PDF。模型以十分钟为时间区间，处理负载、光伏和电价波动下的购电与储能调度。论文中的数值及结论应以重新运行和核验后的结果为准。

## 仓库内容

- `src/main.py`：统一入口；`src/code/` 是求解、实验、导出和核验程序，`src/figures/` 与 `src/_utils/` 是绘图程序及公共模块。
- `results/`：原始提交包中的五份结果工作簿，供对照；请勿把它们当作无需核验的真值。
- `docs/paper.pdf`：现有论文 PDF，供核对；原始可编辑论文源文件未包含在收到的提交包中。
- `docs/ai-usage.pdf`：原始提交包中的 AI 工具使用说明。
- `.agents/skills/`：针对本题的数据检查、模型代码、结果核验和论文写作流程。

题目原始附件没有放进本仓库。运行时需自行准备官方 C 题附件目录，其中应有 `附件1.xlsx` 至 `附件4.xlsx`，以及 `附件5/result1.xlsx`、`result2.xlsx`、`result3.xlsx`、`result4-2.xlsx`、`result4-3.xlsx` 五份空模板。不要用仓库的 `results/` 替代空模板。

## 复现

原工程记录的运行环境为 Python 3.13。先安装核心依赖：

```powershell
python -m pip install -r requirements.txt
```

在仓库之外选择新的输出目录，生成五份工作簿：

```powershell
python src/main.py --attachments "<官方C题附件目录>" --output-dir "<外部输出目录>"
```

运行主模型及核验程序：

```powershell
python src/main.py --attachments "<官方C题附件目录>" --stage core --work-dir "<外部空计算目录>"
```

`--stage all` 会继续运行补充实验和绘图，耗时明显更长；使用前安装 `requirements-figures.txt`。工作目录必须在仓库之外，且应为空目录。输出目录中已有同名结果时，程序默认拒绝覆盖；确认要覆盖时才使用 `--overwrite`。

## 核验范围

`--stage core` 会执行 `solve.py`、`validate.py`、`report_data.py`、`export_xlsx.py` 和 `check_exports.py`。这些程序检查物理约束、费用及导出表格的一致性；完整补充实验请用 `--stage all`。论文 PDF 是既有成果；若修改模型或数据，需重新核对论文中的每一个相关数字、图和结论。

如需将重算表格与仓库中的原提交表格逐格对照：

```powershell
python scripts/compare_workbooks.py --expected results --actual "<外部计算目录>/results"
```

本次复核的环境与结果见 `VERIFICATION.md`。

## 开源许可

本仓库的代码、结果表、论文、说明文档和 Codex Skills 均采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可。转载、使用或改编时，请注明来源并标明所作修改；完整条款见 `LICENSE`。
