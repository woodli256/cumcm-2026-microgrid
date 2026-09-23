# 项目约定

本仓库保存 2026 数学建模竞赛 C 题的既有源码与结果。改动模型时保留题目四问的信息边界、十分钟时间区间和费用口径；不要把尚未发布的预测或未来实测值用于当前决策。原始附件保持只读，计算输出放在仓库之外。

按任务选择 `.agents/skills/` 中的相关 Skill：

- `cumcm-data-check`：检查官方附件的结构和输入口径。
- `cumcm-model-code`：生成或修改模型、求解及实验代码。
- `cumcm-results-audit`：复算与核验五份结果表。
- `cumcm-paper-writing`：依据可追溯结果生成或修订论文。

运行方式和文件说明见 `README.md`。不要把 `docs/paper.pdf` 视为可编辑论文源文件。
