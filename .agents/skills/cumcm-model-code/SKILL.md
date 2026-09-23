---
name: cumcm-model-code
description: Generate or revise Python modeling, optimization, forecast, and experiment code for this 2026 C-problem microgrid repository.
---

# C 题模型代码

先定位目标：`src/code/solve.py` 是主求解流程，`validate.py` 检查物理与费用，其他 `code/` 文件是独立实验，`src/main.py` 编排运行。保留现有统一入口和输出格式；新增实验优先写成独立模块，避免把论文章节逻辑嵌入求解器。

该题以 144 个十分钟区间表示一天。改动前明确功率与电量单位、储能效率、期初期末储量、合同调整和紧急购电的结算口径。四问可用信息不同；预测器和滚动决策只能读取决策时已经发布的数据。对改变目标函数、约束或信息边界的代码，给出可检验的数学说明和对应核验。

先用官方附件在仓库外运行受影响的最小实验，再运行 `python src/main.py --attachments "<附件目录>" --stage core --work-dir "<外部空目录>"`。比较原有结果表及关键指标；数值改变时记录原因，通知论文流程重核相关表述。不要用生成的结果反向当作模型输入。
