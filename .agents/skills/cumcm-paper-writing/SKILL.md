---
name: cumcm-paper-writing
description: Generate or revise a mathematical modeling paper for this 2026 C-problem microgrid project using verified calculations, tables, and figures.
---

# C 题论文写作

先读取题目要求、`docs/paper.pdf` 的相关段落以及本次运行的核验结果。既有 PDF 是参考成果，仓库没有可编辑的原始论文源文件；需要生成源文件时，根据用户要求选择 LaTeX 或文档格式，并把源文件及构建说明保存在明确的论文目录。

分别说明四问的假设、信息可得时间、优化模型、求解办法、结果和局限。公式中的时间步长、单位、目标函数、储能效率和费用口径应与 `src/code/` 一致。数字、表格和图必须能追溯到本次运行的输出；发现与既有 PDF 不一致时先核对原因，再修改正文。不要臆造实验、文献或显著性结论；引用需核对原文。

生成后检查图表标题、编号、引用、公式符号和页面布局，并逐项对照五份结果表。若论文将用于正式提交，还应核对当次竞赛的格式与 AI 使用说明要求；`docs/ai-usage.pdf` 是既有说明，不能替代对新生成内容的真实披露。
