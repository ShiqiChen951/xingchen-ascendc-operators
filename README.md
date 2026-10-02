# xingchen-ascendc-operators
中国电信星辰杯高校 AI 算子开发挑战赛（A组）参赛实现：基于 Ascend C 开发昇腾 NPU 自定义算子，涵盖 Sparse Softmax、mHcPre 和 DenseLightningIndexerGradKLLoss，记录正确性调试与性能优化过程。

本项目整理了我参加“中国电信星辰杯高校 AI 算子开发挑战赛（A组）”期间完成的 Ascend C 自定义算子实现及开发复盘。

比赛面向昇腾 NPU，要求根据给定算子的数学定义和接口规范完成实现，并通过正确性测试与性能评测。开发工作涉及 Host 侧算子注册、形状与数据类型推导、Tiling 参数设计，以及 AI Core Kernel 编程。

本仓库包含三个算子：
- Sparse Softmax：实现基于 index 分组和 CSR ptr 分段的 Softmax，支持多维输入，完成向量指数计算与中间结果复用等优化。
- mHcPre：实现 mHC 架构的前处理计算，包括 RMSNorm、参数投影、激活与加权聚合，完成 UB 分块及流水同步、多核写入相关问题的排查。
- DenseLightningIndexerGradKLLoss：实现 Lightning Indexer 的 KL 损失及相关梯度计算，通过梯度计算融合、Softmax 统计量缓存和 Online Softmax 减少重复计算。

项目采用先建立正确性基线、再逐步优化性能的开发方式，并记录编译错误、运行错误、精度问题和性能瓶颈的定位过程。
