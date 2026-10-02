# xingchen-ascendc-operators
中国电信星辰杯高校 AI 算子开发挑战赛（A组）参赛实现：基于 Ascend C 开发昇腾 NPU 自定义算子，涵盖 Sparse Softmax、mHcPre 和 DenseLightningIndexerGradKLLoss，记录正确性调试与性能优化过程。

本项目整理了我参加“中国电信星辰杯高校 AI 算子开发挑战赛（A组）”期间完成的 Ascend C 自定义算子实现及开发复盘。

比赛面向昇腾 NPU，要求根据给定算子的数学定义和接口规范完成实现，并通过正确性测试与性能评测。开发工作涉及 Host 侧算子注册、形状与数据类型推导、Tiling 参数设计，以及 AI Core Kernel 编程。

本仓库包含三个算子：
- Sparse Softmax：实现基于 index 分组和 CSR ptr 分段的 Softmax，支持多维输入，完成向量指数计算与中间结果复用等优化。
- mHcPre：实现 mHC 架构的前处理计算，包括 RMSNorm、参数投影、激活与加权聚合，完成 UB 分块及流水同步、多核写入相关问题的排查。
- DenseLightningIndexerGradKLLoss：实现 Lightning Indexer 的 KL 损失及相关梯度计算，通过梯度计算融合、Softmax 统计量缓存和 Online Softmax 减少重复计算。

项目采用先建立正确性基线、再逐步优化性能的开发方式，并记录编译错误、运行错误、精度问题和性能瓶颈的定位过程。

# 星辰杯 Ascend C 自定义算子开发与优化

本项目整理了“中国电信星辰杯高校 AI 算子开发挑战赛（A组）”中的三个参赛算子：Sparse Softmax、DenseLightningIndexerGradKLLoss 和 mHcPre。比赛由中电信人工智能科技有限公司与华为技术有限公司共同发起，围绕昇腾 NPU 上的算子实现与性能评测展开。

项目基于 Ascend C，目标平台为 Ascend 910B。开发过程覆盖数学语义理解、Host 注册、Tiling 参数设计、AI Core Kernel 实现、正确性调试和性能迭代。

## 结果概览

| 题目 | 难度 | 记录中的正确性结果 | 记录中的性能 | 当前源码状态 |
|---|---|---|---|---|
| Sparse Softmax | 简单 | 5/5 Pass | 五个测试点平均约 5.28 μs；相对对照版本降低约 17.6% | 单核；批量 Vector Exp；小组复用指数结果 |
| DenseLightningIndexerGradKLLoss | 中等 | 7/7 Pass，评测显示输出错误占比 0.00% | 七个测试点约 560.98 μs～11.3 ms | 仅核 0 实际计算；梯度融合、UB 缓存、Online Softmax |
| mHcPre | 困难 | 复盘记录为 Pass | 已记录的一次通过结果约 99.48 μs | 单核正确性基线；UB 分块、混合精度、显式流水同步 |

**数据说明：**上述结果来自开发复盘文档中的比赛评测记录。本 README 对照了本仓库源码，但未在新的 Ascend 910B 环境中重新执行测试。压缩包没有包含原始评测日志、每个测试点的完整输入规模和环境版本，因此这些数值用于记录当时结果，不代表任意输入下的耗时或通用性能保证。评测显示 0.00% 错误占比，也不等于与参考输出逐位完全相同。

## 仓库内容与代码入口

| 目录 | 内容 |
|---|---|
| `SparseSoftmax_problem_298_template/` | 分组 Softmax 题目与工程 |
| `DenseLightningIndexerGradKlLoss_problem_299_template/` | Indexer KL 损失和梯度题目与工程 |
| `MhcPre_problem_300_template/` | mHC 前处理题目与工程 |

每个工程包含题目 Markdown、`code/CMakeLists.txt`、`code/op_host/` 和 `code/op_kernel/`。建议先读题目，再读 Host 和 Tiling，最后按 `Process()` 的调用顺序读 Kernel。

| 算子 | Host | Kernel |
|---|---|---|
| Sparse Softmax | [sparse_softmax.cpp](SparseSoftmax_problem_298_template/code/op_host/sparse_softmax.cpp) | [sparse_softmax.cpp](SparseSoftmax_problem_298_template/code/op_kernel/sparse_softmax.cpp) |
| DenseLightningIndexerGradKLLoss | [dense_lightning_indexer_grad_kl_loss.cpp](DenseLightningIndexerGradKlLoss_problem_299_template/code/op_host/dense_lightning_indexer_grad_kl_loss.cpp) | [dense_lightning_indexer_grad_kl_loss.cpp](DenseLightningIndexerGradKlLoss_problem_299_template/code/op_kernel/dense_lightning_indexer_grad_kl_loss.cpp) |
| mHcPre | [mhc_pre.cpp](MhcPre_problem_300_template/code/op_host/mhc_pre.cpp) | [mhc_pre.cpp](MhcPre_problem_300_template/code/op_kernel/mhc_pre.cpp) |

## 先理解几个开发概念

| 名称 | 通俗解释 | 代码中的作用 |
|---|---|---|
| Host | CPU 侧的准备工作 | 注册输入输出，读取形状、类型和属性，决定启动方式 |
| TilingData | 给 NPU 的任务说明 | 传递维度、分块大小、模式、核分工和常量 |
| Kernel | NPU 上真正执行计算的函数 | 完成归一化、投影、梯度等运算 |
| GM / GlobalTensor | 设备上的大容量全局存储及其访问接口 | 保存输入与最终输出 |
| UB / LocalTensor | 每个计算核中的片上工作区及其访问接口 | 暂存分块数据和可复用的中间结果 |
| Scalar / Vector | 标量流水和向量流水 | 前者处理标量与控制，后者批量计算 |
| MTE / DataCopy | 数据搬运流水及搬运操作 | 在 GM 和 UB 之间传输数据 |
| SetFlag / WaitFlag | 不同流水间的数据依赖同步 | 确保前一步完成后再读取结果 |

这些工程并不是单纯把数学公式翻译成 C++。Host 注册的数据类型、Tiling 参数和 Kernel 对内存的解释必须一致，搬运与计算之间也需要正确同步。

## 1. Sparse Softmax：每个组分别归一化

### 1.1 题目在计算什么

普通 Softmax 对一个集合计算归一化权重；Sparse Softmax 先分组，再对每个组单独计算。例如：

```text
src   = [1, 1, 1, 1]
index = [0, 0, 1, 2]
out   = [0.5, 0.5, 1, 1]
```

前两个元素属于同一组，平分权重；后两个元素各自独立成组，所以输出都是 1。这类操作可以用于图神经网络中同一节点相关边的注意力归一化。

代码采用稳定的计算方式：先求组内最大值 `m`，再计算 `exp(x-m)`，最后除以指数和加 `eps`。减去相同的最大值不改变理想 Softmax 的比例，却能避免直接对很大的数取指数。

### 1.2 两种分组方式

| 模式 | 示例 | 含义 |
|---|---|---|
| index | `[0,0,1,1,1,2]` | 每个位置直接给出组编号，同组元素可以不连续 |
| CSR ptr | `[0,2,5,6]` | 相邻边界确定组区间：`[0,2)`、`[2,5)`、`[5,6)` |

Host 根据可选输入选择模式；当前实现中非空 index 优先。Kernel 中 `Process()` 分派到 `ProcessIndex()` 或 `ProcessPtr()`。

### 1.3 代码如何处理多维输入

Host 将张量按指定 `dim` 拆成 `outerSize × dimSize × innerSize`，负维度先加上张量的维数。例如形状 `[128,32,16]`、`dim=1`，得到 `128 × 32 × 16`。

固定一个 `(outer,inner)`，沿中间的 `dimSize` 处理分组。对应线性地址为：

```cpp
outer * dimSize * innerSize + d * innerSize + inner
```

因此不需要针对一维、二维、三维张量分别写一套算法。`GroupId()` 根据 index 的布局读取组号，ptr 路径则按区间边界遍历。

### 1.4 实现与调试过程

| 阶段 | 现象或工作 | 处理与结果 |
|---|---|---|
| 建立基线 | 实现两种模式和 max、Exp、sum、归一化 | 先把完整计算路径写出来 |
| 正确性排查 | 能编译运行，但隐藏测试 0/5，记录中错误占比为 100% | 排查 Exp、流水同步和分组逻辑 |
| 接口核对 | Kernel 曾按 int64 读取 index/ptr，实际工程接口使用 int32 | 统一 Host 注册和 `GlobalTensor<int32_t>`，恢复 5/5 Pass |
| 性能优化 | 同一组的指数结果被重复计算 | 批量 Vector Exp；小组缓存指数结果 |
| 多核尝试 | 按独立 slice 切分后只有 3/5 通过，两个测试点约 50% 错误 | 恢复单核，保留已验证的计算优化 |

int32/int64 的问题可以理解为“拿错尺子读内存”：原数据每个索引占 4 字节，却按 8 字节读取，分组编号和边界都会被错误解释。最终代码与实际接口保持一致。

多核尝试的精确根因没有充分确认，不能把回退单核直接等同于已经证明某一种硬件冲突。

### 1.5 优化后的代码如何减少工作

代码设置 `SOFTMAX_TILE_ELEMS=256`，为指数输入、输出和位置记录分配 UB 缓冲。

对于不超过 256 个元素的组：

1. 求组内最大值。
2. 将 `x-max` 收集到 UB，index 模式同时记录原位置。
3. `RunExp()` 批量执行一次 Vector Exp。
4. 对指数结果求和，只计算一次 `1/(sum+eps)`。
5. 复用 UB 中的指数结果，乘倒数并写回对应位置。

关键变化是：求指数和后不再重新计算指数。较大的组使用分块路径；由于不能将整个组的指数结果同时保存在当前缓冲中，仍存在重复读取或计算。256 是本实现的固定分块选择，并未证明是所有输入的最优值。

### 1.6 最终结果与性能

| 测试点 | 对照版本 / μs | 优化版本 / μs | 优化版本结果 |
|---|---:|---:|---|
| 1 | 7.78 | 5.58 | Pass |
| 2 | 4.60 | 4.72 | Pass |
| 3 | 7.36 | 5.96 | Pass |
| 4 | 6.66 | 5.92 | Pass |
| 5 | 5.64 | 4.20 | Pass |
| 算术平均 | 6.408 | 5.276 | 5/5 Pass |

按上表计算，平均时延下降 `(6.408-5.276)/6.408 ≈ 17.6%`。复盘中的“约 6.36 μs”与这五项明细略有出入，这里使用可直接复算的明细平均值。

第二个测试点略慢，说明该优化并非对所有规模都收益；现有记录不足以区分控制开销和测量波动。当前 Host 明确设置 `SetBlockDim(1)`，不能把这组收益描述成多核并行带来的加速。

## 2. DenseLightningIndexerGradKLLoss：让轻量打分器学习完整注意力

### 2.1 题目在计算什么

可以将完整 Attention 看作“老师”，Lightning Indexer 看作更轻量的“学生”：老师生成目标概率分布 `p`，学生生成预测分布 `q`，通过 KL 散度衡量两者差异，并计算学生参数的梯度。

代码的主要计算如下：

- Attention 对 query/key 做缩放点积，各 head 分别 Softmax，再平均得到目标分布 `p`。
- Indexer 对 queryIndex/keyIndex 做点积，经过 ReLU，再按 weights 加权求和，Softmax 得到 `q`。
- loss 累加 `p * log((p+eps)/(q+eps))`。
- 输出 `dQueryIndex`、`dKeyIndex`、`dWeights` 和 FP32 loss。

对于代码使用的反向公式，核心中间量是 `dI=q-p`。点积为正时，经 ReLU 反向得到 `dS=weight*dI`，再同时更新三个梯度；点积不大于零时，该路径跳过。

### 2.2 从 Process() 理解最终代码

| 步骤 | 代码工作 | 为什么这样做 |
|---|---|---|
| 初始化 batch | 清零 dKeyIndex | 多个 query 会向同一个 key 梯度累加 |
| 处理一个 query | `ComputeIndexerMaxAndSum()` | 获取学生分布的归一化统计量 |
| 准备教师分布 | 缓存每个 Attention head 的 max 和指数和 | 同一个 query/head 的所有 key 都能复用 |
| 初始化 UB 梯度区 | 清零 dWeights 和 dQueryIndex | 保持当前 query 的累加状态 |
| 遍历 key | 求一次 p、q、dI，累计 loss | 避免按梯度维度重复求概率 |
| 遍历 Indexer head | 求点积、检查 ReLU、求 dS | 共享同一个反向中间量 |
| 更新三个梯度 | UB 累加 dWeights/dQueryIndex，GM 累加 dKeyIndex | 一个计算循环完成多个输出 |
| 完成 query | 写回 UB 梯度 | 减少部分反复写 GM 的操作 |

`gradBuf_` 存放 FP32 的 dWeights/dQueryIndex 累加值，`mainSoftmaxBuf_` 存放 Attention 的归一化统计量。需要注意，dKeyIndex 当前仍在 GM 中按输出 dtype 反复读写，并不是所有梯度都全程 FP32 驻留 UB；FP16 输出时仍存在逐次转换产生的精度和访存开销。

### 2.3 开发过程中最大的瓶颈

第一版按数学公式拆成多个辅助函数，方便验证，但不同梯度函数会反复调用同一条概率计算链。

例如计算 dQueryIndex 的每个维度时，都重新求 dI；dI 又要计算两套概率，内部还会重复扫描所有 key 求 Softmax 的最大值和指数和。随后 dKeyIndex 再把这些工作做一次。

这就像统计班级成绩时，为每个学生的每个科目反复重新计算全班总分：结果可能正确，但重复工作非常多。

### 2.4 调试与优化过程

| 阶段 | 问题 | 修改 | 记录中的结果 |
|---|---|---|---|
| Host 接口 | GetWorkspaceSize 返回 161002，尚未执行 Kernel | loss 的 Host dtype 组合与 Kernel FP32 类型保持一致 | 进入 Kernel 测试阶段 |
| 标量基线 | 小规模通过，大规模超时 | 检查辅助函数调用链 | 2/7 Pass，其余 TLE |
| 第一次融合 | 不同梯度维度反复计算 dI | 融合 dWeights 和 dQueryIndex，UB 累加 | 第 3 个测试点由 TLE 转 Pass |
| 第二次融合 | dKeyIndex 仍重复计算概率和 dS | 三个梯度共用同一次 dI/dS | 7/7 Pass |
| Softmax 缓存 | 每个 key 重新求 Attention max/sum | 按 query/head 提前计算，保存到 UB | 减少重复扫描，进入 μs～ms 级记录 |
| Online Softmax | Indexer 先求 max 再求 sum，需要两遍扫描 | 一遍扫描动态更新 max 和指数和 | 最终文档记录仍为 7/7 Pass |

早期文档记录第一次融合中测试点 1 约 `571 ms → 290 ms`、测试点 2 约 `494 ms → 251 ms`。这属于早期版本记录，不与下面的最终版本数据混为一组基准，也不据此宣称相对官方实现的加速比。

### 2.5 Online Softmax 通俗解释

遍历分数时同时维护当前最大值 `m` 和相对它的指数和 `s`：

- 新分数不超过 m：直接增加 `exp(score-m)`。
- 新分数超过 m：将旧指数和重新缩放为 `s*exp(m-score)`，加上新最大项的 1，再更新 m。

这相当于“基准变了，就把之前的累计值换算到新基准”，把求 max 和求指数和合并成一次扫描。当前代码仍会为输出概率再计算分数，因此不能理解成整个 Softmax 只访问一次输入。

### 2.6 最终结果与性能

| 测试点 | 缓存优化阶段 | 文档最终记录 | 最终正确性 |
|---|---:|---:|---|
| 1 | 约 592 μs | 560.98 μs | Pass / 0.00% |
| 2 | 约 587 μs | 555.84 μs | Pass / 0.00% |
| 3 | 约 1.19 ms | 1.13 ms | Pass / 0.00% |
| 4 | 约 2.43 ms | 2.33 ms | Pass / 0.00% |
| 5 | 约 9.49 ms | 9.01 ms | Pass / 0.00% |
| 6 | 约 3.14 ms | 2.95 ms | Pass / 0.00% |
| 7 | 约 12.2 ms | 11.3 ms | Pass / 0.00% |

当前 Host 启动多个核，但 `Process()` 的第一段是 `if (GetBlockIdx()!=0) return;`，因此**实际只有核 0 计算**。这个版本完成了计算复用和算法层面的初步优化，还没有充分利用多核、向量和矩阵计算能力。

源码还保留了早期独立梯度和非缓存概率函数。阅读最终执行路径时应以 `Process()` 的实际调用为准，不能把保留函数都当作当前主流程。

## 3. mHcPre：将多个隐状态分支组合成后续层输入

### 3.1 题目在计算什么

mHcPre 是 mHC 超连接架构中的前处理算子。对于每个 token，输入有 n 个长度为 D 的分支；算子根据这些分支计算混合权重，再将它们加权组合成一个长度 D 的输入，并生成后续连接需要的 hPost 和 hRes。

| 张量 | 形状 | 当前接口 dtype |
|---|---|---|
| x | `(B,S,n,D)` | FP16 / FP32 |
| phi | `(n²+2n,nD)` | FP32 |
| alpha | `(3,)` | FP32 |
| bias | `(n²+2n,)` | FP32 |
| gamma，可选 | `(n,D)` | FP32 |
| hIn | `(B,S,D)` | 与 x 相同 |
| hPost | `(B,S,n)` | FP32 |
| hRes | `(B,S,n,n)` | FP32 |

这里的 n 表示多个隐状态分支，并不是 n 个 attention head。

### 3.2 最终代码的五个步骤

Host 将 `(B,S)` 展平成 `rowNum=B*S`，每一行有 `nD=n*D` 个输入元素，投影输出长度为 `projDim=n*n+2*n`。

| 函数 | 通俗解释 | 关键实现 |
|---|---|---|
| `ComputeInvRms()` | 测量这一行输入的整体幅度，得到缩放系数 | 分块平方和；`invRms=1/sqrt(mean(x²)+normEps)` |
| `ComputeWAll()` | 用参数矩阵从输入中提取混合权重信息 | 可选 gamma 缩放；与 phi 各行点积；结果乘 invRms |
| `ComputeHPreAndHPost()` | 从投影结果生成输入混合和后续输出权重 | 两段 Sigmoid；hPre 留在 UB，hPost 写 GM |
| `ComputeHRes()` | 生成残差连接参数 | 对剩余 n² 项做线性缩放和偏置，不做 Sigmoid |
| `ComputeHIn()` | 按 hPre 将 n 个分支加权相加 | 沿 D 分块，FP32 累加，最后转成 x 的 dtype |

代码中的具体变换为：

```text
w[j]       = invRms * sum_k(x[k] * gamma[k] * phi[j,k])
             （未提供 gamma 时，gamma[k] 按 1 处理）
hPre[i]    = sigmoid(alpha[0] * w[i] + bias[i]) + hcEps
hPost[i]   = 2 * sigmoid(alpha[1] * w[n+i] + bias[n+i])
hRes[t]    = alpha[2] * w[2n+t] + bias[2n+t]
hIn[d]    = sum_i(hPre[i] * x[i,d])
```

注意 hIn 加权的是原始 x，不是投影结果。hPre 只作为内部中间量，没有作为最终输出。

### 3.3 如何使用 UB 和混合精度

输入可能很长，因此 RMS 统计和 hIn 聚合采用 `MHC_PRE_TILE_ELEMS=4096` 分块，避免把整行输入一次性放入 UB。

w 和 hPre 较小，分别使用最多 80 个、8 个 float 的缓冲。这些容量依赖题目的 n 范围，当前实现不能不加修改地推广到任意 n。

投影目前仍主要通过 Scalar `GetValue()` 从 GM 读取 x、gamma 和 phi，而不是全部采用块搬运或 Matmul。投影使用四路独立 FP32 累加器，最后合并结果，改变了长点积的累加顺序，也减少了中间值反复读写 UB；不能保证这种顺序在所有输入上都比单路累加误差更小。

### 3.4 编译、接口和正确性调试过程

| 阶段 | 现象 | 处理 |
|---|---|---|
| 补全工程 | 初始 Kernel 的 Init/Process 基本为空 | 建立 Host、Tiling 和五阶段计算链 |
| 编译限制 | AICore 对当时使用的 unsigned integer→float 转换报错 | Host 预计算 `invND=1/nD`，Kernel 改用乘法 |
| 数学函数适配 | `sqrtf` 在当前编译环境不可用 | 改成受支持的 `sqrt` |
| Host Runtime Error | GetWorkspaceSize 返回 161002 | 核对异构 dtype；参数和 hPost/hRes 使用 FP32，x/hIn 跟随输入类型 |
| 流水依赖 | 搬运或 Exp 的结果马上被 Scalar 读取 | 加入 MTE2→Scalar、Scalar→Vector、Vector→Scalar 事件同步 |
| 数值与布局核对 | 能运行但 Wrong Answer | 检查 phi 索引、gamma 展平、w 切分、偏置和 hIn 地址；调整点积累加 |
| 并行排查 | 早期多核版本约 37%～50% 输出错误，增加同步后仍有错误 | 保持计算公式不变，改为单核；记录由 WA 转为 Pass |

不同复盘文档强调了不同排查阶段。最终源码同时保留异构 dtype、四路累加、显式同步和单核执行，不能把全部正确性改善都归因于一个修改。

### 3.5 多核写入问题应如何理解

多个核虽然可能写不同元素，但相邻的小输出可能落在同一条 CacheLine 上。复盘据此怀疑 Scalar `GlobalTensor::SetValue()` 写 GM 时存在跨核缓存写回覆盖。

**已观察到的证据是：**计算逻辑不变，关闭多核后通过测试，说明问题与多核执行有关。现有材料不足以排除所有切分、缓存或同步因素，因此 CacheLine 冲突应作为待进一步验证的根因假设，而不是已经独立证明的结论。

当前 Host 设置 `used_core_num=1`、`rows_per_core=row_num`。这既是定位问题的控制变量实验，也是当前保留的正确性基线；不代表已完成安全的多核优化。

### 3.6 最终结果与性能

复盘记录的通过结果约为 **99.48 μs**。材料未提供完整测试点数量和各点耗时，因此不将这一数值写成所有测试点的平均时延，也不计算缺少对照数据的加速比。

当前最大的潜在瓶颈是投影部分大量 Scalar GM 读取。后续可以尝试 x/phi 分块、向量或矩阵计算、hIn 的批量乘加，以及明确保证写入安全的行级多核调度。

## 开发方法与后续工作

本项目采用“先正确，再优化”的方式：先写与公式对应的基线，区分编译、Host 校验、Kernel 执行和数值错误，再对重复计算、数据复用和硬件执行进行逐项修改。

当前已经实现并在比赛记录中通过验证的内容包括：

- Sparse Softmax 的 index/CSR 双模式、多维处理和小组指数复用。
- DenseLightningIndexerGradKLLoss 的损失与三个梯度、融合计算和归一化统计量复用。
- mHcPre 的完整前处理链路、混合精度接口、UB 分块及流水同步。

后续工作包括补齐原始评测日志和可复现实验、独立验证并行写入问题、设计多核归约、减少 Scalar GM 访问，以及评估 Vector/Matmul 和流水化的收益。Dense 算子中 dKeyIndex 的高精度累加与批量写回也值得进一步改进。

## 构建与复现范围

三个工程均使用 CMake，`find_package(ASC REQUIRED)`，目标计算单元为 `ascend910b`；需要比赛平台对应的 Ascend C/CANN 开发环境，不能直接用普通 Windows C++ 环境构建。

各工程的算子包配置有所不同：mHcPre 使用 `TYPE RUN`，Sparse Softmax 和 DenseLightningIndexerGradKLLoss 使用 `TYPE SHARED`。当前仓库未附平台提交入口、独立调用程序、测试输入及完整环境版本，所以这里不提供未经验证的通用构建命令。复现时需结合实际平台的构建、安装和调用流程。

## License

仓库附有 Apache License 2.0，详见 [LICENSE](LICENSE)。比赛模板和第三方内容的原有版权及许可声明仍需保留；仓库许可证不替代这些内容原本适用的许可要求。
