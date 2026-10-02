# Sparse Softmax 算子

## 1.  算子说明

Sparse Softmax（稀疏 Softmax）是图神经网络（GNN）中的核心算子，用于对稀疏连接的节点特征进行分组归一化。与标准 Softmax 对整个维度做归一化不同，Sparse Softmax 根据 index 或 ptr 指定的分组信息，在每个组内独立计算 Softmax。该算子广泛应用于 GAT（Graph Attention Network）、GraphUNet、MessagePassing 等图神经网络的消息传递与注意力机制中。

在 GNN 的注意力计算流程中，Sparse Softmax 的位置如下：

```
Edge features / Attention scores
        ↓
Sparse Softmax（本算子）：按目标节点分组，对每组的注意力分数做 Softmax
        ↓
归一化后的注意力权重 → 用于加权聚合邻居节点特征
```

**参考资料：** https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.softmax

## 2. 输入输出说明

### 输入规格

| 参数名 | 类型 | 数据类型 | 维度(shape) | 说明 |
|--------|------|----------|-------------|------|
| src | 必选输入 | float16、float32、bfloat16 | [*]（任意形状） | 源张量，待计算 Softmax 的值 |
| index | 可选输入 | int64 | [*]（与 src 同形状） | 分组索引，指定每个元素所属的组。index 和 ptr 必须提供其中之一 |
| ptr | 可选输入 | int64 | [num_groups + 1] | CSR 格式的分组指针，ptr[i] 到 ptr[i+1] 为第 i 组的元素范围。给定 ptr 时按 CSR 方式计算，效率更高 |
| num_nodes | 可选输入 | int | - | 节点数量，即 index 中 max_val + 1，默认自动推断 |
| dim | 可选输入 | int | - | 执行归一化的维度，默认为 0 |

### 输出规格

| 参数名 | 数据类型 | 维度(shape) | 说明 |
|--------|----------|-------------|------|
| out | float16、float32、bfloat16 | 与 src 相同 | 分组 Softmax 结果，与 src 形状一致 |

### 形状约束

- index 和 ptr 必须提供其中之一，不可同时为 None
- index 的形状必须与 src 在 dim 维度上匹配
- ptr 为长度 num_groups + 1 的单调非递减序列，ptr[0]=0，ptr[-1]=src.size(dim)
- 当 dim < 0 时，实际维度为 dim + src.dim()
- 输出 out 与 src 形状相同

### 计算公式

**基于 index 模式：**

```
src_max[k] = max({src[i] | index[i] = k})    对每个组 k 求最大值
out[i] = exp(src[i] - src_max[index[i]]) / (Σ_{j: index[j]=index[i]} exp(src[j] - src_max[index[i]]) + ε)
```

**基于 ptr 模式（CSR）：**

```
src_max[k] = max({src[i] | ptr[k] ≤ i < ptr[k+1]})    对每个组 k 求最大值
out[i] = exp(src[i] - src_max[k]) / (Σ_{j=ptr[k]}^{ptr[k+1]-1} exp(src[j] - src_max[k]) + ε)
```

其中 ε = 1e-16，用于数值稳定性。

## 3. 算子逻辑说明

### 3.1 基于 index 的计算流程

1. **分组求最大值**：使用 scatter 操作，按 index 分组对 src 求 max，得到每个组的最大值 src_max
2. **减最大值**：通过 index_select 将 src_max 映射回每个元素，计算 src - src_max，保证数值稳定性
3. **求指数**：对减去最大值后的结果计算 exp
4. **分组求和**：使用 scatter 操作，按 index 分组对 exp 结果求 sum，加上 ε 防止除零
5. **归一化**：通过 index_select 将 sum 映射回每个元素，计算 out = exp_result / sum

### 3.2 基于 ptr 的计算流程（CSR）

1. **分组求最大值**：使用 segment 操作，按 ptr 指示的范围对 src 求 max，得到每个组的最大值 src_max
2. **扩展与减最大值**：使用 repeat_interleave 将 src_max 扩展到与 src 相同长度，计算 src - src_max
3. **求指数**：对减去最大值后的结果计算 exp
4. **分组求和**：使用 segment 操作，按 ptr 指示的范围对 exp 结果求 sum，加上 ε
5. **扩展与归一化**：使用 repeat_interleave 将 sum 扩展，计算 out = exp_result / sum

## 4.  任务要求

- **精度保障**：针对不同 Data type 和不同 shape 维度，设计算子逻辑，保证算子精度正确
- **性能优化**：充分发挥系统带宽能力，算子性能更优。重点关注 scatter/segment 操作的并行效率和内存访问模式
- **切分最优**：探索输入 tensor 的切分方式，找到不同输入 shape 场景下的最优解
- **泛化功能**：必须实现算子泛化功能，满足各类合法输入场景的计算需求
- **双模式支持**：需同时实现 index 模式和 ptr（CSR）模式两种分组方式

## 5.  功能示例

```python
import torch
from torch_geometric.utils import softmax

# 示例1：基于 index 的分组 Softmax
# 4个元素，分3组：index=[0,0,1,2] 表示前2个元素为组0，第3个为组1，第4个为组2
src = torch.tensor([1., 1., 1., 1.])
index = torch.tensor([0, 0, 1, 2])

out = softmax(src, index)
# 输出：tensor([0.5000, 0.5000, 1.0000, 1.0000])
# 组0：exp(1)/[exp(1)+exp(1)] = 0.5，组1和组2各只有1个元素，softmax为1.0

# 示例2：基于 ptr 的分组 Softmax（CSR格式）
# ptr=[0,2,3,4] 表示组0=[0:2]，组1=[2:3]，组2=[3:4]
src = torch.tensor([1., 1., 1., 1.])
ptr = torch.tensor([0, 2, 3, 4])

out = softmax(src, None, ptr)
# 输出：tensor([0.5000, 0.5000, 1.0000, 1.0000])
# 结果与 index 模式一致，但计算路径不同

# 示例3：多维张量 + 指定维度
# src 为 4x4 矩阵，按 dim=-1（最后一维）做 Softmax
src = torch.randn(4, 4)
ptr = torch.tensor([0, 4])

out = softmax(src, index, dim=-1)
# 输出：4x4 矩阵，每行按分组归一化

# 示例4：GNN 注意力场景
# 边的注意力分数，按目标节点分组 Softmax
# 假设5条边，目标节点 index=[0, 0, 1, 1, 1]
attention_scores = torch.tensor([0.5, 0.3, 0.8, 0.2, 0.6])
target_index = torch.tensor([0, 0, 1, 1, 1])

normalized_attention = softmax(attention_scores, target_index)
# 输出：每个目标节点的注意力权重归一化
# 组0：[exp(0.5), exp(0.3)] / [exp(0.5)+exp(0.3)]
# 组1：[exp(0.8), exp(0.2), exp(0.6)] / [exp(0.8)+exp(0.2)+exp(0.6)]
```

## 6.  测试用例覆盖范围

- **数据类型**：float16、float32、bfloat16
- **分组模式**：index 模式、ptr（CSR）模式
- **维度场景**：一维(S=100)、二维(S=256, D=64)、多维(S=128, D=32, H=16)
- **dim 参数**：dim=0、dim=-1、dim=1
- **分组特征**：均匀分组（每组元素数相同）、非均匀分组（每组元素数不同）、单元素组、大组
- **边界场景**：index 中所有元素为同一组、每组只有1个元素、大量小组（稀疏图）
- **数值稳定性**：大数值输入（验证减最大值策略的有效性）、零值输入