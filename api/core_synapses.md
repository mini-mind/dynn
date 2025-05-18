# `dynn.core.synapses` API 文档

## 模块概览

`dynn.core.synapses` 模块负责定义和管理神经网络中神经元群体之间的突触连接。它提供了创建、初始化和修改突触权重及连接模式的功能，并且可以与学习规则模块集成以实现突触可塑性。

主要组件包括：

*   `SynapseCollection`: 一个用于表示和操作从一个神经元群体（突触前）到另一个神经元群体（突触后）的突触连接集合的类。

## 类详细说明

### `SynapseCollection`

#### 类名及其构造函数签名

`SynapseCollection(pre_population, post_population, name="synapses")`

#### 类描述

`SynapseCollection` 类是 DyNN 框架中管理神经元之间突触连接的核心。它存储突触权重矩阵、连接的存在性（掩码），并提供方法来初始化这些连接，计算由突触前脉冲引起的突触后电流，以及应用学习规则。一个 `SynapseCollection` 实例代表了从一个源 `NeuronPopulation` (突触前) 到一个目标 `NeuronPopulation` (突触后) 的所有连接。

#### 构造函数参数

*   `pre_population` (`NeuronPopulation`): 突触前神经元群体。即发出连接的神经元群体。
*   `post_population` (`NeuronPopulation`): 突触后神经元群体。即接收连接的神经元群体。
*   `name` (`str`, 可选, 默认值: `"synapses"`): 此突触集合的名称，用于识别和调试。

#### 主要属性/特性

*   `pre_pop` (`NeuronPopulation`): 对突触前神经元群体的引用。
*   `post_pop` (`NeuronPopulation`): 对突触后神经元群体的引用。
*   `name` (`str`): 突触集合的名称。
*   `weights` (`np.ndarray`): 一个二维 NumPy 数组，表示突触权重。维度为 `(num_post_neurons, num_pre_neurons)`，其中 `weights[i, j]` 是从 `pre_pop` 中的第 `j` 个神经元到 `post_pop` 中的第 `i` 个神经元的连接权重。
*   `connection_mask` (`np.ndarray`): 一个与 `weights` 形状相同的布尔类型二维 NumPy 数组。如果 `connection_mask[i, j]` 为 `True`，则表示从 `pre_pop[j]` 到 `post_pop[i]` 的连接实际存在；否则表示不存在连接。这对于实现稀疏连接非常重要。
*   `learning_rule` (`object` 或 `None`): 附加到此突触集合的学习规则实例。如果为 `None`，则连接是静态的（不可塑）。
*   `is_excitatory` (`bool`): 一个布尔标志，指示此突触集合中的连接是兴奋性的 (`True`) 还是抑制性的 (`False`)。默认为 `True`。此属性会影响权重的符号。

#### 主要方法

*   `initialize_weights(self, dist_config, connectivity_type='full', **kwargs)`:
    *   根据指定的统计分布和连接拓扑类型来初始化突触权重和连接掩码。
    *   **参数**:
        *   `dist_config` (`tuple` 或 `float`): 权重值的初始分布。如果是一个 `float`，所有存在的连接权重都初始化为该值。如果是一个 `tuple`，例如 `('uniform', (low, high))` 或 `('normal', (mean, std))`，则从相应分布中采样权重值。
        *   `connectivity_type` (`str`, 可选, 默认值: `'full'`): 指定连接的拓扑结构。支持的类型包括：
            *   `'full'`: 全连接，所有可能的突触前-突触后对之间都存在连接。
            *   `'sparse_prob'`: 按概率稀疏连接。需要额外关键字参数 `prob` (float, 连接概率，默认0.1)。
            *   `'sparse_num'`: 每个突触后神经元有固定数量的稀疏连接。需要额外关键字参数 `num_connections` (int, 每个突触后神经元的传入连接数，默认约为突触前神经元数量的10%)。
            *   `'neighborhood'`: 邻域连接，通常用于当突触前和突触后群体是同一个群体或具有空间结构时。需要额外关键字参数 `radius` (int, 邻域半径，默认1) 和 `allow_self_connections` (bool, 是否允许自连接，默认 `False`，仅当 `pre_pop` is `post_pop` 时相关)。此实现目前主要针对相同大小的群体或群体内连接进行了简化。
        *   `**kwargs`: 传递给特定连接类型的额外参数。
    *   **注意**: 此方法会同时设置 `self.weights` 和 `self.connection_mask`。对于不存在的连接，权重值通常设置为0。

*   `set_excitatory(self, is_excitatory)`:
    *   设置此突触集合是兴奋性还是抑制性，并相应地调整现有连接权重的符号。
    *   如果 `is_excitatory` 为 `True`，已连接的权重将取其绝对值。
    *   如果 `is_excitatory` 为 `False`，已连接的权重将取其绝对值的负数。
    *   **参数**:
        *   `is_excitatory` (`bool`): `True` 表示兴奋性，`False` 表示抑制性。

*   `get_input_currents(self, pre_spikes)`:
    *   根据当前时间步突触前神经元的脉冲活动，计算传递给每个突触后神经元的总输入电流。
    *   **参数**:
        *   `pre_spikes` (`np.ndarray`): 一个一维布尔数组，长度等于突触前神经元的数量。`pre_spikes[j]` 为 `True` 表示 `pre_pop` 中的第 `j` 个神经元在当前时间步发放了脉冲。
    *   **返回值**: `np.ndarray` - 一个一维浮点数数组，长度等于突触后神经元的数量。每个元素表示对应突触后神经元接收到的总输入电流。
    *   计算方式为：`I_post = (Weights * ConnectionMask) · PreSpikes` (点积)。

*   `apply_learning_rule(self, pre_spikes, post_spikes, dt, current_time)`:
    *   如果此突触集合已附加了一个学习规则 (即 `self.learning_rule` 不为 `None`)，则调用该学习规则的更新方法来修改突触权重。
    *   此方法通常在仿真器的每个时间步结束时被调用。
    *   **参数**:
        *   `pre_spikes` (`np.ndarray`): 突触前神经元的脉冲活动 (同 `get_input_currents` 中的定义)。
        *   `post_spikes` (`np.ndarray`): 一个一维布尔数组，指示突触后神经元群体中哪些神经元发放了脉冲。
        *   `dt` (`float`): 仿真时间步长。
        *   `current_time` (`float`): 当前仿真时间。

*   `set_learning_rule(self, learning_rule_instance)`:
    *   为此突触集合附加一个学习规则实例。 
    *   **参数**:
        *   `learning_rule_instance` (`object`): 一个实现了学习规则逻辑的对象实例 (例如 `STDP` 类的实例)。

*   `get_weights(self)`:
    *   返回当前的突触权重矩阵。
    *   **返回值**: `np.ndarray` - `self.weights`。

*   `get_connection_mask(self)`:
    *   返回当前的连接掩码矩阵。
    *   **返回值**: `np.ndarray` - `self.connection_mask`。

*   `__repr__(self)`:
    *   返回该突触集合的字符串表示形式，包括名称、突触前/后群体名称、权重矩阵形状以及实际连接数。
    *   **返回值**: `str`。

## (可选) 使用示例

```python
import numpy as np
from dynn.core.neurons import NeuronPopulation # 假设 NeuronPopulation 已定义
from dynn.core.synapses import SynapseCollection

# 假设我们已经创建了突触前和突触后神经元群体
# (需要 dynn.core.neurons.IzhikevichNeuron 或类似的定义)
class DummyNeuron:
    def __init__(self, name="neuron"):
        self.name = name
    def __len__(self):
        return 1 # Simplified for example

class DummyPopulation:
    def __init__(self, num_neurons, name):
        self.neurons = [DummyNeuron(name=f"{name}_{i}") for i in range(num_neurons)]
        self.name = name
    def __len__(self):
        return len(self.neurons)

pre_pop = DummyPopulation(num_neurons=10, name="InputPop")
post_pop = DummyPopulation(num_neurons=5, name="OutputPop")

# 1. 创建一个突触集合
synapses1 = SynapseCollection(pre_pop, post_pop, name="Input_to_Output")
print(synapses1)

# 2. 初始化权重
# 全连接，权重从正态分布 N(0.5, 0.1) 采样
synapses1.initialize_weights(dist_config=('normal', (0.5, 0.1)), connectivity_type='full')
print(f"Weights after full init (shape {synapses1.get_weights().shape}):\n", synapses1.get_weights())
print(f"Connections: {np.sum(synapses1.get_connection_mask())}")

# 将其设置为抑制性
synapses1.set_excitatory(False)
print(f"Weights after setting to inhibitory:\n", synapses1.get_weights())

# 稀疏连接，每个突触后神经元有2个来自突触前的连接，权重固定为0.8
synapses2 = SynapseCollection(pre_pop, post_pop, name="SparseFixed")
synapses2.initialize_weights(
    dist_config=0.8,
    connectivity_type='sparse_num',
    num_connections=2
)
print(synapses2)
print(f"Weights for sparse_num:\n", synapses2.get_weights())
print(f"Connection mask for sparse_num:\n", synapses2.get_connection_mask().astype(int))

# 3. 模拟获取输入电流
# 假设突触前群体中有一些神经元发放了脉冲
pre_spike_activity = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=bool)

# 使用 synapses1 (全连接，抑制性)
input_currents_from_syn1 = synapses1.get_input_currents(pre_spike_activity)
print(f"Input currents from synapses1 (inhibitory): {input_currents_from_syn1}")

# 使用 synapses2 (稀疏连接，兴奋性，固定权重0.8)
input_currents_from_syn2 = synapses2.get_input_currents(pre_spike_activity)
print(f"Input currents from synapses2 (excitatory): {input_currents_from_syn2}")

# 4. 附加和应用学习规则 (概念性)
class DummySTDP:
    def update_weights(self, synapse_collection, pre_spikes, post_spikes, dt, current_time):
        print(f"Learning rule applied to {synapse_collection.name} at time {current_time}ms")
        # 实际的STDP会在这里修改 synapse_collection.weights
        # 例如: synapse_collection.weights += 0.01 * np.outer(post_spikes, pre_spikes)
        pass

stdp_rule = DummySTDP()
synapses1.set_learning_rule(stdp_rule)

post_spike_activity = np.array([0, 1, 0, 1, 0], dtype=bool)
synapses1.apply_learning_rule(pre_spike_activity, post_spike_activity, dt=0.1, current_time=10.0)

``` 