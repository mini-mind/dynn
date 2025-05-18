# `dynn.utils.probes` API 文档

## 模块概览

`dynn.utils.probes` 模块提供了一系列"探针"类，用于在SNN仿真过程中记录和监控各种数据。这些探针可以附加到神经网络上，以在指定的时间间隔内收集特定神经元群体、突触集合或自定义数据的状态。

主要组件包括：

*   `BaseProbe`: 所有探针类型的抽象基类，定义了通用接口和核心记录逻辑。
*   `PopulationProbe`: 用于记录一个或多个神经元群体中特定状态变量（如膜电位 `v`、脉冲发放 `fired`）的时间序列数据。
*   `SynapseProbe`: 用于记录一个或多个突触集合的状态，最常见的是记录突触权重矩阵的变化。
*   `CustomDataProbe`: 一个灵活的探针，允许用户提供一个自定义函数来在每个记录点收集和存储任何可计算的数据。

## 类详细说明

### `BaseProbe`

#### 类名及其构造函数签名

`BaseProbe(name, record_interval=1)`

#### 类描述

`BaseProbe` 是所有特定探针类的抽象基类。它管理记录的间隔、存储记录的数据和对应的时间戳，并提供了重置、数据获取和导出到CSV文件的通用功能。实际的数据收集逻辑由其子类通过重写 `_collect_data` 方法来实现。

#### 构造函数参数

*   `name` (`str`): 探针的唯一名称，用于识别。
*   `record_interval` (`int`, 可选, 默认值: `1`): 指定每隔多少个仿真时间步记录一次数据。例如，`record_interval=1` 表示每个时间步都记录，`record_interval=10` 表示每10个时间步记录一次。

#### 主要属性/特性

*   `name` (`str`): 探针的名称。
*   `record_interval` (`int`): 记录间隔。
*   `data` (`dict`): 一个字典，用于存储被记录变量的时间序列数据。字典的键是变量名 (字符串)，值是一个列表，列表中的每个元素对应一个记录时间点的值。
*   `time_data` (`list`): 一个列表，存储每个数据记录点对应的仿真时间戳 (通常为毫秒)。

#### 主要方法

*   `attempt_record(self, network, current_time_ms)`:
    *   由仿真器或网络在每个时间步调用，以尝试进行数据记录。只有当从上次记录开始的时间步数达到 `record_interval` 时，才会实际调用 `_collect_data` 方法进行记录。
    *   **参数**:
        *   `network` (`NeuralNetwork`): 当前的神经网络实例。
        *   `current_time_ms` (`float`): 当前仿真时间 (毫秒)。
    *   **返回值**: `bool` - 如果实际进行了记录，则为 `True`，否则为 `False`。

*   `_collect_data(self, network, current_time_ms)`:
    *   **抽象方法**，必须由子类实现。此方法包含实际从 `network` 中收集特定数据并将其存储到 `self.data` 中的逻辑。当满足记录条件时，由 `attempt_record` 内部调用。
    *   **参数**: 
        *   `network` (`NeuralNetwork`): 神经网络实例。
        *   `current_time_ms` (`float`): 当前仿真时间。
    *   **注意**: 直接调用此方法通常不是必需的，应通过 `attempt_record`。

*   `get_data(self)`:
    *   返回探针记录的所有数据的副本。
    *   **返回值**: `dict` - 一个字典，包含两个键：
        *   `'time'`: 一个包含所有记录时间戳的列表 (副本)。
        *   `'data'`: 一个字典 (副本)，其结构与 `self.data` 相同，包含每个被记录变量的时间序列数据。

*   `reset(self)`:
    *   重置探针的内部状态，清空所有已记录的 `data` 和 `time_data`，并将记录计数器归零。

*   `export_to_csv(self, filepath, delimiter=',')`:
    *   将记录的所有数据（时间和变量数据）导出到一个CSV文件中。如果记录的变量是向量或数组，它们会被展平，并在表头中用 `variable_name_index` 的形式表示每一列。
    *   **参数**:
        *   `filepath` (`str`): 要保存CSV文件的完整路径。
        *   `delimiter` (`str`, 可选, 默认值: `,`): CSV文件中使用的分隔符。

*   `__repr__(self)`:
    *   返回该探针实例的字符串表示形式，包括名称、间隔和已记录的点数。

---

### `PopulationProbe`

#### 类名及其构造函数签名

`PopulationProbe(name, population_name, state_vars, record_interval=1)`

#### 类描述

`PopulationProbe` 继承自 `BaseProbe`，专门用于记录神经网络中特定神经元群体 (`NeuronPopulation`) 的一个或多个状态变量随时间的变化情况。例如，它可以记录群体中所有神经元的膜电位、恢复变量或脉冲发放状态。

#### 构造函数参数

*   `name` (`str`): 探针的名称。
*   `population_name` (`str`): 要探测的神经元群体的名称 (必须是在 `NeuralNetwork` 中注册的群体名称)。
*   `state_vars` (`list` of `str`): 一个字符串列表，指定要从该神经元群体中记录的状态变量的名称 (例如 `['v', 'u', 'fired']`)。这些变量名应与 `NeuronPopulation` 对象中定义的状态变量名一致。
*   `record_interval` (`int`, 可选, 默认值: `1`): 记录间隔。

#### 主要方法

*   `_collect_data(self, network, current_time_ms)`:
    *   从 `network` 中获取名为 `self.population_name` 的神经元群体，然后请求该群体的 `self.state_vars` 中指定的所有状态变量的当前值。这些值（通常是NumPy数组，每个数组对应群体中所有神经元的一个状态）被存储到 `self.data` 字典中，键为状态变量名。

*   `__repr__(self)`:
    *   返回该群体探针实例的详细字符串表示形式。

---

### `SynapseProbe`

#### 类名及其构造函数签名

`SynapseProbe(name, synapse_collection_name, record_weights=True, record_interval=1)`

#### 类描述

`SynapseProbe` 继承自 `BaseProbe`，用于记录神经网络中特定突触集合 (`SynapseCollection`) 的状态。最常见的用途是记录突触权重矩阵随时间的变化，这对于分析学习过程非常有用。

#### 构造函数参数

*   `name` (`str`): 探针的名称。
*   `synapse_collection_name` (`str`): 要探测的突触集合的名称 (必须是在 `NeuralNetwork` 中注册的突触集合名称)。
*   `record_weights` (`bool`, 可选, 默认值: `True`): 如果为 `True`，则探针将记录突触集合的权重矩阵。未来可以扩展以记录其他突触属性。
*   `record_interval` (`int`, 可选, 默认值: `1`): 记录间隔。

#### 主要方法

*   `_collect_data(self, network, current_time_ms)`:
    *   从 `network` 中获取名为 `self.synapse_collection_name` 的突触集合。如果 `self.record_weights` 为 `True`，则获取该突触集合当前的权重矩阵 (通常是一个NumPy数组的副本) 并将其存储到 `self.data['weights']` 中。

*   `__repr__(self)`:
    *   返回该突触探针实例的详细字符串表示形式。

---

### `CustomDataProbe`

#### 类名及其构造函数签名

`CustomDataProbe(name, data_provider_fn, data_keys, record_interval=1)`

#### 类描述

`CustomDataProbe` 继承自 `BaseProbe`，提供了一种高度灵活的方式来记录仿真过程中的自定义数据。用户需要提供一个回调函数 (`data_provider_fn`)，该函数在每个记录点被调用，并返回一个包含待记录数据的字典。这使得用户可以记录任何可以从网络状态、仿真时间或其他外部来源计算得到的信息。

#### 构造函数参数

*   `name` (`str`): 探针的名称。
*   `data_provider_fn` (`callable`): 一个用户提供的函数。该函数应接受两个参数：`network` (当前的 `NeuralNetwork` 实例) 和 `current_time_ms` (当前的仿真时间)。它必须返回一个字典，该字典的键应与 `data_keys` 参数中指定的键相对应。
*   `data_keys` (`list` of `str`): 一个字符串列表，指定了期望从 `data_provider_fn` 返回的字典中的键。探针将为这些键中的每一个创建一个条目来存储其时间序列数据。
*   `record_interval` (`int`, 可选, 默认值: `1`): 记录间隔。

#### 主要方法

*   `_collect_data(self, network, current_time_ms)`:
    *   调用用户提供的 `self.data_provider_fn(network, current_time_ms)` 函数。期望该函数返回一个字典。
    *   对于 `self.data_keys` 中的每一个键，它会尝试从返回的字典中获取相应的值，并将其存储到 `self.data` 中。如果返回的数据是NumPy数组，则存储其副本。

*   `__repr__(self)`:
    *   返回该自定义数据探针实例的详细字符串表示形式。

## (可选) 使用示例

```python
import numpy as np
# 假设已定义：NeuralNetwork, NeuronPopulation, SynapseCollection
# from dynn.core.network import NeuralNetwork
# from dynn.core.neurons import NeuronPopulation
# from dynn.core.synapses import SynapseCollection
from dynn.utils.probes import PopulationProbe, SynapseProbe, CustomDataProbe

# --- 模拟所需的类占位符 (实际应从dynn.core导入) ---
class MockNeuronPopulation:
    def __init__(self, name, num_neurons):
        self.name = name
        self.num_neurons = num_neurons
        self.v = np.random.rand(num_neurons) * -70
        self.fired = np.zeros(num_neurons, dtype=bool)
    def get_all_states(self, state_vars):
        states = {}
        if 'v' in state_vars: states['v'] = self.v
        if 'fired' in state_vars: states['fired'] = self.fired
        return states
    def update(self, I_inj, dt):
        self.v += I_inj * dt + np.random.randn(self.num_neurons) * 0.1
        self.fired = self.v > -55.0
        self.v[self.fired] = -65.0

class MockSynapseCollection:
    def __init__(self, name, pre_pop, post_pop):
        self.name = name
        self.weights = np.random.rand(post_pop.num_neurons, pre_pop.num_neurons)
    def get_weights(self):
        return self.weights
    def update(self, learning_rule_instance, dt):
        # 模拟权重变化
        self.weights += np.random.randn(*self.weights.shape) * 0.001

class MockNeuralNetwork:
    def __init__(self, name="TestNet"):
        self.name = name
        self.populations = {}
        self.synapses = {}
        self.probes = []
        self.current_time = 0.0
    def add_population(self, pop):
        self.populations[pop.name] = pop
    def add_synapses(self, syn):
        self.synapses[syn.name] = syn
    def add_probe(self, probe):
        self.probes.append(probe)
    def step(self, inputs_map, dt):
        # 简化步骤
        for pop_name, pop_obj in self.populations.items():
            input_current = inputs_map.get(pop_name, np.zeros(pop_obj.num_neurons))
            pop_obj.update(input_current, dt)
        for syn_obj in self.synapses.values():
            syn_obj.update(None, dt) # 假设学习规则在此处应用
        for probe in self.probes:
            probe.attempt_record(self, self.current_time)
        self.current_time += dt
# --- 占位符结束 ---

# 1. 创建网络组件
net = MockNeuralNetwork()
pop1 = MockNeuronPopulation("Pop1", 10)
pop2 = MockNeuronPopulation("Pop2", 5)
syn12 = MockSynapseCollection("Syn12", pop1, pop2)

net.add_population(pop1)
net.add_population(pop2)
net.add_synapses(syn12)

# 2. 创建和添加探针
# PopulationProbe: 记录 Pop1 的膜电位和脉冲发放情况，每2步记录一次
pop_probe = PopulationProbe(name="Pop1_v_fired", 
                          population_name="Pop1", 
                          state_vars=['v', 'fired'], 
                          record_interval=2)
print(pop_probe)

# SynapseProbe: 记录 Syn12 的权重，每5步记录一次
syn_probe = SynapseProbe(name="Syn12_weights", 
                         synapse_collection_name="Syn12", 
                         record_weights=True, 
                         record_interval=5)
print(syn_probe)

# CustomDataProbe: 记录 Pop1 中发放脉冲的神经元数量，每个时间步记录
def count_pop1_spikes(network, current_time_ms):
    pop1_ref = network.populations.get("Pop1")
    if pop1_ref:
        num_spikes = np.sum(pop1_ref.fired)
        return {"pop1_spike_count": num_spikes, "network_time": current_time_ms}
    return {"pop1_spike_count": None, "network_time": current_time_ms}

custom_probe = CustomDataProbe(name="Pop1_SpikeCount", 
                               data_provider_fn=count_pop1_spikes, 
                               data_keys=["pop1_spike_count", "network_time"], 
                               record_interval=1)
print(custom_probe)

net.add_probe(pop_probe)
net.add_probe(syn_probe)
net.add_probe(custom_probe)

# 3. 运行仿真
duration_ms = 10.0
dt_ms = 1.0
num_steps = int(duration_ms / dt_ms)

print(f"\nRunning simulation for {num_steps} steps...")
for i in range(num_steps):
    # 模拟一些输入
    inputs = {"Pop1": np.random.rand(pop1.num_neurons) * 5.0}
    net.step(inputs, dt_ms)

print("Simulation finished.")

# 4. 获取和检查数据
pop_probe_data = pop_probe.get_data()
print(f"\n{pop_probe.name} recorded {len(pop_probe_data['time'])} points.")
if pop_probe_data['time']:
    print(f"  Time points: {pop_probe_data['time']}")
    print(f"  First 'v' record for Pop1: {pop_probe_data['data']['v'][0]}")
    print(f"  First 'fired' record for Pop1: {pop_probe_data['data']['fired'][0]}")

syn_probe_data = syn_probe.get_data()
print(f"\n{syn_probe.name} recorded {len(syn_probe_data['time'])} points.")
if syn_probe_data['time']:
    print(f"  Time points: {syn_probe_data['time']}")
    print(f"  Shape of first 'weights' record for Syn12: {syn_probe_data['data']['weights'][0].shape}")

custom_probe_data = custom_probe.get_data()
print(f"\n{custom_probe.name} recorded {len(custom_probe_data['time'])} points.")
if custom_probe_data['time']:
    print(f"  Time points: {custom_probe_data['time']}")
    print(f"  First 'pop1_spike_count' record: {custom_probe_data['data']['pop1_spike_count'][0]}")
    print(f"  First 'network_time' record: {custom_probe_data['data']['network_time'][0]}")

# 5. 导出到 CSV (可选)
pop_probe.export_to_csv("pop1_probe_data.csv")
syn_probe.export_to_csv("syn12_weights_data.csv")
custom_probe.export_to_csv("custom_probe_data.csv")

# 6. 重置探针 (如果需要再次运行或清除内存)
pop_probe.reset()
syn_probe.reset()
custom_probe.reset()
print("\nProbes have been reset.")
print(pop_probe)
print(syn_probe)
print(custom_probe)
``` 