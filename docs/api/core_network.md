# `dynn.core.network` API 文档

## 模块概览

`dynn.core.network` 模块定义了 `NeuralNetwork` 类，它是整个脉冲神经网络 (SNN) 的顶层容器和管理器。它负责组织神经元群体、突触连接，协调仿真步骤中的信息流，并管理数据探针。

主要组件包括：

*   `NeuralNetwork`: 表示和管理整个SNN的类，包括其所有组件和仿真逻辑的核心步骤。

## 类详细说明

### `NeuralNetwork`

#### 类名及其构造函数签名

`NeuralNetwork(name="SNN")`

#### 类描述

`NeuralNetwork` 类是构建和仿真SNN的核心。它作为一个中心枢纽，注册并管理多个 `NeuronPopulation` 和 `SynapseCollection` 对象。它还定义了哪些群体作为网络的输入和输出，并提供了一个 `step` 方法来执行单个仿真时间步，包括处理输入、神经元更新、突触传递、学习规则应用和数据记录。

#### 构造函数参数

*   `name` (`str`, 可选, 默认值: `"SNN"`): 神经网络的名称，用于识别。

#### 主要属性/特性

*   `name` (`str`): 网络的名称。
*   `populations` (`dict`): 存储网络中所有神经元群体的字典。键是群体名称 (字符串)，值是 `NeuronPopulation` 对象。
*   `synapses` (`dict`): 存储网络中所有突触集合的字典。键是突触集合名称 (字符串)，值是 `SynapseCollection` 对象。
*   `input_population_names` (`list[str]`): 一个字符串列表，包含被指定为接收外部输入的神经元群体的名称。
*   `output_population_names` (`list[str]`): 一个字符串列表，包含被指定为产生网络行为输出的神经元群体的名称。
*   `probes` (`list[BaseProbe]`): 一个列表，存储附加到此网络的所有探针实例 (`BaseProbe` 的子类实例)。

#### 主要方法

##### 组件管理

*   `add_population(self, population)`:
    *   将一个 `NeuronPopulation` 实例注册到网络中。
    *   **参数**:
        *   `population` (`NeuronPopulation`): 要添加的神经元群体实例。
    *   **异常**: 如果具有相同名称的群体已存在，则引发 `ValueError`。

*   `add_synapses(self, synapse_collection)`:
    *   将一个 `SynapseCollection` 实例注册到网络中。
    *   在添加之前，会检查其突触前和突触后群体是否已在网络中注册。
    *   **参数**:
        *   `synapse_collection` (`SynapseCollection`): 要添加的突触集合实例。
    *   **异常**: 如果具有相同名称的突触集合已存在，或者其关联的神经元群体未在网络中注册，则引发 `ValueError`。

*   `set_input_populations(self, population_names)`:
    *   指定一个或多个神经元群体作为网络的输入接口。
    *   **参数**:
        *   `population_names` (`list[str]`): 一个包含输入群体名称的列表。
    *   **异常**: 如果列表中的任何名称不对应于已注册的群体，则引发 `ValueError`。

*   `set_output_populations(self, population_names)`:
    *   指定一个或多个神经元群体作为网络的输出接口。
    *   **参数**:
        *   `population_names` (`list[str]`): 一个包含输出群体名称的列表。
    *   **异常**: 如果列表中的任何名称不对应于已注册的群体，则引发 `ValueError`。

*   `get_population(self, name)`:
    *   通过名称检索已注册的神经元群体。
    *   **参数**:
        *   `name` (`str`): 要检索的群体名称。
    *   **返回值**: `NeuronPopulation` - 对应的神经元群体实例。
    *   **异常**: 如果找不到指定名称的群体，则引发 `KeyError`。

*   `get_synapses(self, name)`:
    *   通过名称检索已注册的突触集合。
    *   **参数**:
        *   `name` (`str`): 要检索的突触集合名称。
    *   **返回值**: `SynapseCollection` - 对应的突触集合实例。
    *   **异常**: 如果找不到指定名称的突触集合，则引发 `KeyError`。

##### 探针管理

*   `add_probe(self, probe_instance)`:
    *   向网络添加一个数据探针实例 (`BaseProbe` 的子类)。探针用于在仿真过程中记录网络的状态或活动。
    *   **参数**:
        *   `probe_instance` (`BaseProbe`): 要添加的探针实例。
    *   **异常**: 如果 `probe_instance` 不是 `BaseProbe` 的实例，则引发 `TypeError`。

*   `_record_probes(self, current_time)`:
    *   一个内部辅助方法，用于在每个仿真时间步调用所有已注册探针的 `attempt_record` 方法，从而记录数据。
    *   **参数**:
        *   `current_time` (`float`): 当前的仿真时间。

*   `get_probe_data(self, probe_name)`:
    *   获取指定名称的探针所记录的数据。
    *   **参数**:
        *   `probe_name` (`str`): 探针的名称。
    *   **返回值**: `dict` (或其他类型) - 探针记录的数据，具体格式由探针的 `get_data()` 方法定义。
    *   **异常**: 如果找不到指定名称的探针，则引发 `KeyError`。

*   `get_all_probes(self)`:
    *   返回网络中所有已注册探针的列表。
    *   **返回值**: `list[BaseProbe]`。

##### 仿真与控制

*   `step(self, input_currents_map, dt, current_time)`:
    *   执行单个仿真时间步。这是网络仿真的核心方法，按以下顺序执行操作：
        1.  **应用外部输入**: 将 `input_currents_map`中指定的电流注入到对应的输入神经元群体。
        2.  **计算突触电流**: 对于每个突触集合，使用其突触前群体在前一时刻（或本轮更新前）的脉冲活动，计算传递给突触后群体的电流，并累加到总输入电流中。
        3.  **更新神经元状态**: 对网络中的每个神经元群体，使用其接收到的总输入电流 (外部 + 突触) 和时间步长 `dt` 来更新其内部状态 (例如膜电位、恢复变量)，并确定哪些神经元在本时间步发放了脉冲。记录本轮产生的脉冲。
        4.  **应用学习规则**: 对于每个附加了学习规则的突触集合，使用本轮在突触前和突触后群体中新产生的脉冲活动来更新其突触权重。
        5.  **记录探针数据**: 调用 `_record_probes` 方法记录所有探针的数据。
        6.  **收集输出**: 从指定的输出神经元群体收集本轮产生的脉冲状态。
    *   **参数**:
        *   `input_currents_map` (`dict`): 一个字典，键是输入神经元群体的名称 (字符串)，值是注入该群体的电流向量 (`np.ndarray`)。
        *   `dt` (`float`): 仿真时间步长 (ms)。
        *   `current_time` (`float`): 当前仿真时间 (ms)。
    *   **返回值**: `dict` - 一个字典，键是输出神经元群体的名称 (字符串)，值是该群体在本时间步的脉冲状态 (布尔型 `np.ndarray`)。

*   `reset(self)`:
    *   重置网络的状态。这包括：
        *   调用每个神经元群体的 `reset_states()` 方法，将其神经元状态恢复到初始值。
        *   如果突触连接的学习规则有 `reset` 方法，则调用它来重置学习规则的内部状态。
        *   调用每个探针的 `reset()` 方法，清空其已记录的数据。

*   `__repr__(self)`:
    *   返回神经网络对象的字符串表示形式，包括其名称以及注册的群体和突触集合的数量。
    *   **返回值**: `str`。

## (可选) 使用示例

```python
import numpy as np
# 假设 NeuronPopulation, SynapseCollection, TraceSTDP, BaseProbe 已定义
# from dynn.core.neurons import NeuronPopulation, IzhikevichNeuron
# from dynn.core.synapses import SynapseCollection
# from dynn.core.learning_rules import TraceSTDP
# from dynn.utils.probes import BaseProbe # 需要实际的 BaseProbe 定义

from dynn.core.network import NeuralNetwork

# --- 为示例创建虚拟组件 (如果实际类不可用) ---
class DummyNeuron:
    def __init__(self, **kwargs): pass
    def update(self, I_inj, dt): self.fired = np.random.rand() < 0.1 * (1+I_inj); return self.fired
    def get_state(self): return {}
    def reset(self): pass

class DummyPopulation:
    def __init__(self, num_neurons, name="dummy_pop", neuron_model_class=DummyNeuron, **kwargs):
        self.name = name
        self.num_neurons = num_neurons
        self.neurons = [neuron_model_class(**kwargs) for _ in range(num_neurons)]
        self.spike_trace_pre = np.zeros(num_neurons)
        self.spike_trace_post = np.zeros(num_neurons)
    def __len__(self): return self.num_neurons
    def update(self, I_inj_vector, dt, current_time):
        return np.array([n.update(I_inj_vector[i], dt) for i, n in enumerate(self.neurons)])
    def get_spikes(self): return np.array([n.fired for n in self.neurons])
    def reset_states(self): [n.reset() for n in self.neurons]
    def __repr__(self): return f"<DummyPopulation {self.name} ({self.num_neurons})>"

class DummySynapseCollection:
    def __init__(self, pre, post, name="dummy_syn"):
        self.pre_pop = pre; self.post_pop = post; self.name = name
        self.weights = np.random.rand(len(post), len(pre))
        self.connection_mask = np.ones_like(self.weights, dtype=bool)
        self.learning_rule = None
    def get_input_currents(self, pre_spikes): return np.dot(self.weights, pre_spikes.astype(float))
    def apply_learning_rule(self, pre_spikes, post_spikes, dt, current_time): 
        if self.learning_rule: self.learning_rule.update_weights(self,pre_spikes,post_spikes,dt,current_time)
    def set_learning_rule(self, rule): self.learning_rule = rule
    def __repr__(self): return f"<DummySynCollection {self.name}>"

class DummyLearningRule: 
    def update_weights(self, *args): print(f"{self.__class__.__name__} updated weights.")
    def reset(self): pass

class DummyProbe: # Simplified from BaseProbe
    def __init__(self, name, target_obj_name, attribute_or_method, record_interval_ms=1, network_ref=None):
        self.name = name; self.target_obj_name = target_obj_name
        self.attribute_or_method = attribute_or_method; self.record_interval_ms = record_interval_ms
        self.data = []; self.last_record_time = -np.inf; self.network_ref = network_ref
    def attempt_record(self, network, current_time):
        if current_time - self.last_record_time >= self.record_interval_ms:
            # Simplified: directly access network components for demo
            try:
                target_pop = network.get_population(self.target_obj_name)
                if self.attribute_or_method == 'fired': value = target_pop.get_spikes().copy()
                elif self.attribute_or_method == 'v': value = np.array([n.v for n in target_pop.neurons]) # Needs neuron.v
                else: value = None
                self.data.append((current_time, value))
                self.last_record_time = current_time
            except Exception as e: print(f"Probe {self.name} error: {e}")
    def get_data(self): return self.data
    def reset(self): self.data = []; self.last_record_time = -np.inf
# --- 结束虚拟组件定义 ---

# 1. 创建网络
net = NeuralNetwork(name="MySimpleNet")

# 2. 创建并添加神经元群体
input_pop = DummyPopulation(num_neurons=10, name="InputLayer")
hidden_pop = DummyPopulation(num_neurons=20, name="HiddenLayer")
output_pop = DummyPopulation(num_neurons=5, name="OutputLayer")

net.add_population(input_pop)
ordnet.add_population(hidden_pop)
ordnet.add_population(output_pop)

# 3. 创建并添加突触连接
syn_ih = DummySynapseCollection(input_pop, hidden_pop, name="Input_to_Hidden")
syn_ho = DummySynapseCollection(hidden_pop, output_pop, name="Hidden_to_Output")

net.add_synapses(syn_ih)
net.add_synapses(syn_ho)

# 4. (可选) 设置学习规则
stdp = DummyLearningRule() # Replace with actual TraceSTDP if available
syn_ih.set_learning_rule(stdp)
syn_ho.set_learning_rule(stdp)

# 5. 指定输入和输出群体
net.set_input_populations(["InputLayer"])
net.set_output_populations(["OutputLayer"])

# 6. (可选) 添加探针
# 假设 DummyProbe 模拟了 BaseProbe 的行为
spike_probe_output = DummyProbe(name="OutputSpikes", target_obj_name="OutputLayer", attribute_or_method='fired', record_interval_ms=1)
# voltage_probe_hidden = DummyProbe(name="HiddenVoltages", target_obj_name="HiddenLayer", attribute_or_method='v', record_interval_ms=5)

net.add_probe(spike_probe_output)
# net.add_probe(voltage_probe_hidden)

print(net)

# 7. 模拟网络运行一个时间步
dt = 0.5 # ms
current_sim_time = 0.0

# 外部输入电流，只针对InputLayer
external_currents = {
    "InputLayer": np.random.rand(input_pop.num_neurons) * 10 
}

output_activity = net.step(external_currents, dt, current_sim_time)
print(f"Time: {current_sim_time}ms, Output activity: {output_activity["OutputLayer"]}")

current_sim_time += dt
external_currents["InputLayer"] = np.random.rand(input_pop.num_neurons) * 5 # 新的输入
output_activity = net.step(external_currents, dt, current_sim_time)
print(f"Time: {current_sim_time}ms, Output activity: {output_activity["OutputLayer"]}")

# 8. 获取探针数据
sp_data = net.get_probe_data("OutputSpikes")
print(f"Probe data for OutputSpikes (first 5 entries): {sp_data[:5]}")

# 9. 重置网络
net.reset()
print(f"Probe data after reset: {net.get_probe_data('OutputSpikes')}")
``` 