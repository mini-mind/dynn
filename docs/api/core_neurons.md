# `dynn.core.neurons` API 文档

## 模块概览

`dynn.core.neurons` 模块负责定义单个神经元的模型以及管理神经元群体。它提供了构建神经网络基本计算单元的构建块。

主要组件包括：

*   `IzhikevichNeuron`: 一个可参数化的类Izhikevich尖峰神经元模型。
*   `NeuronPopulation`: 用于管理一组神经元对象，并提供对整个群体进行操作的便捷方法。

## 类详细说明

### `IzhikevichNeuron`

#### 类名及其构造函数签名

`IzhikevichNeuron(a=0.02, b=0.2, c=-65.0, d=8.0, v_thresh=30.0, initial_v=-70.0, initial_u=None)`

#### 类描述

`IzhikevichNeuron` 类实现了一个广泛使用的尖峰神经元模型，由 Eugene M. Izhikevich 于2003年提出。该模型以其计算效率和能够再现多种神经元放电模式的能力而闻名。此类允许用户通过参数调整来配置神经元的动态行为。

#### 构造函数参数

*   `a` (`float`, 可选, 默认值: `0.02`): 恢复变量 `u` 的时间尺度参数。较小的值会导致恢复速度变慢。
*   `b` (`float`, 可选, 默认值: `0.2`): 恢复变量 `u` 对膜电位 `v` 的敏感度参数。较大的值会使 `u` 对 `v` 的亚阈值波动更敏感。
*   `c` (`float`, 可选, 默认值: `-65.0`): 脉冲发放后膜电位 `v` 的重置值 (mV)。
*   `d` (`float`, 可选, 默认值: `8.0`): 脉冲发放后恢复变量 `u` 的重置增加值。
*   `v_thresh` (`float`, 可选, 默认值: `30.0`): 神经元发放脉冲的膜电位阈值 (mV)。
*   `initial_v` (`float`, 可选, 默认值: `-70.0`): 神经元的初始膜电位 (mV)。
*   `initial_u` (`float`, 可选, 默认值: `None`): 神经元的初始恢复变量。如果为 `None`，则会根据 `b * initial_v` 计算得到。

#### 主要属性/特性

*   `v` (`float`): 神经元的当前膜电位 (mV)。
*   `u` (`float`): 神经元的当前恢复变量。
*   `a`, `b`, `c`, `d` (`float`): 控制神经元动态的Izhikevich模型参数。
*   `v_thresh` (`float`): 脉冲发放阈值。
*   `fired` (`bool`): 表示在当前时间步神经元是否发放了脉冲。
*   `last_spike_time` (`float`): 神经元上次发放脉冲的时间。初始为 `-np.inf`。

#### 主要方法

*   `update(self, I_inj, dt)`:
    *   根据注入电流 `I_inj` 和时间步长 `dt` 更新神经元的状态 (膜电位 `v` 和恢复变量 `u`)。
    *   **参数**:
        *   `I_inj` (`float`): 在当前时间步注入到神经元的电流。
        *   `dt` (`float`): 仿真时间步长 (ms)。
    *   **返回值**: `bool` - 如果神经元在此次更新后发放了脉冲，则返回 `True`，否则返回 `False`。

*   `get_state(self)`:
    *   返回一个包含神经元关键内部状态变量的字典。
    *   **返回值**: `dict` - 包含键 `"v"`, `"u"`, `"fired"`, `"last_spike_time"` 及其对应值的字典。

*   `reset(self, initial_v=-70.0, initial_u=None)`:
    *   将神经元的状态重置为其初始值。
    *   **参数**:
        *   `initial_v` (`float`, 可选, 默认值: `-70.0`): 重置后的初始膜电位。
        *   `initial_u` (`float`, 可选, 默认值: `None`): 重置后的初始恢复变量。如果为 `None`，则根据 `b * initial_v` 计算。

---

### `NeuronPopulation`

#### 类名及其构造函数签名

`NeuronPopulation(num_neurons, neuron_model_class=IzhikevichNeuron, neuron_params=None, initial_v_dist=None, initial_u_dist=None, name="population")`

#### 类描述

`NeuronPopulation` 类用于创建和管理一组神经元。它提供了一个方便的接口来对整个神经元群体执行操作，例如初始化参数、更新状态、收集脉冲等。这个类抽象了对单个神经元的操作，使得构建和管理大规模神经网络更加容易。

#### 构造函数参数

*   `num_neurons` (`int`): 群体中神经元的数量。
*   `neuron_model_class` (`class`, 可选, 默认值: `IzhikevichNeuron`): 用于创建群体中每个神经元的类。必须是一个神经元模型类，例如 `IzhikevichNeuron`。
*   `neuron_params` (`dict` 或 `list[dict]`, 可选, 默认值: `None`):
    *   单个神经元模型的参数。
    *   如果提供一个 `dict`，则群体中的所有神经元将共享这些参数。
    *   如果提供一个 `list[dict]`，列表的长度必须等于 `num_neurons`，每个字典对应一个神经元的参数。
    *   如果为 `None`，则所有神经元将使用 `neuron_model_class` 的默认参数。
*   `initial_v_dist` (`tuple` 或 `float`, 可选, 默认值: `None`):
    *   膜电位 `v` 的初始值或其分布方式。
    *   如果是一个 `float`，所有神经元的初始 `v` 都设置为该值。
    *   如果是一个 `tuple`，用于指定分布类型及其参数，例如：
        *   `('uniform', (low, high))`: 从 `low` 到 `high` 的均匀分布中采样。
        *   `('normal', (mean, std))`: 从均值为 `mean`、标准差为 `std` 的正态分布中采样。
    *   如果为 `None`，将使用神经元模型或 `neuron_params` 中指定的 `initial_v`，如果都没有则使用神经元模型的默认初始 `v`。
*   `initial_u_dist` (`tuple` 或 `float`, 可选, 默认值: `None`):
    *   恢复变量 `u` 的初始值或其分布方式。
    *   如果是一个 `float`，所有神经元的初始 `u` 都设置为该值。
    *   如果是一个 `tuple` (格式同 `initial_v_dist`)。
    *   如果为 `None`，将使用神经元模型或 `neuron_params` 中指定的 `initial_u`；如果都没有，则根据对应神经元的 `b` 参数和初始 `v` 计算。
*   `name` (`str`, 可选, 默认值: `"population"`): 神经元群体的名称，方便识别。

#### 主要属性/特性

*   `num_neurons` (`int`): 群体中的神经元数量。
*   `name` (`str`): 群体的名称。
*   `neurons` (`list`): 包含群体中所有神经元实例的列表。
*   `spike_trace_pre` (`np.ndarray`): 长度为 `num_neurons` 的数组，存储每个神经元的突触前脉冲迹 (通常用于STDP)。
*   `spike_trace_post` (`np.ndarray`): 长度为 `num_neurons` 的数组，存储每个神经元的突触后脉冲迹 (通常用于STDP)。

#### 主要方法

*   `update(self, I_inj_vector, dt, current_time)`:
    *   更新群体中所有神经元的状态。
    *   **参数**:
        *   `I_inj_vector` (`np.ndarray`): 长度为 `num_neurons` 的一维数组，包含每个神经元在当前时间步接收到的注入电流。
        *   `dt` (`float`): 仿真时间步长 (ms)。
        *   `current_time` (`float`): 当前仿真时间 (ms)。该时间会用于更新发放脉冲神经元的 `last_spike_time`。
    *   **返回值**: `np.ndarray` - 一个布尔类型的数组，长度为 `num_neurons`，其中 `True` 表示对应索引的神经元在当前时间步发放了脉冲。

*   `get_spikes(self)`:
    *   获取群体中所有神经元在当前时间步的脉冲发放状态。
    *   **返回值**: `np.ndarray` - 一个布尔类型的数组，长度为 `num_neurons`，`True` 表示对应神经元发放了脉冲。

*   `get_all_states(self, state_keys=None)`:
    *   获取群体中所有神经元指定的内部状态变量。
    *   **参数**:
        *   `state_keys` (`list[str]`, 可选, 默认值: `None`): 一个字符串列表，指定需要获取的状态变量的名称 (例如 `["v", "u"]`)。如果为 `None`，则获取神经元模型 `get_state()` 方法返回的所有状态。
    *   **返回值**: `dict` - 字典的键是状态名称 (字符串)，值是 `np.ndarray`，包含群体中所有神经元对应的状态值。

*   `set_parameters(self, neuron_indices, param_name, param_value)`:
    *   为群体中的一个或多个指定神经元设置参数。
    *   (注意：此方法在之前的文件阅读中未完全显示，具体参数和行为可能需要参考完整代码或进一步确认。)
    *   **参数**:
        *   `neuron_indices` (`int` 或 `list[int]` 或 `np.ndarray`): 要修改参数的神经元的索引（或索引列表/数组）。
        *   `param_name` (`str`): 要修改的参数名称 (例如 `'a'`, `'b'`, `'c'`, `'d'`)。
        *   `param_value` (`any`): 要设置的参数的新值。如果 `neuron_indices` 是多个索引，`param_value` 可以是一个单一值（应用于所有指定神经元）或一个与 `neuron_indices` 长度相同的值列表/数组。

*   `reset_states(self, initial_v_dist=None, initial_u_dist=None)`:
    *   重置群体中所有神经元的状态到初始值或指定的分布。
    *   (注意：此方法在之前的文件阅读中未完全显示，具体参数和行为可能需要参考完整代码或进一步确认。)
    *   **参数**:
        *   `initial_v_dist` (`tuple` 或 `float`, 可选, 默认值: `None`): 格式同构造函数中的 `initial_v_dist`。如果提供，则用于重置膜电位 `v`。
        *   `initial_u_dist` (`tuple` 或 `float`, 可选, 默认值: `None`): 格式同构造函数中的 `initial_u_dist`。如果提供，则用于重置恢复变量 `u`。

*   `__len__(self)`:
    *   返回群体中的神经元数量。
    *   **返回值**: `int`

*   `__getitem__(self, index)`:
    *   通过索引获取群体中的单个神经元对象。
    *   **参数**:
        *   `index` (`int`): 神经元的索引。
    *   **返回值**: `object` - 对应索引的神经元实例。

*   `__repr__(self)`:
    *   返回神经元群体的字符串表示形式。
    *   **返回值**: `str` - 例如 `<NeuronPopulation: my_pop (100 neurons)>`

## (可选) 使用示例

```python
from dynn.core.neurons import IzhikevichNeuron, NeuronPopulation
import numpy as np

# 示例1: 创建单个Izhikevich神经元并更新
neuron = IzhikevichNeuron(a=0.02, b=0.25, c=-60, d=6, initial_v=-65)
print(f"Initial state: v={neuron.v:.2f}, u={neuron.u:.2f}")

# 模拟注入电流并更新
dt = 0.5 # ms
for t in range(100): # 模拟100个时间步
    I_input = 10 if t > 20 and t < 80 else 0 # 模拟一个电流脉冲
    fired = neuron.update(I_input, dt)
    if fired:
        print(f"Time {t*dt:.1f}ms: Spike! v_reset={neuron.v:.2f}, u_new={neuron.u:.2f}")
# print(f"Final state: v={neuron.v:.2f}, u={neuron.u:.2f}")
# print(neuron.get_state())

# 示例2: 创建一个神经元群体
pop_size = 10
# 为群体中的奇数和偶数神经元设置不同的参数b
neuron_params_list = []
for i in range(pop_size):
    if i % 2 == 0:
        neuron_params_list.append({'b': 0.2, 'd': 8}) # RS-like
    else:
        neuron_params_list.append({'b': 0.25, 'd': 2}) # FS-like

# 初始化v为-65mV，u从均匀分布U(-15, -10)采样
pop1 = NeuronPopulation(
    num_neurons=pop_size,
    neuron_model_class=IzhikevichNeuron,
    neuron_params=neuron_params_list,
    initial_v_dist=-65.0, # 所有神经元v=-65
    initial_u_dist=('uniform', (-15, -10)), # u从均匀分布采样
    name="MixedPopulation"
)
print(pop1)
# print(f"Initial v for pop1: {pop1.get_all_states(['v'])['v']}")
# print(f"Initial u for pop1: {pop1.get_all_states(['u'])['u']}")


# 更新群体状态
current_input_to_pop1 = np.random.rand(pop_size) * 15 # 随机电流输入
current_time_ms = 0.0
for _ in range(50): # 模拟50步
    fired_mask = pop1.update(current_input_to_pop1, dt, current_time_ms)
    if np.any(fired_mask):
        print(f"Time {current_time_ms:.1f}ms - Spikes from neurons: {np.where(fired_mask)[0]}")
    # 更新输入，或根据其他逻辑设置
    current_input_to_pop1 = np.random.rand(pop_size) * (10 + 5 * np.sin(current_time_ms/10.0)) 
    current_time_ms += dt

# 获取所有神经元的膜电位
all_v = pop1.get_all_states(state_keys=["v"])
# print(f"All v values: {all_v['v']}")

# 获取单个神经元
neuron_5 = pop1[5]
# print(f"State of neuron 5: {neuron_5.get_state()}")
``` 