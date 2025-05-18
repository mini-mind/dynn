# `dynn.io.output_decoders` API 文档

## 模块概览

`dynn.io.output_decoders` 模块提供了一系列类，用于将脉冲神经网络 (SNN) 输出神经元的脉冲活动转换为外部世界或高级控制模块可以理解的信号或动作。这些解码器是 SNN 与其执行环境或任务进行交互的关键部分。

主要组件包括：

*   `BaseOutputDecoder`: 所有输出解码器的抽象基类，定义了通用接口。
*   `InstantaneousSpikeCountDecoder`: 基于输出神经元群体在单个时间步的瞬时脉冲发放情况来决定离散动作。
*   `MountainCarActionDecoder`: 针对 OpenAI Gym 的 "MountainCar-v0" 环境定制的输出解码器，它继承自 `InstantaneousSpikeCountDecoder`，将输出神经元的脉冲映射到环境的三个离散动作 (左、无、右)。

## 类详细说明

### `BaseOutputDecoder`

#### 类名及其构造函数签名

`BaseOutputDecoder(source_pop_name, **kwargs)`

#### 类描述

`BaseOutputDecoder` 是所有特定输出解码器类的抽象基类。它定义了输出解码器应具备的基本结构和核心方法 `decode()`，该方法必须由子类实现以定义具体的解码逻辑。

#### 构造函数参数

*   `source_pop_name` (`str`): SNN 中作为输出信号来源的神经元群体的名称。
*   `**kwargs`: 允许子类接收其他特定参数。

#### 主要方法

*   `decode(self, spike_activities_map, dt=None, current_time=None)`:
    *   这是一个抽象方法，必须由子类实现。它负责从指定的输出神经元群体的脉冲活动中提取信息，并将其转换为外部可理解的信号或动作。
    *   **参数**:
        *   `spike_activities_map` (`dict`): 一个字典，其键是 SNN 中神经元群体的名称 (字符串)，值是该群体在当前时间步的脉冲状态 (布尔型 NumPy 数组)。解码器通常只关心 `self.source_pop_name` 对应的数据。
        *   `dt` (`float`, 可选): 当前仿真时间步长。某些时间相关的解码方案可能需要此参数。
        *   `current_time` (`float`, 可选): 当前仿真总时间。某些时间相关的解码方案可能需要此参数。
    *   **返回值**: `any` - 解码后的动作或信号，其类型和值取决于具体的解码器实现和应用场景。
    *   **注意**: 调用此方法时会引发 `NotImplementedError`，除非在子类中被重写。

*   `get_source_population_name(self)`:
    *   返回此解码器实例配置的源神经元群体的名称。
    *   **返回值**: `str` - 源群体的名称。

*   `__repr__(self)`:
    *   返回该解码器实例的字符串表示形式。

---

### `InstantaneousSpikeCountDecoder`

#### 类名及其构造函数签名

`InstantaneousSpikeCountDecoder(source_pop_name, num_actions, default_action=None, **kwargs)`

#### 类描述

`InstantaneousSpikeCountDecoder` 根据输出神经元群体在单个时间步（瞬时）的脉冲发放情况来决定一个离散动作。其核心思想是，如果源群体中的某个神经元发放了脉冲，则选择与该神经元关联的动作。这种解码器适用于那些动作决策可以基于当前时刻网络输出的场景，避免了在一个时间窗口内累积脉冲活动。

默认情况下，如果多个神经元同时发放脉冲，它会选择索引最小的那个发放脉冲的神经元所对应的动作。

#### 构造函数参数

*   `source_pop_name` (`str`): 作为输出信号来源的神经元群体的名称。
*   `num_actions` (`int`): 智能体可能采取的离散动作的总数量。通常，这与源神经元群体中的神经元数量相对应，即每个神经元代表一个特定的动作。
*   `default_action` (`any`, 可选, 默认值: `None`): 如果在当前时间步没有任何源神经元发放脉冲，则返回此默认动作。如果为 `None` 且没有脉冲，`decode` 方法当前返回 `None`。
*   `**kwargs`: 传递给基类构造函数的其他参数。

#### 主要方法

*   `decode(self, spike_activities_map, dt=None, current_time=None)`:
    *   从 `spike_activities_map` 中获取源神经元群体的脉冲状态，并将其解码为一个离散动作。
    *   如果源群体中有神经元发放脉冲，则返回第一个发放脉冲的神经元索引 (被限制在 `0` 到 `num_actions-1` 范围内) 作为动作。
    *   如果没有神经元发放脉冲，则返回 `self.default_action` (如果已设置)，否则返回 `None`。
    *   **参数**: 同 `BaseOutputDecoder.decode`。
    *   **返回值**: `int` 或 `any` - 解码后的离散动作 (通常是整数索引 `0` 到 `num_actions-1`)，或者是 `default_action`，或者 `None`。
    *   **异常**: 如果 `self.source_pop_name` 不在 `spike_activities_map` 中，或脉冲数据格式不正确，则引发 `ValueError`。

*   `__repr__(self)`:
    *   返回该解码器实例的详细字符串表示形式。

---

### `MountainCarActionDecoder`

#### 类名及其构造函数签名

`MountainCarActionDecoder(source_pop_name, num_neurons_for_action=3, default_action_idx=1, **kwargs)`

#### 类描述

`MountainCarActionDecoder` 是一个专门为 OpenAI Gym 的 "MountainCar-v0" 环境设计的输出解码器。它继承自 `InstantaneousSpikeCountDecoder`。该环境有三个离散动作：向左推 (0)，不推 (1)，向右推 (2)。此解码器假定源输出神经元群体有三个神经元，每个神经元分别对应这三个动作中的一个。

#### 构造函数参数

*   `source_pop_name` (`str`): 作为输出信号来源的神经元群体的名称。
*   `num_neurons_for_action` (`int`, 可选, 默认值: `3`): 期望的源神经元群体中的神经元数量。对于 "MountainCar-v0"，这通常是3，对应三个可能的动作。
*   `default_action_idx` (`int`, 可选, 默认值: `1`): 如果没有任何输出神经元发放脉冲，则采取的默认动作的索引。对于 "MountainCar-v0"，默认值 `1` 对应"不推"动作。
*   `**kwargs`: 传递给父类 `InstantaneousSpikeCountDecoder` 构造函数的其他参数。

#### 主要方法

*   `decode(self, spike_activities_map, dt=None, current_time=None)`:
    *   该方法直接继承自 `InstantaneousSpikeCountDecoder`。其行为是：如果源群体的神经元0发放脉冲，则输出动作0；如果神经元1发放，则动作1；如果神经元2发放，则动作2。如果没有脉冲，则输出 `default_action_idx`。如果多个神经元同时发放，则选择索引最小的那个对应的动作。

*   `__repr__(self)`:
    *   返回该解码器实例的详细字符串表示形式。

## (可选) 使用示例

```python
import numpy as np
from dynn.io.output_decoders import InstantaneousSpikeCountDecoder, MountainCarActionDecoder

# 示例 1: InstantaneousSpikeCountDecoder
# 假设输出群体 OutputPopA 有 4 个神经元，对应 4 个动作 (0, 1, 2, 3)
# 如果没有脉冲，默认动作为 -1 (表示无有效动作)
instant_decoder = InstantaneousSpikeCountDecoder(
    source_pop_name="OutputPopA",
    num_actions=4,
    default_action=-1
)
print(instant_decoder)

spike_map1 = {"OutputPopA": np.array([0, 0, 1, 0], dtype=bool)} # 神经元2发放脉冲
action1 = instant_decoder.decode(spike_map1)
print(f"Spikes: {spike_map1['OutputPopA']}, Decoded action: {action1}") # 应该输出 2

spike_map2 = {"OutputPopA": np.array([0, 0, 0, 0], dtype=bool)} # 没有脉冲
action2 = instant_decoder.decode(spike_map2)
print(f"Spikes: {spike_map2['OutputPopA']}, Decoded action: {action2}") # 应该输出 -1

spike_map3 = {"OutputPopA": np.array([1, 0, 1, 0], dtype=bool)} # 神经元0和2发放脉冲
action3 = instant_decoder.decode(spike_map3)
print(f"Spikes: {spike_map3['OutputPopA']}, Decoded action: {action3}") # 应该输出 0 (第一个)

# 示例 2: MountainCarActionDecoder
# 假设输出群体 OutputPopMC 有 3 个神经元
mc_decoder = MountainCarActionDecoder(
    source_pop_name="OutputPopMC",
    default_action_idx=1 # 无操作
)
print(mc_decoder)

mc_spike_map1 = {"OutputPopMC": np.array([1, 0, 0], dtype=bool)} # 神经元0 (左推)
action_mc1 = mc_decoder.decode(mc_spike_map1)
print(f"MC Spikes: {mc_spike_map1['OutputPopMC']}, Decoded MC action: {action_mc1}") # 应该输出 0

mc_spike_map2 = {"OutputPopMC": np.array([0, 0, 0], dtype=bool)} # 没有脉冲
action_mc2 = mc_decoder.decode(mc_spike_map2)
print(f"MC Spikes: {mc_spike_map2['OutputPopMC']}, Decoded MC action: {action_mc2}") # 应该输出 1 (默认)

mc_spike_map3 = {"OutputPopMC": np.array([0, 0, 1], dtype=bool)} # 神经元2 (右推)
action_mc3 = mc_decoder.decode(mc_spike_map3)
print(f"MC Spikes: {mc_spike_map3['OutputPopMC']}, Decoded MC action: {action_mc3}") # 应该输出 2

``` 