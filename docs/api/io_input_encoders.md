# `dynn.io.input_encoders` API 文档

## 模块概览

`dynn.io.input_encoders` 模块提供了一系列类，用于将来自外部环境或数据源的原始观察值转换为脉冲神经网络 (SNN) 输入神经元可以理解的信号，通常是注入电流或目标脉冲模式。这些编码器是连接 SNN 与外部世界的桥梁。

主要组件包括：

*   `BaseInputEncoder`: 所有输入编码器的抽象基类，定义了通用接口。
*   `GaussianEncoder`: 使用一组高斯感受野将一维标量值编码为神经元群体的注入电流。
*   `DirectCurrentInjector`: 直接将观察向量（或其一部分）按比例作为电流注入目标神经元群体。
*   `MountainCarPositionEncoder`: 针对 OpenAI Gym 的 "MountainCar-v0" 环境定制的编码器，它继承自 `GaussianEncoder`，仅使用小车的位置信息进行编码。

## 类详细说明

### `BaseInputEncoder`

#### 类名及其构造函数签名

`BaseInputEncoder(target_pop_name, **kwargs)`

#### 类描述

`BaseInputEncoder` 是所有特定输入编码器类的抽象基类。它定义了输入编码器应具备的基本结构和核心方法 `encode()`，该方法必须由子类实现以定义具体的编码逻辑。

#### 构造函数参数

*   `target_pop_name` (`str`): 将接收编码后输入的神经元群体的名称。
*   `**kwargs`: 允许子类接收其他特定参数。

#### 主要方法

*   `encode(self, observation, dt=None, current_time=None)`:
    *   这是一个抽象方法，必须由子类实现。它负责将外部观察值 `observation` 转换为SNN输入神经元的活动（通常是注入电流）。
    *   **参数**:
        *   `observation`: 来自环境的原始观察值。其具体格式取决于环境和编码器类型。
        *   `dt` (`float`, 可选): 当前仿真时间步长。某些时间相关的编码方案可能需要此参数。
        *   `current_time` (`float`, 可选): 当前仿真总时间。某些时间相关的编码方案可能需要此参数。
    *   **返回值**: `dict` - 一个字典，键是目标输入群体的名称 (即 `self.target_pop_name`)，值是表示注入电流的 NumPy 数组。
    *   **注意**: 调用此方法时会引发 `NotImplementedError`，除非在子类中被重写。

*   `get_target_population_name(self)`:
    *   返回此编码器实例配置的目标神经元群体的名称。
    *   **返回值**: `str` - 目标群体的名称。

*   `__repr__(self)`:
    *   返回该编码器实例的字符串表示形式。

---

### `GaussianEncoder`

#### 类名及其构造函数签名

`GaussianEncoder(target_pop_name, num_neurons, min_val, max_val, sigma_scale=0.1, current_amplitude=10.0, **kwargs)`

#### 类描述

`GaussianEncoder` 使用一组高斯感受野 (Gaussian receptive fields) 将一个一维标量输入值编码为目标神经元群体中各个神经元的注入电流。每个神经元对输入值的一个特定范围敏感，其激活强度（即注入电流大小）呈高斯曲线状，峰值位于该神经元的"首选"输入值处。这种编码方式常用于表示连续变量。

#### 构造函数参数

*   `target_pop_name` (`str`): 目标输入神经元群体的名称。
*   `num_neurons` (`int`): 目标群体中的神经元数量。这些神经元的高斯感受野中心将均匀分布在 `[min_val, max_val]` 范围内。
*   `min_val` (`float`): 期望编码的输入值的最小值。
*   `max_val` (`float`): 期望编码的输入值的最大值。
*   `sigma_scale` (`float`, 可选, 默认值: `0.1`): 控制高斯感受野宽度的参数。实际的高斯标准差 `sigma` 计算为 `sigma_scale * (max_val - min_val)`。较小的 `sigma_scale` 导致更窄的感受野，即神经元对输入值的选择性更高。
*   `current_amplitude` (`float`, 可选, 默认值: `10.0`): 当输入值恰好位于神经元高斯感受野中心时，注入该神经元的最大电流幅值。
*   `**kwargs`: 传递给基类构造函数的其他参数。

#### 主要方法

*   `encode(self, observation_value, dt=None, current_time=None)`:
    *   将单个标量观察值 `observation_value` 编码为一组注入电流。
    *   **参数**:
        *   `observation_value` (`float` or `int` or single-element list/array): 要编码的标量值。
    *   **返回值**: `dict` - `{self.target_pop_name: currents_array}`，其中 `currents_array` 是一个长度为 `num_neurons` 的 NumPy 数组，表示每个神经元接收到的注入电流。

*   `__repr__(self)`:
    *   返回该编码器实例的详细字符串表示形式。

---

### `DirectCurrentInjector`

#### 类名及其构造函数签名

`DirectCurrentInjector(target_pop_name, num_neurons, observation_slice=None, scale_factor=1.0, **kwargs)`

#### 类描述

`DirectCurrentInjector` 提供了一种直接的编码方式，它将输入的观察向量（或其一部分）经过可选的缩放后，直接作为电流注入到目标神经元群体的每个神经元中。这种编码器适用于观察空间的维度与目标神经元群体大小相匹配，或者可以进行切片和缩放以匹配的情况。

#### 构造函数参数

*   `target_pop_name` (`str`): 目标输入神经元群体的名称。
*   `num_neurons` (`int`): 目标群体中的神经元数量。编码器期望处理后的观察向量维度与此数量匹配。
*   `observation_slice` (`slice`, 可选, 默认值: `None`): 一个 Python `slice` 对象，用于从输入的观察向量中提取相关的部分。例如，`slice(0, 10)` 会选择观察向量的前10个元素。如果为 `None`，则使用整个观察向量。
*   `scale_factor` (`float` 或 `np.ndarray`, 可选, 默认值: `1.0`): 一个乘数或一组乘数，用于缩放从观察中选定的值。如果是一个浮点数，则所有选定的观察值都会乘以该因子。如果是一个 NumPy 数组，其长度必须与切片后的观察值数量（即 `num_neurons`）相同，用于对每个元素进行独立的缩放。
*   `**kwargs`: 传递给基类构造函数的其他参数。

#### 主要方法

*   `encode(self, observation, dt=None, current_time=None)`:
    *   将观察向量（或其切片）编码为注入电流。
    *   **参数**:
        *   `observation` (`np.ndarray` 或 `list`): 输入的观察向量。
    *   **返回值**: `dict` - `{self.target_pop_name: currents_array}`，其中 `currents_array` 是一个长度为 `num_neurons` 的 NumPy 数组，表示注入电流。
    *   **异常**: 如果处理后的观察维度与 `num_neurons` 不匹配，则引发 `ValueError`。

*   `__repr__(self)`:
    *   返回该编码器实例的详细字符串表示形式。

---

### `MountainCarPositionEncoder`

#### 类名及其构造函数签名

`MountainCarPositionEncoder(target_pop_name, num_neurons, pos_min=-1.2, pos_max=0.6, sigma_scale=0.1, current_amplitude=10.0, **kwargs)`

#### 类描述

`MountainCarPositionEncoder` 是一个专门为 OpenAI Gym 中的 "MountainCar-v0" 环境设计的输入编码器。它继承自 `GaussianEncoder`，并根据 "MountainCar-v0" 环境的特定需求进行配置：它只使用观察向量中的第一个元素（即小车的位置信息）进行高斯编码，忽略速度信息。

#### 构造函数参数

*   `target_pop_name` (`str`): 目标输入神经元群体的名称。
*   `num_neurons` (`int`): 目标群体中的神经元数量。
*   `pos_min` (`float`, 可选, 默认值: `-1.2`): 小车可能位置的最小值 (根据 "MountainCar-v0" 环境定义)。
*   `pos_max` (`float`, 可选, 默认值: `0.6`): 小车可能位置的最大值 (根据 "MountainCar-v0" 环境定义)。
*   `sigma_scale` (`float`, 可选, 默认值: `0.1`): 同 `GaussianEncoder` 中的定义，用于计算高斯感受野的宽度。
*   `current_amplitude` (`float`, 可选, 默认值: `10.0`): 同 `GaussianEncoder` 中的定义，表示最大注入电流幅值。
*   `**kwargs`: 传递给父类 `GaussianEncoder` 构造函数的其他参数。

#### 主要方法

*   `encode(self, observation, dt=None, current_time=None)`:
    *   从 "MountainCar-v0" 环境的观察向量中提取位置信息，并使用父类 `GaussianEncoder` 的逻辑将其编码为注入电流。
    *   **参数**:
        *   `observation` (`list` 或 `np.ndarray`): "MountainCar-v0" 环境的观察向量，通常格式为 `[position, velocity]`。
    *   **返回值**: `dict` - `{self.target_pop_name: currents_array}`，格式同 `GaussianEncoder.encode`。
    *   **异常**: 如果观察格式不正确（例如，不是列表或数组，或者长度不足），则引发 `ValueError`。

*   `__repr__(self)`:
    *   返回该编码器实例的详细字符串表示形式。

## (可选) 使用示例

```python
import numpy as np
from dynn.io.input_encoders import GaussianEncoder, DirectCurrentInjector, MountainCarPositionEncoder

# 示例 1: GaussianEncoder
gaussian_enc = GaussianEncoder(
    target_pop_name="InputPopulationA",
    num_neurons=10,
    min_val=0.0,
    max_val=1.0,
    sigma_scale=0.15,
    current_amplitude=15.0
)
print(gaussian_enc)
currents_A = gaussian_enc.encode(observation_value=0.5)
print(f"Currents for obs=0.5: {currents_A['InputPopulationA']}")
currents_B = gaussian_enc.encode(observation_value=0.0)
print(f"Currents for obs=0.0: {currents_B['InputPopulationA']}")

# 示例 2: DirectCurrentInjector
direct_enc = DirectCurrentInjector(
    target_pop_name="InputPopulationB",
    num_neurons=5,
    observation_slice=slice(2, 7), # 从观察中取索引2到6的元素
    scale_factor=2.0
)
print(direct_enc)
observation_vector = np.array([10, 20, 1, 2, 3, 4, 5, 60, 70])
currents_C = direct_enc.encode(observation_vector)
print(f"Selected obs: {observation_vector[slice(2,7)]}, Currents: {currents_C['InputPopulationB']}")

# 示例 3: MountainCarPositionEncoder
mc_pos_enc = MountainCarPositionEncoder(
    target_pop_name="InputPopulationMC",
    num_neurons=20,
    pos_min=-1.2, 
    pos_max=0.6,
    sigma_scale=0.1,
    current_amplitude=12.0
)
print(mc_pos_enc)
# MountainCar-v0 observation: [position, velocity]
mc_observation = [-0.5, 0.02] # 小车在位置-0.5，速度0.02
currents_MC = mc_pos_enc.encode(mc_observation)
print(f"Currents for MC obs ({mc_observation}): {currents_MC['InputPopulationMC']}")
``` 