# `dynn.core.simulator` API 文档

## 模块概览

`dynn.core.simulator` 模块提供了 `Simulator` 类，它是 DyNN 框架的离散时间仿真引擎。该引擎负责驱动神经网络的仿真过程，管理时间步进，并协调网络状态的更新。

主要组件包括：

*   `Simulator`: 控制仿真循环，按时间步长迭代执行网络更新的类。

## 类详细说明

### `Simulator`

#### 类名及其构造函数签名

`Simulator(network, dt=1.0)`

#### 类描述

`Simulator` 类是 DyNN 仿真的主控制器。它接收一个 `NeuralNetwork` 实例和仿真时间步长 `dt` 作为输入。它提供了多种运行仿真的方法，例如运行单个时间步、运行指定数量的时间步或运行指定的总时长。在每个时间步中，它会调用 `NeuralNetwork` 对象的 `step` 方法来更新网络状态。它还支持通过回调函数和输入生成函数与外部环境或用户逻辑进行交互。

#### 构造函数参数

*   `network` (`NeuralNetwork`): 要进行仿真的 `NeuralNetwork` 对象实例。
*   `dt` (`float`, 可选, 默认值: `1.0`): 仿真时间步长。单位通常是毫秒 (ms)，应与网络模型中参数的时间尺度一致。

#### 主要属性/特性

*   `network` (`NeuralNetwork`): 仿真器管理的神经网络对象。
*   `dt` (`float`): 仿真时间步长。
*   `current_time` (`float`): 当前仿真进行到的总时间。初始为 `0.0`，并在每个 `run_step` 后增加 `dt`。
*   `total_steps_run` (`int`): 自上次重置以来，仿真器已执行的总时间步数。

#### 主要方法

*   `run_step(self, input_currents_map)`:
    *   执行单个仿真时间步。
    *   此方法会调用其管理的 `self.network` 对象的 `step` 方法，并传递当前的 `input_currents_map`、`self.dt` 和 `self.current_time`。
    *   之后，它会将 `self.current_time` 增加 `self.dt`，并递增 `self.total_steps_run`。
    *   **参数**:
        *   `input_currents_map` (`dict`): 一个字典，其键是网络中输入神经元群体的名称 (字符串)，值是注入该群体的电流向量 (`np.ndarray`)。
    *   **返回值**: `dict` - 从 `self.network.step` 返回的输出脉冲映射，键是输出群体名称，值是该群体的脉冲状态 (布尔数组)。

*   `run_n_steps(self, num_steps, input_generator_fn=None, callback_fn=None, stop_condition_fn=None)`:
    *   执行指定数量 (`num_steps`) 的仿真步骤。
    *   在每个步骤中：
        1.  如果提供了 `input_generator_fn`，则调用它来获取当前步骤的输入电流。
        2.  调用 `run_step` 执行仿真。
        3.  如果提供了 `callback_fn`，则在步骤完成后调用它。
        4.  如果提供了 `stop_condition_fn`，则检查其返回值，若为 `True` 则提前终止仿真。
    *   **参数**:
        *   `num_steps` (`int`): 要执行的仿真步数。
        *   `input_generator_fn` (`callable`, 可选): 一个函数，签名应为 `fn(current_time, dt, previous_step_outputs)`。它应返回一个 `input_currents_map` 字典。`previous_step_outputs` 是上一步 `network.step` 的返回结果。如果为 `None`，则假定输入群体没有外部电流注入。
        *   `callback_fn` (`callable`, 可选): 一个函数，签名应为 `fn(current_time, current_inputs, current_outputs, simulator_instance)`。在每个仿真步骤之后被调用，可用于数据记录、与环境交互、打印状态等。
        *   `stop_condition_fn` (`callable`, 可选): 一个函数，签名应为 `fn(current_time, current_outputs, simulator_instance)`。在每个步骤后检查，如果返回 `True`，则仿真会提前停止。
    *   **返回值**: `dict` 或 `None` - 最后一个仿真步骤的网络输出。如果仿真由于某种原因未能运行（例如 `num_steps` <= 0），可能返回 `None` 或初始的 `last_outputs` 值。

*   `run_for_duration(self, total_duration, input_generator_fn=None, callback_fn=None, stop_condition_fn=None)`:
    *   执行仿真，直到累计的仿真时间 (`self.current_time`) 达到或超过 `total_duration`。
    *   它首先根据 `total_duration` 和 `self.dt` 计算出需要运行的总步数，然后调用 `run_n_steps` 来执行这些步骤。
    *   **参数**:
        *   `total_duration` (`float`): 总仿真时长 (单位与 `dt` 相同，通常是毫秒)。
        *   `input_generator_fn` (`callable`, 可选): 同 `run_n_steps`。
        *   `callback_fn` (`callable`, 可选): 同 `run_n_steps`。
        *   `stop_condition_fn` (`callable`, 可选): 同 `run_n_steps`。
    *   **返回值**: `dict` 或 `None` - 最后一个仿真步骤的网络输出。
    *   **注意**: 如果 `total_duration` 小于 `dt`，可能会打印警告，但仍会尝试运行至少匹配目标时长的步数。

*   `reset(self, reset_time=True)`:
    *   重置仿真器和其管理的神经网络的状态。
    *   具体操作包括：
        *   调用 `self.network.reset()` 来重置所有神经元群体的状态、学习规则的内部状态（如果适用）以及所有探针的数据。
        *   如果 `reset_time` 为 `True` (默认)，则将 `self.current_time` 重置为 `0.0`。
        *   将 `self.total_steps_run` 重置为 `0`。
    *   **参数**:
        *   `reset_time` (`bool`, 可选, 默认值: `True`): 是否将当前仿真时间 (`self.current_time`) 也重置为 `0.0`。

*   `__repr__(self)`:
    *   返回仿真器对象的字符串表示形式，包括其关联的网络名称、`dt`、当前仿真时间和已运行的总步数。
    *   **返回值**: `str`。

## (可选) 使用示例

```python
import numpy as np
# 假设 NeuralNetwork, NeuronPopulation 等核心类已定义
# from dynn.core.network import NeuralNetwork
# (需要 dynn.core.network.NeuralNetwork 和其依赖的虚拟或真实组件)
from dynn.core.simulator import Simulator

# --- 为示例创建虚拟组件 (如果实际类不可用) ---
class DummyPopulation:
    def __init__(self, num_neurons, name="dummy_pop"):
        self.name = name; self.num_neurons = num_neurons; self.size = num_neurons
    def __len__(self): return self.num_neurons
    def reset_states(self): print(f"Population {self.name} reset.")

class DummyNetwork:
    def __init__(self, name="dummy_net"):
        self.name = name
        self.input_population_names = ["InputPop"]
        self.output_population_names = ["OutputPop"]
        self._populations = { # 内部使用，get_population 会访问它
            "InputPop": DummyPopulation(10, "InputPop"),
            "OutputPop": DummyPopulation(5, "OutputPop")
        }
    def step(self, dt, time_elapsed, inputs):
        print(f"Network {self.name} step at time {time_elapsed} with dt {dt}. Inputs: {list(inputs.keys())}")
        # 模拟输出
        output_pop_name = self.output_population_names[0]
        num_output_neurons = len(self._populations[output_pop_name])
        return {output_pop_name: np.random.rand(num_output_neurons) < 0.2}
    def reset(self): 
        print(f"Network {self.name} reset.")
        for pop in self._populations.values(): pop.reset_states()
    def get_population(self, name): return self_populations[name]
# --- 结束虚拟组件定义 ---

# 1. 创建一个网络实例 (使用虚拟网络)
my_network = DummyNetwork(name="TestNet")

# 2. 创建仿真器实例
sim = Simulator(network=my_network, dt=0.1) # dt = 0.1 ms
print(sim)

# 3. 运行单个步骤
# 假设 TestNet 有一个名为 "InputPop" 的输入群体
initial_inputs = {"InputPop": np.random.rand(10) * 5} # 10 个输入神经元的电流
print(f"\nRunning single step from t={sim.current_time:.2f}ms...")
outputs_step1 = sim.run_step(initial_inputs)
print(f"Outputs after step 1 (t={sim.current_time:.2f}ms): {outputs_step1}")
print(sim)

# 4. 定义输入生成器和回调函数 (可选)
def my_input_generator(current_time, dt, previous_outputs):
    print(f"  Input generator called at t={current_time:.2f}ms. Prev outputs: {previous_outputs}")
    # 生成变化的输入
    new_inputs = np.sin(current_time / 10.0) * 5 + np.random.rand(10) * 2
    return {"InputPop": new_inputs}

def my_callback(current_time, current_inputs, current_outputs, simulator_instance):
    print(f"  Callback at t={current_time:.2f}ms. Outputs: {current_outputs["OutputPop"][:2]}...")
    if current_time > 0.8: # 示例：记录一些东西或检查条件
        pass 

def my_stop_condition(current_time, current_outputs, simulator_instance):
    if current_time >= 0.5: # 提前停止条件
        print(f"  Stop condition met at t={current_time:.2f}ms.")
        return True
    return False

# 5. 运行指定数量的步骤 (例如10步)
print(f"\nRunning for 10 steps from t={sim.current_time:.2f}ms...")
sim.reset(reset_time=True) # 重置时间以便从0开始这个演示
last_outputs_n_steps = sim.run_n_steps(
    num_steps=10,
    input_generator_fn=my_input_generator,
    callback_fn=my_callback,
    stop_condition_fn=my_stop_condition # 会在 t=0.5ms 时停止
)
print(f"Finished run_n_steps. Last outputs: {last_outputs_n_steps}")
print(sim)

# 6. 运行指定总时长 (例如1.5ms)
print(f"\nRunning for a total duration of 1.5ms from t={sim.current_time:.2f}ms...")
sim.reset(reset_time=True) # 重置时间
last_outputs_duration = sim.run_for_duration(
    total_duration=1.5, # ms
    input_generator_fn=my_input_generator,
    callback_fn=my_callback,
    stop_condition_fn=None # 这次不提前停止
)
print(f"Finished run_for_duration. Last outputs: {last_outputs_duration}")
print(sim)

# 7. 重置仿真器
print("\nResetting simulator...")
sim.reset()
print(sim)

``` 