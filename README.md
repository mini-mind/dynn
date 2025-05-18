# DyNN: 动态脉冲神经网络仿真框架

DyNN (Dynamic Neural Networks) 是一个使用 Python 从零开始构建的脉冲神经网络 (SNN) 仿真框架。它旨在提供一个模块化、灵活且可配置的平台，用于研究和开发基于 SNN 的智能体，特别是那些采用尖峰时间依赖可塑性 (STDP) 并结合奖励调制进行学习的智能体。

## 主要特性

*   **模块化设计**: 清晰分离的核心组件，包括神经元模型、突触连接、学习规则、网络管理和仿真引擎。
*   **可配置性**: 通过结构化的配置文件管理SNN模型参数、连接拓扑、学习规则参数、输入/输出方案及仿真参数。
*   **核心SNN组件** (`dynn.core`):
    *   可参数化的神经元模型 (例如，类Izhikevich模型)。
    *   高效的突触连接管理，支持权重矩阵和灵活的拓扑结构配置。
    *   实现的学习规则，如奖励调制的STDP。
    *   统一的网络对象和离散时间仿真引擎。
*   **输入/输出接口** (`dynn.io`):
    *   灵活的输入编码器，将外部信号转换为SNN输入。
    *   多样的输出解码器，从SNN活动中提取决策或信号。
    *   奖励信号处理器，如滑动平均平滑。
*   **实用工具** (`dynn.utils`):
    *   数据记录探针 (`probes.py`)，用于监控仿真过程中的各种变量。
*   **目标应用**: 初期聚焦于强化学习场景，例如使用SNN控制 OpenAI Gym 环境 (如 "MountainCar-v0")。

## 安装

```bash
# 详细安装步骤待定
# 通常包括克隆仓库和安装依赖
# pip install -r requirements.txt
```

## 快速开始 (API 概览)

以下是如何使用 DyNN 构建和仿真一个简单SNN的基本流程：

```python
from dynn.core.neurons import NeuronPopulation, IzhikevichNeuron
from dynn.core.synapses import SynapseCollection
from dynn.core.network import NeuralNetwork
from dynn.core.simulator import Simulator
from dynn.config import default_config # 假设的配置入口

# 1. 加载或定义配置 (dt, 神经元参数等)
config = default_config
dt = config.simulation.dt

# 2. 创建网络
network = NeuralNetwork(dt=dt)

# 3. 创建神经元群体
input_pop = NeuronPopulation(IzhikevichNeuron, 100, params=config.neurons.input_params, name="Input")
output_pop = NeuronPopulation(IzhikevichNeuron, 10, params=config.neurons.output_params, name="Output")
network.add_population(input_pop)
network.add_population(output_pop)

# 4. 创建突触连接
connections = SynapseCollection(input_pop, output_pop, initial_weights=...) # ...表示权重初始化策略
network.add_synapses(connections)

# 5. (可选) 添加学习规则
# from dynn.core.learning_rules import STDP
# stdp_rule = STDP(connections, dt=dt, ...)
# network.add_learning_rule(stdp_rule)

# 6. (可选) 添加探针
# from dynn.utils.probes import PopulationProbe
# spike_probe = PopulationProbe(output_pop, 'fired', record_interval_ms=10)
# network.add_probe(spike_probe)

# 7. 创建并运行仿真器
simulator = Simulator(network)
# simulator.run_for_duration(duration_ms=1000, external_inputs={'Input': input_stimulus_array})

# 8. 获取并分析结果
# results = spike_probe.get_data()
```

更详细的用例请参考 `experiments/` 目录下的实验脚本，例如 `experiments/mountain_car_v0_experiment.py`。

## 模块设计

关于 DyNN 框架的详细模块设计、各个组件的职责和相互关系，请参阅 `DESIGN.md`。
