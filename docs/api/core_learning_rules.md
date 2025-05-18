# `dynn.core.learning_rules` API 文档

## 模块概览

`dynn.core.learning_rules` 模块提供了神经网络中突触可塑性规则的实现。这些规则定义了突触权重如何根据神经元活动（例如脉冲时序）和外部信号（例如奖励）进行修改。

主要组件包括：

*   `BaseLearningRule`: 所有学习规则的抽象基类，定义了通用接口。
*   `TraceSTDP`: 一种基于脉冲迹的尖峰时间依赖可塑性 (STDP) 规则，实现了权重依赖的乘性STDP。

## 类详细说明

### `BaseLearningRule`

#### 类名及其构造函数签名

`BaseLearningRule(lr_ltp=0.001, lr_ltd=0.001)`

#### 类描述

`BaseLearningRule` 是 DyNN 框架中所有学习规则的基类。它定义了学习规则应具备的基本结构和接口，包括设置学习率和处理奖励调制。具体的权重更新逻辑由子类实现。

#### 构造函数参数

*   `lr_ltp` (`float`, 可选, 默认值: `0.001`): 长期增强 (LTP) 的基础学习率。
*   `lr_ltd` (`float`, 可选, 默认值: `0.001`): 长期削弱 (LTD) 的基础学习率。

#### 主要属性/特性

*   `lr_ltp` (`float`): LTP 的基础学习率。
*   `lr_ltd` (`float`): LTD 的基础学习率。
*   `reward_modulation` (`float`): 当前的奖励调制因子，默认为 `1.0` (表示没有调制)。此因子会乘以基础学习率以得到有效学习率。

#### 主要方法

*   `update_weights(self, synapse_collection, pre_spikes, post_spikes, dt, current_time)`:
    *   这是一个抽象方法，必须由子类实现。它定义了如何根据突触前后的神经元活动来更新 `synapse_collection` 中的突触权重。
    *   **参数**:
        *   `synapse_collection` (`SynapseCollection`): 要更新其权重的突触集合对象。
        *   `pre_spikes` (`np.ndarray`): 布尔数组，指示突触前神经元群体中哪些神经元发放了脉冲。
        *   `post_spikes` (`np.ndarray`): 布尔数组，指示突触后神经元群体中哪些神经元发放了脉冲。
        *   `dt` (`float`): 仿真时间步长。
        *   `current_time` (`float`): 当前仿真时间。
    *   **注意**: 调用此方法时会引发 `NotImplementedError`，除非在子类中被重写。

*   `set_reward_modulation(self, reward_signal)`:
    *   根据外部提供的奖励信号设置学习率的调制因子 (`self.reward_modulation`)。
    *   默认实现直接使用 `reward_signal` 作为调制因子。根据需求规格，此调制应同向影响LTP和LTD。
    *   **参数**:
        *   `reward_signal` (`float`): 标量奖励信号，通常是经过处理（例如平滑）的奖励值。

*   `get_effective_lr_ltp(self)`:
    *   获取经过奖励调制后的有效LTP学习率。
    *   **返回值**: `float` - `self.lr_ltp * self.reward_modulation`。

*   `get_effective_lr_ltd(self)`:
    *   获取经过奖励调制后的有效LTD学习率。
    *   **返回值**: `float` - `self.lr_ltd * self.reward_modulation`。

---

### `TraceSTDP`

#### 类名及其构造函数签名

`TraceSTDP(lr_ltp=0.005, lr_ltd=0.005, tau_pre=20, tau_post=20, w_max=1.0, w_min=0.0, trace_increase=1.0)`

#### 类描述

`TraceSTDP` 类实现了基于脉冲迹的尖峰时间依赖可塑性 (STDP) 规则。当神经元发放脉冲时，其关联的突触前或突触后迹会增加，然后随时间指数衰减。权重的改变取决于这些迹变量的值以及当前权重的大小（权重依赖性）和外部奖励信号（通过继承的 `set_reward_modulation`）。该实现是一种乘性STDP，即权重的改变量与 `(w_max - W)` (LTP) 或 `(W - w_min)` (LTD) 成比例。

#### 构造函数参数

*   `lr_ltp` (`float`, 可选, 默认值: `0.005`): LTP的基础学习率。
*   `lr_ltd` (`float`, 可选, 默认值: `0.005`): LTD的基础学习率。
*   `tau_pre` (`float`, 可选, 默认值: `20`): 突触前脉冲迹的时间常数 (ms)。控制突触前脉冲影响的持续时间。
*   `tau_post` (`float`, 可选, 默认值: `20`): 突触后脉冲迹的时间常数 (ms)。控制突触后脉冲影响的持续时间。
*   `w_max` (`float`, 可选, 默认值: `1.0`): 突触权重的最大允许值。用于权重裁剪和乘性更新规则。
*   `w_min` (`float`, 可选, 默认值: `0.0`): 突触权重的最小允许值。用于权重裁剪和乘性更新规则。
*   `trace_increase` (`float`, 可选, 默认值: `1.0`): 每当神经元发放脉冲时，其对应的脉冲迹变量增加的固定值。

#### 主要属性/特性

*   继承自 `BaseLearningRule`: `lr_ltp`, `lr_ltd`, `reward_modulation`。
*   `tau_pre` (`float`): 突触前迹的时间常数。
*   `tau_post` (`float`): 突触后迹的时间常数。
*   `w_max` (`float`): 权重的上限。
*   `w_min` (`float`): 权重的下限。
*   `trace_increase` (`float`): 脉冲触发的迹增量。

#### 主要方法

*   `update_traces(self, population, spikes, trace_type, dt)`:
    *   更新指定神经元群体 (`population`) 的脉冲迹。这包括两步：首先根据时间常数 `tau_pre` 或 `tau_post` 和时间步长 `dt` 对现有迹进行指数衰减，然后在发放了脉冲 (`spikes` 中标记为 `True` 的神经元) 的神经元对应的迹上增加 `self.trace_increase`。
    *   脉冲迹存储在 `NeuronPopulation` 对象的 `spike_trace_pre` 和 `spike_trace_post` 属性中。
    *   **参数**:
        *   `population` (`NeuronPopulation`): 要更新其脉冲迹的神经元群体。
        *   `spikes` (`np.ndarray`): 布尔数组，指示 `population` 中哪些神经元发放了脉冲。
        *   `trace_type` (`str`): 指定要更新的迹类型，应为 `'pre'` (突触前) 或 `'post'` (突触后)。
        *   `dt` (`float`): 仿真时间步长。

*   `update_weights(self, synapse_collection, pre_spikes, post_spikes, dt, current_time)`:
    *   重写自 `BaseLearningRule`。根据STDP规则更新 `synapse_collection` 中的突触权重。
    *   **逻辑步骤**:
        1.  调用 `update_traces` 更新突触前群体 (`synapse_collection.pre_pop`) 的 `spike_trace_pre` 和突触后群体 (`synapse_collection.post_pop`) 的 `spike_trace_post`。
        2.  获取有效的学习率 (`lr_ltp * reward_modulation`, `lr_ltd * reward_modulation`)。
        3.  计算权重变化 `delta_w`:
            *   **LTP**: 当一个突触后神经元 `i` 发放脉冲 (`post_spikes[i]` is True) 时，其所有传入连接 `j` (如果存在) 的权重增加量为 `effective_lr_ltp * pre_pop.spike_trace_pre[j] * (w_max - weights[i,j])`。
            *   **LTD**: 当一个突触前神经元 `j` 发放脉冲 (`pre_spikes[j]` is True) 时，其所有传出连接 `i` (如果存在) 的权重减少量为 `effective_lr_ltd * post_pop.spike_trace_post[i] * (weights[i,j] - w_min)` (注意符号，LTD是负向变化)。
        4.  将计算得到的 `delta_w` 应用于 `synapse_collection.weights` (仅对 `connection_mask` 中标记为 `True` 的连接)。
        5.  对更新后的权重进行裁剪，确保它们保持在 `[w_min, w_max]` 区间内 (具体区间会根据 `synapse_collection.is_excitatory` 调整，例如兴奋性权重通常为非负)。
        6.  确保未连接的权重（掩码为False）保持为0。
    *   **参数**: 同 `BaseLearningRule.update_weights`。

*   `__repr__(self)`:
    *   返回 `TraceSTDP` 规则实例的字符串表示形式，包含其主要参数。
    *   **返回值**: `str`。

## (可选) 使用示例

```python
import numpy as np
# 假设 NeuronPopulation 和 SynapseCollection 已经定义和实例化
# from dynn.core.neurons import NeuronPopulation 
# from dynn.core.synapses import SynapseCollection
from dynn.core.learning_rules import TraceSTDP, BaseLearningRule

# Dummy classes for demonstration if real ones are not available here
class DummyPopulation:
    def __init__(self, num_neurons, name):
        self.name = name
        self.num_neurons = num_neurons
        self.spike_trace_pre = np.zeros(num_neurons)
        self.spike_trace_post = np.zeros(num_neurons)
    def __len__(self):
        return self.num_neurons

class DummySynapseCollection:
    def __init__(self, pre_pop, post_pop, name="dummy_syn"):
        self.pre_pop = pre_pop
        self.post_pop = post_pop
        self.name = name
        self.weights = np.random.rand(len(post_pop), len(pre_pop)) * 0.5
        self.connection_mask = np.ones_like(self.weights, dtype=bool)
        self.is_excitatory = True

pre_pop = DummyPopulation(10, "PrePop")
post_pop = DummyPopulation(5, "PostPop")
syn_collection = DummySynapseCollection(pre_pop, post_pop)

# 1. 创建 TraceSTDP 学习规则实例
stdp_rule = TraceSTDP(
    lr_ltp=0.01,
    lr_ltd=0.012,
    tau_pre=15,  # ms
    tau_post=25, # ms
    w_max=1.0,
    w_min=0.01,
    trace_increase=1.0
)
print(stdp_rule)

# 2. 模拟一些脉冲活动
pre_spikes_t1 = np.array([1,0,1,0,0,1,0,0,0,0], dtype=bool)
post_spikes_t1 = np.array([0,1,0,0,1], dtype=bool)
dt = 1.0 # ms
current_time_t1 = 10.0 # ms

# 3. 更新权重 (第一次)
print(f"Weights before STDP at t1:\n{syn_collection.weights}")
stdp_rule.update_weights(syn_collection, pre_spikes_t1, post_spikes_t1, dt, current_time_t1)
print(f"Weights after STDP at t1:\n{syn_collection.weights}")
print(f"Pre-traces after t1: {pre_pop.spike_trace_pre}")
print(f"Post-traces after t1: {post_pop.spike_trace_post}")

# 4. 模拟后续脉冲和奖励调制
pre_spikes_t2 = np.array([0,1,0,1,0,0,0,1,0,0], dtype=bool)
post_spikes_t2 = np.array([1,0,1,0,0], dtype=bool)
current_time_t2 = 20.0 # ms

# 假设收到了一个正奖励信号
reward = 1.5 
stdp_rule.set_reward_modulation(reward)
print(f"Effective LTP learning rate with reward {reward}: {stdp_rule.get_effective_lr_ltp()}")

stdp_rule.update_weights(syn_collection, pre_spikes_t2, post_spikes_t2, dt, current_time_t2)
print(f"Weights after STDP at t2 (with reward modulation):\n{syn_collection.weights}")
print(f"Pre-traces after t2: {pre_pop.spike_trace_pre}")
print(f"Post-traces after t2: {post_pop.spike_trace_post}")

# 5. 假设收到负奖励 (惩罚) 或较小的奖励
reward_punish = 0.5
stdp_rule.set_reward_modulation(reward_punish)
print(f"Effective LTP learning rate with reward {reward_punish}: {stdp_rule.get_effective_lr_ltp()}")
# ... 再次调用 update_weights 将使用新的调制学习率
``` 