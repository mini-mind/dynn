# `dynn.io.reward_processors` API 文档

## 模块概览

`dynn.io.reward_processors` 模块提供了用于处理来自环境的原始奖励信号的类。这些处理器可以将原始奖励转换为更适合驱动SNN学习规则（尤其是奖励调制的STDP）的标量信号，例如通过平滑化处理。

主要组件包括：

*   `BaseRewardProcessor`: 所有奖励处理器的抽象基类。
*   `SlidingWindowSmoother`: 使用滑动窗口平均法来平滑原始奖励信号。
*   `MountainCarCustomReward`: 一个为 OpenAI Gym "MountainCar-v0" 环境设计的特定奖励 shaping 函数，它可以计算一个基于位置和速度变化的形状化奖励，这个奖励随后可以被进一步处理（例如平滑）。

## 类详细说明

### `BaseRewardProcessor`

#### 类名及其构造函数签名

`BaseRewardProcessor(**kwargs)`

#### 类描述

`BaseRewardProcessor` 是所有特定奖励处理器类的抽象基类。它定义了奖励处理器应具备的基本结构和核心方法 `process()` 和 `reset()`。

#### 构造函数参数

*   `**kwargs`: 允许子类接收其他特定参数。

#### 主要方法

*   `process(self, raw_reward, dt=None, current_time=None)`:
    *   这是一个抽象方法，必须由子类实现。它负责处理原始奖励信号 `raw_reward` 并返回一个（通常是标量的）奖励信号，该信号将用于调制学习规则。
    *   **参数**:
        *   `raw_reward` (`float`): 来自环境的原始奖励值，或者由某个奖励函数自定义计算的奖励值。
        *   `dt` (`float`, 可选): 当前仿真时间步长。
        *   `current_time` (`float`, 可选): 当前仿真总时间。
    *   **返回值**: `float` - 处理后的标量奖励信号。
    *   **注意**: 调用此方法时会引发 `NotImplementedError`，除非在子类中被重写。

*   `reset(self)`:
    *   重置处理器的任何内部状态（例如，历史记录、累积值等）。子类应根据需要实现此方法。

*   `__repr__(self)`:
    *   返回该处理器实例的字符串表示形式。

---

### `SlidingWindowSmoother`

#### 类名及其构造函数签名

`SlidingWindowSmoother(window_size=100, **kwargs)`

#### 类描述

`SlidingWindowSmoother` 实现了一种奖励信号平滑技术。它维护一个最近接收到的原始奖励值的历史记录（在一个固定大小的滑动窗口内），并计算这些值的平均值作为输出的平滑奖励信号。这种平滑处理有助于减少奖励信号中的高频噪声，为学习规则提供更稳定的指导信号。

#### 构造函数参数

*   `window_size` (`int`, 可选, 默认值: `100`): 用于计算滑动平均的窗口大小，即历史记录中保留的最近奖励值的数量。
*   `**kwargs`: 传递给基类构造函数的其他参数。

#### 主要方法

*   `process(self, raw_reward, dt=None, current_time=None)`:
    *   将 `raw_reward` 添加到历史记录中，并计算当前窗口内所有奖励的平均值。
    *   **参数**: 同 `BaseRewardProcessor.process`。
    *   **返回值**: `float` - 当前滑动窗口内的平均奖励值。

*   `reset(self)`:
    *   清空奖励历史记录，并将当前平滑奖励重置为0.0。

*   `__repr__(self)`:
    *   返回该平滑器实例的详细字符串表示形式，包括窗口大小。

---

### `MountainCarCustomReward`

#### 类名及其构造函数签名

`MountainCarCustomReward(goal_position=0.5, position_weight=1.0, velocity_weight=0.1, **kwargs)`

#### 类描述

`MountainCarCustomReward` 是一个为 OpenAI Gym 的 "MountainCar-v0" 环境设计的奖励处理器，但它的主要功能是根据环境的观察（小车的位置和速度）来计算一个*形状化*的原始奖励，而不是直接平滑环境返回的稀疏奖励。这个形状化的奖励旨在为智能体提供更频繁的反馈，以指导其学习过程。例如，它可以奖励小车向目标位置移动或增加其向目标方向的速度/加速度。这个类计算出的"原始"形状化奖励可以被另一个奖励处理器（如 `SlidingWindowSmoother`）进一步处理。

#### 构造函数参数

*   `goal_position` (`float`, 可选, 默认值: `0.5`): "MountainCar-v0" 环境中目标旗帜的位置。
*   `position_weight` (`float`, 可选, 默认值: `1.0`): 在计算形状化奖励时，赋予位置因素的权重。
*   `velocity_weight` (`float`, 可选, 默认值: `0.1`): 在计算形状化奖励时，赋予速度/加速度因素的权重。
*   `**kwargs`: 传递给基类构造函数的其他参数。

#### 主要属性/特性

*   `previous_position` (`float` or `None`): 上一个时间步的小车位置，用于计算速度变化。
*   `previous_velocity` (`float` or `None`): 上一个时间步的小车速度，用于计算加速度。

#### 主要方法

*   `process(self, observation, dt=None, current_time=None)`:
    *   根据当前的环境观察 `observation` (包含位置和速度) 计算一个形状化的奖励值。
    *   当前的实现奖励小车的位置（越高/越接近目标越好）以及（如果 `dt` 和 `previous_velocity` 可用）朝向目标方向的加速度。
    *   **参数**:
        *   `observation` (`list` 或 `np.ndarray`): 环境的观察向量，期望格式为 `[position, velocity]`。
        *   `dt` (`float`, 可选): 时间步长，用于计算加速度。
        *   `current_time` (`float`, 可选): 当前仿真时间 (未使用)。
    *   **返回值**: `float` - 计算得到的形状化原始奖励值。

*   `reset(self)`:
    *   重置 `previous_position` 和 `previous_velocity` 为 `None`。

*   `__repr__(self)`:
    *   返回该自定义奖励处理器实例的详细字符串表示形式。

## (可选) 使用示例

```python
import numpy as np
from dynn.io.reward_processors import SlidingWindowSmoother, MountainCarCustomReward

# 示例 1: SlidingWindowSmoother
smoother = SlidingWindowSmoother(window_size=5)
print(smoother)

raw_rewards = [0, 0, 10, 0, 0, 0, 20, 0, 0, 0]
print("Raw Rewards | Smoothed Rewards")
print("-----------------------------")
for r in raw_rewards:
    smoothed_r = smoother.process(r)
    print(f"{r:<11} | {smoothed_r:.2f}")

smoother.reset()
print("Smoother reset.")

# 示例 2: MountainCarCustomReward
# (通常与环境交互循环一起使用)
mc_reward_shaper = MountainCarCustomReward(
    position_weight=1.0, 
    velocity_weight=0.5
)
print(mc_reward_shaper)

# 模拟一些观察和时间步长
obs1 = [-0.8, 0.01] # position, velocity
dt = 0.1
shaped_r1 = mc_reward_shaper.process(obs1, dt=dt)
print(f"Obs: {obs1}, Shaped Reward: {shaped_r1:.3f} (velocity term might be small or zero initially)")

obs2 = [-0.7, 0.02] # 小车向右移动并加速
shaped_r2 = mc_reward_shaper.process(obs2, dt=dt)
print(f"Obs: {obs2}, Shaped Reward: {shaped_r2:.3f}")

obs3 = [-0.6, 0.01] # 小车继续向右移动但减速
shaped_r3 = mc_reward_shaper.process(obs3, dt=dt)
print(f"Obs: {obs3}, Shaped Reward: {shaped_r3:.3f}")

mc_reward_shaper.reset()
print("MountainCarCustomReward reset.")

# 结合使用：形状化奖励后再平滑
print("\nCombining MountainCarCustomReward with SlidingWindowSmoother:")
reward_pipeline_smoother = SlidingWindowSmoother(window_size=3)
mc_reward_shaper.reset() # 确保 shaper 也重置了内部状态

observations = [
    [-1.0, 0.0], 
    [-0.9, 0.01],
    [-0.8, 0.02],
    [-0.7, 0.03],
    [-0.6, 0.02],
    [-0.5, 0.01] # 达到目标附近
]

print("Obs        | Shaped R | Smoothed Shaped R")
print("---------------------------------------------")
for obs_step in observations:
    raw_shaped_reward = mc_reward_shaper.process(obs_step, dt=0.1)
    final_reward_for_learning = reward_pipeline_smoother.process(raw_shaped_reward)
    print(f"{str(obs_step):<10} | {raw_shaped_reward:<8.3f} | {final_reward_for_learning:.3f}")

``` 