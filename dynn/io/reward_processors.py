# Placeholder for reward signal processing 

import numpy as np
from collections import deque

class BaseRewardProcessor:
    """奖励处理器的基类。"""
    def __init__(self, **kwargs):
        # 通用参数
        pass

    def process(self, raw_reward, dt=None, current_time=None):
        """
        处理原始奖励信号并返回一个用于调制学习的标量奖励信号。
        子类必须实现此方法。

        参数:
            raw_reward (float): 来自环境的原始（或自定义计算的）奖励值。
            dt (float, optional): 时间步长。
            current_time (float, optional): 当前仿真时间。

        返回:
            float: 处理后的标量奖励信号。
        """
        raise NotImplementedError

    def reset(self):
        """重置处理器的内部状态 (如果需要)。"""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class SlidingWindowSmoother(BaseRewardProcessor):
    """
    提供一个奖励处理器模块，该模块接收来自环境的原始（或自定义计算的）奖励值，
    并基于一个可配置的时间窗口内的滑动平均值来计算最终用于调制学习率的平滑奖励信号。
    """
    def __init__(self, window_size=100, **kwargs):
        """
        初始化滑动窗口平滑器。

        参数:
            window_size (int): 用于计算滑动平均的窗口大小 (时间步数)。
        """
        super().__init__(**kwargs)
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("窗口大小必须是一个正整数。")
        self.window_size = window_size
        self.reward_history = deque(maxlen=self.window_size)
        self.current_smoothed_reward = 0.0 # 初始平滑奖励

    def process(self, raw_reward, dt=None, current_time=None):
        """
        处理原始奖励，返回滑动平均奖励。
        """
        self.reward_history.append(raw_reward)
        if not self.reward_history: # 应该不会发生，因为maxlen>0，并且我们刚添加了一个
            self.current_smoothed_reward = 0.0
        else:
            self.current_smoothed_reward = np.mean(self.reward_history)
        return self.current_smoothed_reward

    def reset(self):
        """
        重置奖励历史和平滑奖励。
        """
        self.reward_history.clear()
        self.current_smoothed_reward = 0.0
        # print("SlidingWindowSmoother has been reset.")

    def __repr__(self):
        return f"SlidingWindowSmoother(window_size={self.window_size})"


# 针对 "MountainCar-v0" 环境的特定奖励函数
# README: "一个可供初步探索的方案是基于小车位置和加速度 (或速度变化) 的组合。"
# 这个通常在环境交互循环中计算，然后传递给 BaseRewardProcessor 或其子类。
# 这里我们可以定义一个函数来计算这个 *原始* 奖励，而不是一个处理器类。
# 或者，一个处理器类可以封装这个逻辑。

class MountainCarCustomReward(BaseRewardProcessor):
    """
    为MountainCar-v0计算一个自定义的原始奖励。
    这个类本身可以计算原始奖励，然后传递给另一个处理器（如平滑器），
    或者它直接输出用于调制的信号（如果不需要进一步平滑）。
    这里，我们让它计算一个"原始"的形状化奖励，然后可以被平滑。
    """
    def __init__(self, goal_position=0.5, position_weight=1.0, velocity_weight=0.1, **kwargs):
        """
        初始化Mountain Car的自定义奖励函数。

        参数:
            goal_position (float): 目标位置 (山顶旗帜的位置)。
            position_weight (float): 位置奖励的权重。
            velocity_weight (float): 速度奖励的权重。
        """
        super().__init__(**kwargs)
        self.goal_position = goal_position
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.previous_position = None # 用于计算速度（如果观察中不直接提供速度，或用于加速度）
        self.previous_velocity = None # 用于计算加速度

    def process(self, observation, dt=None, current_time=None):
        """
        根据观察计算自定义奖励。
        
        参数:
            observation (list or np.array): 环境观察 [position, velocity]。
        
        返回:
            float: 计算得到的原始奖励值。
        """
        position, velocity = observation[0], observation[1]
        
        # 基础奖励：通常是每步 -1，直到到达目标
        # Gym 环境通常自己会返回这个。
        # 这里我们计算一个形状化奖励。
        
        reward = 0.0
        
        # 基于位置的奖励: 越接近目标位置越好 (或基于高度)
        # 高度 h = sin(3 * position)
        # 如果直接用位置，可以奖励向目标方向的移动或当前位置的进展
        # reward += self.position_weight * (position - self.min_position) # 简单的离起点距离
        # 或者，奖励向右移动（朝向目标）
        # reward += self.position_weight * position # 更高的位置更好
        
        # 示例：奖励小车到达更高位置
        # 也可以是奖励与目标位置的距离减少
        # reward += self.position_weight * (np.abs(position - self.goal_position) * -1) # 负距离

        # 尝试 README 中的想法：基于小车位置和加速度 (或速度变化) 的组合
        # 1. 位置：简单地，当前位置（越高越好，或越接近目标越好）
        #    MountainCar的位置范围 [-1.2, 0.6]。目标是0.5。
        #    奖励可以与 (position - (-1.2)) 成正比，即离最左边越远越好。
        #    或者，更直接地，奖励 (position - (-0.5))，即车在右半边更好。
        reward += self.position_weight * position
        
        # 2. 速度/加速度: 
        #    速度：向右的速度（正速度）通常是好的。
        # reward += self.velocity_weight * velocity
        
        #    加速度：如果速度变化是正向的（加速向右，或减速向左），可能是好的。
        if self.previous_velocity is not None and dt is not None and dt > 0:
            acceleration = (velocity - self.previous_velocity) / dt
            reward += self.velocity_weight * acceleration # 这里 velocity_weight 作为加速度的权重
        
        self.previous_position = position
        self.previous_velocity = velocity
        
        # 如果成功到达目标，Gym环境会给出大的正奖励 (例如 +100)，并且 episode 结束。
        # 这里的形状化奖励旨在引导学习过程。
        # Gym环境本身会返回一个reward，这个自定义奖励可以加到那个上面，或者替换它。
        # 通常，学习规则接收的是一个最终的标量奖励信号。
        return reward

    def reset(self):
        self.previous_position = None
        self.previous_velocity = None
        # print("MountainCarCustomReward has been reset.")

    def __repr__(self):
        return (f"MountainCarCustomReward(goal={self.goal_position}, "
                f"pos_w={self.position_weight}, vel_w={self.velocity_weight})") 