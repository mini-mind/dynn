# Placeholder for reward signal processing 

import numpy as np
from collections import deque

class BaseRewardProcessor:
    """奖励处理器的基类。"""
    def __init__(self, **kwargs):
        # 通用参数
        pass

    def process(self, reward, observation=None, action=None, next_observation=None, done=None, **kwargs):
        """
        处理原始奖励信号并返回一个用于调制学习的标量奖励信号。
        子类必须实现此方法。

        参数:
            reward (float): 来自环境的原始（或通过先前步骤计算的）奖励值。
            observation (any, optional): 当前观察。
            action (any, optional): 执行的动作。
            next_observation (any, optional): 下一个观察。
            done (bool, optional): 轮次是否结束。
            **kwargs: 其他参数，例如 dt, current_time。

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

    def process(self, reward, observation=None, action=None, next_observation=None, done=None, **kwargs):
        """
        处理原始奖励，返回滑动平均奖励。
        其他参数 (observation, action, etc.) 在此特定处理器中未使用。
        """
        self.reward_history.append(reward) # 只使用 reward
        if not self.reward_history:
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


# # 针对 "MountainCar-v0" 环境的特定奖励函数
# # README: "一个可供初步探索的方案是基于小车位置和加速度 (或速度变化) 的组合。"
# # 这个通常在环境交互循环中计算，然后传递给 BaseRewardProcessor 或其子类。
# # 这里我们可以定义一个函数来计算这个 *原始* 奖励，而不是一个处理器类。
# # 或者，一个处理器类可以封装这个逻辑。
# 
# class MountainCarCustomReward(BaseRewardProcessor):
#     """
#     为MountainCar-v0计算一个自定义的原始奖励。
#     这个类本身可以计算原始奖励，然后传递给另一个处理器（如平滑器），
#     或者它直接输出用于调制的信号（如果不需要进一步平滑）。
#     这里，我们让它计算一个"原始"的形状化奖励，然后可以被平滑。
#     """
#     def __init__(self, goal_position=0.5, position_weight=1.0, velocity_weight=0.1, **kwargs):
#         """
#         初始化Mountain Car的自定义奖励函数。
# 
#         参数:
#             goal_position (float): 目标位置 (山顶旗帜的位置)。
#             position_weight (float): 位置奖励的权重。
#             velocity_weight (float): 速度奖励的权重。
#         """
#         super().__init__(**kwargs)
#         self.goal_position = goal_position
#         self.position_weight = position_weight
#         self.velocity_weight = velocity_weight
#         self.previous_position = None # 用于计算速度（如果观察中不直接提供速度，或用于加速度）
#         self.previous_velocity = None # 用于计算加速度
# 
#     # 之前这里的签名是 process(self, observation, dt=None, current_time=None)
#     # 需要改成新的标准签名
#     def process(self, reward, observation=None, action=None, next_observation=None, done=None, **kwargs):
#         """
#         根据观察计算自定义奖励。
#         在这个被注释掉的例子中，`reward` (来自环境的原始奖励) 可能被忽略或与形状化奖励结合。
#         
#         参数:
#             observation (list or np.array): 环境观察 [position, velocity]。
#         
#         返回:
#             float: 计算得到的（形状化）奖励值。
#         """
#         if observation is None or len(observation) < 2:
#             # 如果此处理器期望观察值，但未提供，则返回原始奖励或0
#             return reward if reward is not None else 0.0
#             
#         position, velocity = observation[0], observation[1]
#         
#         shaped_reward_component = 0.0
#         shaped_reward_component += self.position_weight * position
#         
#         # dt = kwargs.get('dt') # 如果需要 dt
#         if self.previous_velocity is not None: # and dt is not None and dt > 0:
#             # 假设 dt=1 如果没有提供. 对于加速度，dt很重要。
#             # 如果没有dt, 可以使用 (velocity - self.previous_velocity) 作为速度变化量。
#             acceleration_proxy = velocity - self.previous_velocity 
#             shaped_reward_component += self.velocity_weight * acceleration_proxy
#         
#         self.previous_position = position
#         self.previous_velocity = velocity
#         
#         # 决定如何结合原始环境奖励与形状化奖励
#         # 方案A: 仅返回形状化部分: return shaped_reward_component
#         # 方案B: 将形状化部分加到原始奖励上: return (reward if reward is not None else 0.0) + shaped_reward_component
#         # 这里选择方案B作为示例
#         return (reward if reward is not None else 0.0) + shaped_reward_component
# 
#     def reset(self):
#         self.previous_position = None
#         self.previous_velocity = None
#         # print("MountainCarCustomReward has been reset.")
# 
#     def __repr__(self):
#         return (f"MountainCarCustomReward(goal={self.goal_position}, "
#                 f"pos_w={self.position_weight}, vel_w={self.velocity_weight})") 