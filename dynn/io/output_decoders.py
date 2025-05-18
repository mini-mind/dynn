# Placeholder for output decoding mechanisms 

import numpy as np

class BaseOutputDecoder:
    """输出解码器的基类。"""
    def __init__(self, source_pop_name, **kwargs):
        self.source_pop_name = source_pop_name # 源输出神经元群体的名称
        # 其他通用参数

    def decode(self, spike_activities_map, dt=None, current_time=None):
        """
        从SNN输出神经元的活动中提取信息，并将其转换为外部可理解的信号/动作。
        子类必须实现此方法。

        参数:
            spike_activities_map (dict): 
                一个字典，键是SNN中神经元群体的名称，值是该群体当前的脉冲状态 (布尔数组)。
                解码器通常只关心 self.source_pop_name 对应的数据。
            dt (float, optional): 时间步长。
            current_time (float, optional): 当前仿真时间。

        返回:
            any: 解码后的动作或信号。
        """
        raise NotImplementedError

    def get_source_population_name(self):
        return self.source_pop_name

    def __repr__(self):
        return f"{self.__class__.__name__}(source_pop_name='{self.source_pop_name}')"


class InstantaneousSpikeCountDecoder(BaseOutputDecoder):
    """
    基于输出神经元群体瞬时脉冲发放情况来决定离散动作。
    一种简单的实现是：哪个神经元发放脉冲，就对应哪个动作。
    或者，如果多个神经元代表一个动作，则选择脉冲数最多的那个（如果瞬时允许多个脉冲，尽管通常是一个dt一个脉冲状态）。
    这里简化为：每个神经元对应一个离散动作，选择第一个发放脉冲的神经元对应的动作。
    或者，可以实现一个 "赢者通吃" (Winner-Take-All, WTA) 的变体，如果多个神经元发放脉冲。
    """
    def __init__(self, source_pop_name, num_actions, default_action=None, **kwargs):
        """
        初始化瞬时脉冲计数解码器。

        参数:
            source_pop_name (str): 源输出神经元群体的名称。
            num_actions (int): 可能的离散动作数量。
                               期望源神经元群体的数量与 num_actions 相同，
                               或者提供一种映射方式。
                               这里假设 len(source_pop) == num_actions。
            default_action (any, optional): 如果没有神经元发放脉冲，则返回此默认动作。
                                         如果为None，并且没有脉冲，可能会引发错误或返回特定值。
        """
        super().__init__(source_pop_name, **kwargs)
        self.num_actions = num_actions
        self.default_action = default_action
        # 假设动作是从 0 到 num_actions - 1

    def decode(self, spike_activities_map, dt=None, current_time=None):
        """
        解码脉冲活动为单个离散动作。
        基于"赢者通吃" (WTA) 的简化版本：如果有任何神经元发放脉冲，选择第一个发放脉冲的神经元索引作为动作。
        如果需要更复杂的WTA（例如，在多个脉冲中选择最强的，或处理平局），则需要修改此逻辑。
        README: "基于SNN输出神经元的瞬时脉冲发放情况来决定离散动作...避免使用需要在一个时间窗口内累积活动的方法。"
        
        参数:
            spike_activities_map (dict): {pop_name: spike_array}
        
        返回:
            int or any: 解码后的离散动作 (通常是整数索引) 或默认动作。
        """
        if self.source_pop_name not in spike_activities_map:
            raise ValueError(f"源群体 '{self.source_pop_name}' 的脉冲数据未在输入中找到。")

        spikes = spike_activities_map[self.source_pop_name]
        if not isinstance(spikes, np.ndarray) or spikes.ndim != 1:
            raise ValueError("源群体的脉冲数据必须是一维numpy数组。")

        if len(spikes) != self.num_actions:
            print(f"警告: 源群体 '{self.source_pop_name}' 的神经元数量 ({len(spikes)}) "
                  f"与预期的动作数量 ({self.num_actions}) 不匹配。将尝试继续。")
            # 可以选择截断或填充，或者要求严格匹配。这里假设直接使用，如果数量不匹配可能会导致索引问题。

        active_indices = np.where(spikes)[0]

        if len(active_indices) > 0:
            # 如果有多个神经元同时发放脉冲，这里简单选择第一个。
            # 实际应用中可能需要更复杂的竞争或投票机制。
            action = active_indices[0]
            # 确保动作在有效范围内 (如果神经元数量和动作数量可能不匹配)
            return min(action, self.num_actions - 1) 
        else:
            if self.default_action is not None:
                return self.default_action
            else:
                # 根据具体需求，可以选择抛出错误，或返回一个表示"无操作"的特定值
                # 例如，如果 num_actions 是3 (0, 1, 2), 可以返回 2 (对应 Gym MountainCar 的无操作)
                # 这里假设如果没有脉冲且没有默认动作，则返回一个代表无意义的值或者由调用者处理
                # 对于MountainCar (左0, 无1, 右2), 可以返回1作为默认无操作。
                # 为了通用性，返回None，让调用者决定
                return None 
                # raise ValueError("没有神经元发放脉冲，且未设置默认动作。")
    
    def __repr__(self):
        return (f"InstantaneousSpikeCountDecoder(source='{self.source_pop_name}', "
                f"num_actions={self.num_actions}, default_action={self.default_action})")


class MountainCarActionDecoder(InstantaneousSpikeCountDecoder):
    """
    针对 MountainCar-v0 环境的输出解码器。
    动作: 0 (左), 1 (无操作), 2 (右)。
    假设输出神经元群体有3个神经元，分别对应这三个动作。
    """
    def __init__(self, source_pop_name, num_neurons_for_action=3, default_action_idx=1, **kwargs):
        """
        参数:
            source_pop_name (str): 输出神经元群体的名称。
            num_neurons_for_action (int): 期望输出群体中的神经元数量，应等于动作空间大小。
                                         对于MountainCar-v0，这是3。
            default_action_idx (int): 如果没有脉冲，则采取的默认动作索引 (0, 1, 或 2)。
                                     默认为1 (无操作)。
        """
        # MountainCar-v0 有3个离散动作: 0 (向左推), 1 (不推), 2 (向右推)
        super().__init__(source_pop_name, num_actions=num_neurons_for_action, default_action=default_action_idx, **kwargs)
        if num_neurons_for_action != 3:
            print(f"警告: MountainCarActionDecoder 通常期望源群体有3个神经元，但配置为 {num_neurons_for_action}。")

    # decode 方法直接继承自 InstantaneousSpikeCountDecoder，
    # 只要 num_actions (来自父类的构造函数) 设置为3，并且 default_action 也正确设置，
    # 其行为应该符合要求。
    # 即：如果神经元0发放脉冲 -> 动作0
    #     如果神经元1发放脉冲 -> 动作1
    #     如果神经元2发放脉冲 -> 动作2
    #     如果没有脉冲 -> default_action (例如 1)
    #     如果多个发放，取第一个索引 (例如，如果0和2都发放，则动作0)

    def __repr__(self):
        return (f"MountainCarActionDecoder(source='{self.source_pop_name}', "
                f"num_actions={self.num_actions}, default_action={self.default_action})") 