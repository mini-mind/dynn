# Placeholder for input encoding mechanisms 

import numpy as np

class BaseInputEncoder:
    """输入编码器的基类。"""
    def __init__(self, target_pop_name, **kwargs):
        self.target_pop_name = target_pop_name # 目标输入神经元群体的名称
        # 其他通用参数可以在这里初始化

    def encode(self, observation, dt=None, current_time=None):
        """
        将外部观察值转换为SNN输入神经元的活动 (例如，注入电流或目标脉冲)。
        子类必须实现此方法。

        参数:
            observation: 来自环境的原始观察值。
            dt (float, optional): 时间步长，某些编码器可能需要。
            current_time (float, optional): 当前仿真时间，某些编码器可能需要。

        返回:
            dict: 键为目标群体名称，值为注入电流的np.array。
                  例如: {self.target_pop_name: currents_array}
        """
        raise NotImplementedError

    def get_target_population_name(self):
        return self.target_pop_name

    def __repr__(self):
        return f"{self.__class__.__name__}(target_pop_name='{self.target_pop_name}')"


class GaussianEncoder(BaseInputEncoder):
    """
    使用一组高斯感受野将一维标量值编码为神经元群体的注入电流。
    """
    def __init__(self, target_pop_name, num_neurons, 
                 min_val, max_val, 
                 sigma_scale=0.1, current_amplitude=10.0, 
                 **kwargs):
        """
        初始化高斯编码器。

        参数:
            target_pop_name (str): 目标输入神经元群体的名称。
            num_neurons (int): 目标群体中的神经元数量。
            min_val (float): 期望输入值的最小值。
            max_val (float): 期望输入值的最大值。
            sigma_scale (float): 高斯感受野宽度 (sigma) 相对于 (max_val - min_val) 的比例。
                                 实际 sigma = sigma_scale * (max_val - min_val)。
            current_amplitude (float): 高斯激活峰值处的最大注入电流。
        """
        super().__init__(target_pop_name, **kwargs)
        if num_neurons <= 0:
            raise ValueError("神经元数量必须为正。")
        if max_val < min_val: # 修改这里，允许 max_val == min_val
            raise ValueError("max_val 必须大于或等于 min_val。")
            
        self.num_neurons = num_neurons
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.val_range = self.max_val - self.min_val
        self.sigma = sigma_scale * self.val_range
        if self.sigma == 0 and self.num_neurons > 1: # Avoid division by zero if range is tiny but multiple neurons exist
            self.sigma = 1e-6
        elif self.sigma == 0 and self.num_neurons == 1:
            self.sigma = self.val_range if self.val_range > 0 else 1.0 # single neuron covers whole range or default sigma

        self.current_amplitude = float(current_amplitude)
        
        # 计算每个神经元对应的高斯中心点 (means)
        if self.num_neurons == 1:
            self.means = np.array([self.min_val + self.val_range / 2.0])
        else:
            # 在 [min_val, max_val] 范围内均匀分布中心点
            self.means = np.linspace(self.min_val, self.max_val, self.num_neurons)

    def encode(self, observation_value, dt=None, current_time=None):
        """
        将单个标量观察值编码为注入电流。

        参数:
            observation_value (float): 要编码的标量值。

        返回:
            dict: {self.target_pop_name: currents_array}
        """
        if not isinstance(observation_value, (int, float)):
            try:
                # 尝试处理只有一个元素的numpy数组或列表
                if hasattr(observation_value, '__len__') and len(observation_value) == 1:
                    obs_val = float(observation_value[0])
                else:
                    obs_val = float(observation_value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"GaussianEncoder 需要一个标量观察值，但接收到: {observation_value} (类型: {type(observation_value)}) {e}")
        else:
            obs_val = float(observation_value)

        # 高斯函数: A * exp(-(x - mu)^2 / (2 * sigma^2))
        # x 是 observation_value, mu 是 self.means
        if self.sigma == 0: # 处理 sigma 为零的特殊情况 (例如，如果 val_range 为零)
            # 如果 sigma 为零，只有当 obs_val 精确匹配 mean 时才激活，否则为零
            # 这类似于一个非常窄的 delta 函数；更实际的是，如果 range 为0，所有神经元可能都应该有相同的响应
            # 或者，如果只有一个神经元，它应该总是以某种方式响应。
            # 为了避免数值问题和提供一个合理的行为:
            if self.num_neurons == 1:
                 # 如果只有一个神经元，且范围为0，可以假设它总是被激活，或者基于值是否等于mean
                 currents = np.full(self.num_neurons, self.current_amplitude if obs_val == self.means[0] else 0.0)
            else:
                 # 多个神经元，范围为0：所有神经元中心相同，如果obs_val匹配，则都激活
                 currents = np.where(obs_val == self.means, self.current_amplitude, 0.0)
        else:
            exponent = -((obs_val - self.means)**2) / (2 * self.sigma**2)
            currents = self.current_amplitude * np.exp(exponent)
        
        return {self.target_pop_name: currents}

    def __repr__(self):
        return (f"GaussianEncoder(target='{self.target_pop_name}', num_neurons={self.num_neurons}, "
                f"range=[{self.min_val:.2f}, {self.max_val:.2f}], sigma_scale={self.sigma/self.val_range if self.val_range else 0:.2f}, "
                f"amplitude={self.current_amplitude:.2f})")


class DirectCurrentInjector(BaseInputEncoder):
    """
    直接将观察向量（或其一部分）作为电流注入目标神经元群体。
    """
    def __init__(self, target_pop_name, num_neurons, observation_slice=None, scale_factor=1.0, **kwargs):
        """
        初始化直接电流注入器。

        参数:
            target_pop_name (str): 目标输入神经元群体的名称。
            num_neurons (int): 目标群体中的神经元数量。
                               此编码器期望观察向量（切片后）的维度与num_neurons匹配。
            observation_slice (slice, optional): 用于从观察向量中选择相关部分的切片对象。
                                             例如 `slice(0, 10)` 选择前10个元素。
                                             如果为None，则使用整个观察向量。
            scale_factor (float or np.array): 用于缩放观察值的乘数。
                                            如果是np.array，其长度必须与切片后的观察值或num_neurons匹配。
        """
        super().__init__(target_pop_name, **kwargs)
        self.num_neurons = num_neurons
        self.observation_slice = observation_slice
        self.scale_factor = scale_factor

    def encode(self, observation, dt=None, current_time=None):
        """
        将观察（或其切片）直接编码为注入电流。

        参数:
            observation (np.array or list): 观察向量。

        返回:
            dict: {self.target_pop_name: currents_array}
        """
        if not isinstance(observation, (np.ndarray, list)):
            raise ValueError(f"DirectCurrentInjector 需要一个向量观察，但接收到: {type(observation)}")
        
        obs_array = np.asarray(observation).flatten()

        if self.observation_slice:
            selected_obs = obs_array[self.observation_slice]
        else:
            selected_obs = obs_array

        if len(selected_obs) != self.num_neurons:
            raise ValueError(
                f"编码后选择的观察维度 ({len(selected_obs)}) "
                f"与目标群体神经元数量 ({self.num_neurons}) 不匹配。"
            )
        
        currents = selected_obs * self.scale_factor
        return {self.target_pop_name: currents.astype(float)}

    def __repr__(self):
        return (f"DirectCurrentInjector(target='{self.target_pop_name}', num_neurons={self.num_neurons}, "
                f"slice={self.observation_slice}, scale={self.scale_factor})")


# 针对 "MountainCar-v0" 环境，仅使用小车的位置信息。
# 环境观察: [position, velocity]
class MountainCarPositionEncoder(GaussianEncoder):
    """
    针对 MountainCar-v0 环境的输入编码器，仅使用小车的位置信息。
    继承自 GaussianEncoder。
    """
    def __init__(self, target_pop_name, num_neurons, 
                 pos_min=-1.2, pos_max=0.6, # MountainCar-v0 位置范围
                 sigma_scale=0.1, current_amplitude=10.0, 
                 **kwargs):
        """
        参数:
            pos_min (float): 小车位置的最小值。
            pos_max (float): 小车位置的最大值。
            其他参数同 GaussianEncoder。
        """
        super().__init__(target_pop_name, num_neurons, 
                         min_val=pos_min, max_val=pos_max, 
                         sigma_scale=sigma_scale, current_amplitude=current_amplitude, 
                         **kwargs)
    
    def encode(self, observation, dt=None, current_time=None):
        """
        编码 MountainCar 的观察。

        参数:
            observation (list or np.array): 环境的观察值 [position, velocity]。
        """
        if not hasattr(observation, '__len__') or len(observation) < 1:
            raise ValueError(f"MountainCarPositionEncoder 期望至少包含位置信息的观察，但接收到: {observation}")
        
        position = observation[0] # README 要求: "仅使用小车的位置信息 (忽略速度信息)"
        return super().encode(position, dt, current_time) # 调用父类的encode方法

    def __repr__(self):
        return (f"MountainCarPositionEncoder(target='{self.target_pop_name}', num_neurons={self.num_neurons}, "
                f"pos_range=[{self.min_val:.2f}, {self.max_val:.2f}], sigma_scale={self.sigma/self.val_range if self.val_range else 0:.2f}, "
                f"amplitude={self.current_amplitude:.2f})") 