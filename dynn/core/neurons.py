# Placeholder for neuron models and NeuronPopulation 

import numpy as np

class IzhikevichNeuron:
    """
    实现一个可参数化的类Izhikevich尖峰神经元模型。
    允许对标准Izhikevich模型进行简化或修改，以平衡生物学合理性和计算效率。
    """
    def __init__(self, a=0.02, b=0.2, c=-65.0, d=8.0, v_thresh=30.0, initial_v=-70.0, initial_u=None):
        """
        初始化Izhikevich神经元。

        参数:
            a (float): 时间尺度参数。
            b (float): 敏感度参数。
            c (float): 膜电位重置值。
            d (float): 恢复变量重置增加值。
            v_thresh (float): 脉冲发放阈值。
            initial_v (float): 初始膜电位。
            initial_u (float, optional): 初始恢复变量。如果为None，则根据 'b' 和 'initial_v' 计算。
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v_thresh = v_thresh

        self.v = float(initial_v)  # 确保是 float
        if initial_u is None:
            self.u = self.b * self.v
        else:
            self.u = float(initial_u) # 确保是 float
        
        self.fired = False # 当前时间步是否发放脉冲
        self.last_spike_time = -np.inf # 上次脉冲发放的时间

    def update(self, I_inj, dt):
        """
        根据注入电流和时间步长更新神经元状态。

        参数:
            I_inj (float): 注入神经元的电流。
            dt (float): 仿真时间步长。
        """
        self.fired = False
        if self.v >= self.v_thresh:
            self.v = self.c  # 立即重置 v
            self.u += self.d # 更新 u
            self.fired = True
            # self.last_spike_time will be updated by NeuronPopulation or simulator
            return self.fired # 脉冲发放后，直接返回，不再进行后续的v, u更新

        # 使用更简单的欧拉积分方法更新 v 和 u
        # dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        # du/dt = a*(b*v - u)
        # 这里简化模型，直接分步更新，并假设dt足够小
        # 或者使用更精确的积分，但 README 中提到 "允许对标准Izhikevich模型进行简化或修改"
        
        # 按照Izhikevich论文中的数值模拟方法，分两步更新以提高精度
        # For numerical stability, the Euler method with dt = 0.5 ms (or 0.25ms)
        # can be used if one updates v twice per time step:
        # v = v + dt/2 * (0.04*v^2 + 5*v + 140 - u + I)
        # v = v + dt/2 * (0.04*v^2 + 5*v + 140 - u + I)
        # u = u + dt * a*(b*v_old - u) (where v_old is v before the first half-step update)

        # 简单欧拉法
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I_inj)
        du = self.a * (self.b * self.v - self.u)

        self.v += dv * dt
        self.u += du * dt
        
        # 确保v不超过阈值（如果超过，则在下一次update开始时处理）
        # 但Izhikevich原模型是在v达到阈值后立即重置，然后再计算dv, du
        # 为简单起见，这里先更新，然后在下一次迭代开始时检查并发放脉冲和重置

        return self.fired

    def get_state(self):
        """
        返回神经元的关键内部状态变量。
        """
        return {
            "v": self.v,
            "u": self.u,
            "fired": self.fired,
            "last_spike_time": self.last_spike_time
        }

    def reset(self, initial_v=-70.0, initial_u=None):
        """
        重置神经元状态到初始值。
        """
        self.v = initial_v
        if initial_u is None:
            self.u = self.b * self.v
        else:
            self.u = initial_u
        self.fired = False
        self.last_spike_time = -np.inf


class NeuronPopulation:
    """
    管理一组神经元。
    该抽象层应能高效地执行针对整个群体的操作，
    例如参数设置、状态更新、脉冲收集等。
    """
    def __init__(self, num_neurons, neuron_model_class=IzhikevichNeuron, neuron_params=None, 
                 initial_v_dist=None, initial_u_dist=None, name="population"):
        """
        初始化神经元群体。

        参数:
            num_neurons (int): 群体中的神经元数量。
            neuron_model_class (class): 用于创建神经元的类，默认为IzhikevichNeuron。
            neuron_params (dict or list of dicts, optional): 
                单个神经元模型的参数。
                如果是dict，则所有神经元共享这些参数。
                如果是list of dicts，则每个神经元使用对应的参数字典。
            initial_v_dist (tuple or float, optional): 
                膜电位v的初始值或分布。
                如果是float，则所有神经元v初始化为该值。
                如果是tuple (type, params)，例如 ('uniform', (low, high)) 或 ('normal', (mean, std))。
            initial_u_dist (tuple or float, optional): 
                恢复变量u的初始值或分布。
                如果是float，则所有神经元u初始化为该值。
                如果是tuple (type, params)。如果为None，则根据b和v计算。
            name (str): 群体的名称。
        """
        self.num_neurons = num_neurons
        self.name = name
        self.neurons = []

        if neuron_params is None:
            neuron_params = {} # 使用默认参数

        for i in range(num_neurons):
            params_i = neuron_params if isinstance(neuron_params, dict) else neuron_params[i]
            
            # 处理初始状态
            # current_initial_v 和 current_initial_u 来自 initial_v_dist 和 initial_u_dist
            # 如果它们在 params_i 中也存在，则 current_initial_v/u (来自 dist) 优先
            
            # 从 neuron_params 复制，并允许 initial_v/u_dist 覆盖
            effective_params = params_i.copy()

            current_initial_v = self._get_initial_value(initial_v_dist, default_val=effective_params.pop('initial_v', -70.0))
            current_initial_u = self._get_initial_value(initial_u_dist, default_val=effective_params.pop('initial_u', None))


            if current_initial_u is None: # 如果 dist 或 params_i 都没有提供 initial_u
                # 尝试从 effective_params (可能来自 neuron_params) 或默认值获取 b
                b_val = effective_params.get('b', self.get_default_neuron_param('b')) 
                current_initial_u = b_val * current_initial_v
            
            neuron = neuron_model_class(initial_v=current_initial_v, initial_u=current_initial_u, **effective_params)
            self.neurons.append(neuron)
            
        self.spike_trace_pre = np.zeros(num_neurons) # 突触前脉冲迹 (用于STDP)
        self.spike_trace_post = np.zeros(num_neurons) # 突触后脉冲迹 (用于STDP)


    def _get_initial_value(self, dist_config, default_val):
        if dist_config is None:
            return default_val
        if isinstance(dist_config, (int, float)):
            return float(dist_config)
        elif isinstance(dist_config, tuple) and len(dist_config) == 2:
            dist_type, params = dist_config
            if dist_type == 'uniform':
                return np.random.uniform(params[0], params[1])
            elif dist_type == 'normal':
                return np.random.normal(params[0], params[1])
            else:
                raise ValueError(f"不支持的分布类型: {dist_type}")
        else:
            raise ValueError(f"无效的初始值配置: {dist_config}")

    def get_default_neuron_param(self, param_name):
        """辅助方法，获取神经元模型的默认参数值。"""
        # 这可以更通用，但目前只针对 IzhikevichNeuron
        if hasattr(IzhikevichNeuron, param_name):
            return getattr(IzhikevichNeuron(), param_name) # 创建一个临时实例以获取默认值
        
        # 或者从 __init__ 的默认值获取，但这更复杂
        # inspect.signature(IzhikevichNeuron.__init__).parameters[param_name].default
        # 为简化，这里硬编码一些关键的
        defaults = {'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0, 'v_thresh': 30.0}
        if param_name in defaults:
            return defaults[param_name]
        return None # 或抛出错误

    def update(self, I_inj_vector, dt, current_time):
        """
        更新群体中所有神经元的状态。

        参数:
            I_inj_vector (np.array): 长度为 num_neurons 的数组，包含每个神经元的注入电流。
            dt (float): 仿真时间步长。
            current_time (float): 当前仿真时间。

        返回:
            np.array: 布尔数组，指示哪些神经元发放了脉冲。
        """
        fired_mask = np.zeros(self.num_neurons, dtype=bool)
        for i, neuron in enumerate(self.neurons):
            if neuron.update(I_inj_vector[i], dt):
                fired_mask[i] = True
                neuron.last_spike_time = current_time
        return fired_mask

    def get_spikes(self):
        """
        获取当前时间步群体中所有神经元的脉冲状态。
        """
        return np.array([neuron.fired for neuron in self.neurons], dtype=bool)

    def get_all_states(self, state_keys=None):
        """
        获取群体中所有神经元指定的内部状态。

        参数:
            state_keys (list of str, optional): 要获取的状态变量名称列表。
                                             如果为None，则获取所有可用状态。

        返回:
            dict: 键为状态名称，值为包含所有神经元该状态值的列表或数组。
        """
        if not self.neurons:
            return {}

        # 获取第一个神经元的状态作为模板，以确定可用的键
        sample_state = self.neurons[0].get_state()
        if state_keys is None:
            state_keys = list(sample_state.keys())

        all_states = {key: [] for key in state_keys}
        for neuron in self.neurons:
            state = neuron.get_state()
            for key in state_keys:
                if key in state:
                    all_states[key].append(state[key])
        
        # 转换为numpy数组以便于处理
        for key in all_states:
            all_states[key] = np.array(all_states[key])
            
        return all_states

    def set_parameters(self, neuron_indices, param_name, param_value):
        """
        为指定的神经元设置参数。

        参数:
            neuron_indices (int or list or np.array): 要修改参数的神经元索引。
            param_name (str): 要修改的参数名称 (例如 'a', 'b', 'c', 'd')。
            param_value (float or list or np.array): 新的参数值。
                                                    如果neuron_indices是单个索引，则param_value是单个值。
                                                    如果是多个索引，则param_value可以是单个值（应用于所有指定神经元）
                                                    或与neuron_indices长度相同的值列表/数组。
        """
        if isinstance(neuron_indices, int):
            neuron_indices = [neuron_indices]
        
        if not hasattr(param_value, '__iter__') or isinstance(param_value, str): # 单个值应用于所有
            param_values = [param_value] * len(neuron_indices)
        else: # 列表或数组
            if len(param_value) != len(neuron_indices):
                raise ValueError("param_value 的长度必须与 neuron_indices 的长度匹配 (如果 param_value 是列表/数组)")
            param_values = param_value

        for idx, value in zip(neuron_indices, param_values):
            if idx < 0 or idx >= self.num_neurons:
                print(f"警告: 索引 {idx} 超出范围，跳过。")
                continue
            neuron = self.neurons[idx]
            if hasattr(neuron, param_name):
                setattr(neuron, param_name, value)
                # 特殊处理：如果修改了参数b，可能需要更新u的平衡点，但这通常在reset或初始化时处理。
                # 为简单起见，这里只直接设置参数。
                if param_name == 'b' and not hasattr(neuron, '_u_explicitly_set'): # 假设如果u是显式设置的，则不自动调整
                    neuron.u = neuron.b * neuron.v # 重新计算u的平衡点（简化）
            else:
                print(f"警告: 神经元对象没有参数 '{param_name}'。")

    def reset_states(self, initial_v_dist=None, initial_u_dist=None):
        """
        重置群体中所有神经元的状态。
        """
        for neuron in self.neurons:
            current_initial_v = self._get_initial_value(initial_v_dist, default_val=-70.0)
            current_initial_u = self._get_initial_value(initial_u_dist, default_val=None)
            
            if current_initial_u is None:
                 current_initial_u = neuron.b * current_initial_v

            neuron.reset(initial_v=current_initial_v, initial_u=current_initial_u)
        
        self.spike_trace_pre.fill(0)
        self.spike_trace_post.fill(0)

    def __len__(self):
        return self.num_neurons

    def __getitem__(self, index):
        return self.neurons[index]

    def __repr__(self):
        return f"NeuronPopulation(name='{self.name}', num_neurons={self.num_neurons}, model={self.neurons[0].__class__.__name__ if self.neurons else 'N/A'})" 