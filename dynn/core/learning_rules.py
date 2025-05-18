# Placeholder for learning rules (e.g., STDP) and reward modulation 

import numpy as np

class BaseLearningRule:
    """学习规则的基类。"""
    def __init__(self, lr_ltp=0.001, lr_ltd=0.001):
        self.lr_ltp = lr_ltp  # 长期增强的学习率
        self.lr_ltd = lr_ltd  # 长期削弱的学习率
        self.reward_modulation = 1.0 # 奖励调制因子，默认为1 (无调制)

    def update_weights(self, synapse_collection, pre_spikes, post_spikes, dt, current_time):
        """
        更新突触权重。
        子类必须实现此方法。

        参数:
            synapse_collection (SynapseCollection): 要更新权重的突触集合。
            pre_spikes (np.array): 突触前神经元的脉冲 (布尔数组)。
            post_spikes (np.array): 突触后神经元的脉冲 (布尔数组)。
            dt (float): 时间步长。
            current_time (float): 当前仿真时间。
        """
        raise NotImplementedError

    def set_reward_modulation(self, reward_signal):
        """
        根据外部奖励信号设置学习率调制因子。
        具体的调制逻辑可以在子类中定义，或者在这里提供一个通用实现。
        这里假设奖励信号直接作为调制因子，或者可以进行一些转换。
        """
        # README 要求: "调制机制应能以相同的方式 (例如，同向缩放或反向缩放) 
        # 影响权重的长期增强 (LTP) 和长期削弱 (LTD)。"
        # 一个简单的实现是直接将 reward_signal 用作一个乘性因子。
        # 如果 reward_signal 是平滑后的奖励，可以直接用。
        self.reward_modulation = reward_signal 

    def get_effective_lr_ltp(self):
        return self.lr_ltp * self.reward_modulation
    
    def get_effective_lr_ltd(self):
        return self.lr_ltd * self.reward_modulation

class TraceSTDP(BaseLearningRule):
    """
    基于脉冲迹的STDP (尖峰时间依赖可塑性) 规则。
    具体实现权重依赖的乘性STDP。
    """
    def __init__(self, lr_ltp=0.005, lr_ltd=0.005, 
                 tau_pre=20, tau_post=20, 
                 w_max=1.0, w_min=0.0, 
                 trace_increase=1.0):
        """
        初始化基于脉冲迹的STDP学习规则。

        参数:
            lr_ltp (float): LTP的基础学习率。
            lr_ltd (float): LTD的基础学习率。
            tau_pre (float): 突触前脉冲迹的时间常数 (ms)。
            tau_post (float): 突触后脉冲迹的时间常数 (ms)。
            w_max (float): 权重的最大值。
            w_min (float): 权重的最小值。
            trace_increase (float): 脉冲发生时，迹变量增加的固定值。
        """
        super().__init__(lr_ltp, lr_ltd)
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.w_max = w_max
        self.w_min = w_min
        self.trace_increase = trace_increase
        
        # 脉冲迹的衰减因子，假设dt=1ms。如果dt不同，需要在update中调整或传入dt
        # self.decay_pre = np.exp(-1.0/self.tau_pre) # dt=1ms
        # self.decay_post = np.exp(-1.0/self.tau_post) # dt=1ms
        # 在 update_traces 中处理衰减，以便适应可变的 dt

    def update_traces(self, population, spikes, trace_type, dt):
        """
        更新指定群体的脉冲迹。

        参数:
            population (NeuronPopulation): 神经元群体。
            spikes (np.array): 该群体的脉冲状态 (布尔数组)。
            trace_type (str): 'pre' 或 'post'，指示更新哪个迹。
            dt (float): 时间步长。
        """
        if trace_type == 'pre':
            trace_attr = 'spike_trace_pre' 
            tau = self.tau_pre
        elif trace_type == 'post':
            trace_attr = 'spike_trace_post'
            tau = self.tau_post
        else:
            raise ValueError("无效的迹类型")

        current_traces = getattr(population, trace_attr)
        
        # 衰减
        decay_factor = np.exp(-dt / tau) if tau > 0 else 0
        new_traces = current_traces * decay_factor
        
        # 增加 (对于发放脉冲的神经元)
        new_traces[spikes] += self.trace_increase
        
        setattr(population, trace_attr, new_traces)

    def update_weights(self, synapse_collection, pre_spikes, post_spikes, dt, current_time):
        """
        应用STDP规则更新权重。
        实现权重依赖的乘性STDP。
        dW_LTP = lr_LTP * pre_trace * (w_max - W)  (当突触后脉冲发生)
        dW_LTD = -lr_LTD * post_trace * (W - w_min) (当突触前脉冲发生)
        """
        pre_pop = synapse_collection.pre_pop
        post_pop = synapse_collection.post_pop
        weights = synapse_collection.weights
        connection_mask = synapse_collection.connection_mask

        # 1. 更新脉冲迹
        self.update_traces(pre_pop, pre_spikes, 'pre', dt)
        self.update_traces(post_pop, post_spikes, 'post', dt)

        trace_pre_values = pre_pop.spike_trace_pre
        trace_post_values = post_pop.spike_trace_post
        
        effective_lr_ltp = self.get_effective_lr_ltp()
        effective_lr_ltd = self.get_effective_lr_ltd()

        # 2. 计算权重变化
        # 遍历所有存在的连接
        # 向量化操作会更高效，但需要仔细构造
        # (num_post, num_pre)
        
        # LTP: 当post神经元发放脉冲时
        # post_spikes 是 (num_post,) 的向量
        # trace_pre_values 是 (num_pre,) 的向量
        # weights 是 (num_post, num_pre) 的矩阵
        
        # LTP part: triggered by post-synaptic spikes
        # For each post-synaptic neuron `i` that spiked:
        #   For each pre-synaptic neuron `j` connected to `i`:
        #     delta_w_ij_ltp = effective_lr_ltp * trace_pre_values[j] * (self.w_max - weights[i, j])
        
        # LTD part: triggered by pre-synaptic spikes
        # For each pre-synaptic neuron `j` that spiked:
        #   For each post-synaptic neuron `i` connected to `j`:
        #     delta_w_ij_ltd = -effective_lr_ltd * trace_post_values[i] * (weights[i, j] - self.w_min)

        delta_w = np.zeros_like(weights)

        # 处理LTP (当突触后神经元 i 发放脉冲时)
        # post_spikes_expanded.shape = (num_post, 1)
        post_spikes_expanded = post_spikes[:, np.newaxis]
        # trace_pre_expanded.shape = (1, num_pre)
        trace_pre_expanded = trace_pre_values[np.newaxis, :]
        
        # LTP 贡献: 只对发放脉冲的突触后神经元，并且考虑其对应的突触前迹
        # (w_max - W) 项
        factor_ltp = (self.w_max - weights)
        # 如果是抑制性连接，权重是负的，w_max 通常是0， (w_max - W) 应该变成 (-w_min - (-|W|)) = |W| - w_min
        # 为了简化，我们假设 w_min <= W <= w_max 适用于权重的绝对值，或者分别处理兴奋性和抑制性
        # 目前假设 w_min, w_max 是针对权重的绝对值，或者传入的权重已经是处理过的
        # README: "具体的数学形式和参数需通过实验来优化和最终确定。"
        # 这里采用标准形式，假设权重是正的，或者w_min/w_max适应了权重的符号。
        # 对于乘性STDP，通常 w_min >= 0。
        # 如果允许负权重， (w_max - W) 对负权重意味着增强幅度更大 (更负)。
        # (W - w_min) 对负权重意味着削弱幅度更大 (更接近0)。
        
        # 假设w_max > w_min，并且权重在此范围内。
        # 如果 synapse_collection.is_excitatory is False, 权重是负的。
        # 假设 w_min, w_max 是针对兴奋性权重的范围 [0, w_max_exc] 或抑制性权重的范围 [-w_max_inh, 0]
        # 为简单起见，此处的 w_min, w_max 直接用于当前权重值。
        
        dw_ltp = effective_lr_ltp * trace_pre_expanded * factor_ltp
        delta_w += post_spikes_expanded * dw_ltp # 只有当post神经元发放脉冲时，LTP才发生

        # 处理LTD (当突触前神经元 j 发放脉冲时)
        # pre_spikes_expanded.shape = (1, num_pre)
        pre_spikes_expanded = pre_spikes[np.newaxis, :]
        # trace_post_expanded.shape = (num_post, 1)
        trace_post_expanded = trace_post_values[:, np.newaxis]
        
        # LTD 贡献: 只对发放脉冲的突触前神经元，并且考虑其对应的突触后迹
        factor_ltd = (weights - self.w_min)
        dw_ltd = -effective_lr_ltd * trace_post_expanded * factor_ltd
        delta_w += pre_spikes_expanded * dw_ltd # 只有当pre神经元发放脉冲时，LTD才发生

        # 应用变化 (仅对存在的连接)
        synapse_collection.weights += delta_w * connection_mask
        
        # 权重裁剪 (确保在 [w_min, w_max] 或其他定义的范围内)
        # 需要考虑兴奋性/抑制性。如果 synapse_collection.is_excitatory is True, 裁剪到 [max(0, w_min), w_max]
        # 如果 False, 裁剪到 [w_min, min(0, w_max)] (假设w_min < 0, w_max <= 0 for inhibitory)
        # 或者，让w_min, w_max 定义绝对值范围，然后强制符号。

        if synapse_collection.is_excitatory:
            actual_w_min = max(0, self.w_min) # 兴奋性权重通常非负
            actual_w_max = self.w_max
            np.clip(synapse_collection.weights, actual_w_min, actual_w_max, out=synapse_collection.weights)
        else: # 抑制性
            actual_w_max = min(0, self.w_max) # 抑制性权重通常非正
            actual_w_min = self.w_min 
            np.clip(synapse_collection.weights, actual_w_min, actual_w_max, out=synapse_collection.weights)
        
        # 确保未连接的权重仍然为0
        synapse_collection.weights *= connection_mask

    def __repr__(self):
        return f"TraceSTDP(lr_ltp={self.lr_ltp}, lr_ltd={self.lr_ltd}, tau_pre={self.tau_pre}, tau_post={self.tau_post})" 