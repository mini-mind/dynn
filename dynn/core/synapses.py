# Placeholder for synapse and connection management 

import numpy as np

class SynapseCollection:
    """
    定义和管理神经元之间（或群体之间）的突触连接。
    """
    def __init__(self, pre_population, post_population, name="synapses"):
        """
        初始化突触集合。

        参数:
            pre_population (NeuronPopulation): 突触前神经元群体。
            post_population (NeuronPopulation): 突触后神经元群体。
            name (str): 突触集合的名称。
        """
        self.pre_pop = pre_population
        self.post_pop = post_population
        self.name = name

        # 权重矩阵，维度: (num_post_neurons, num_pre_neurons)
        # weights[i, j] 表示从 pre_pop[j] 到 post_pop[i] 的连接权重
        self.weights = np.zeros((len(post_population), len(pre_population)))
        
        # 连接掩码，标记哪些连接存在 (对于稀疏连接)
        self.connection_mask = np.zeros_like(self.weights, dtype=bool)

        # 延迟矩阵 (可选，这里简化为无延迟或单一延迟)
        # self.delays = np.ones_like(self.weights) # 假设单位延迟

        self.learning_rule = None # 稍后可以附加学习规则
        self.is_excitatory = True # 默认为兴奋性，可以通过配置或权重符号区分

    def initialize_weights(self, dist_config, connectivity_type='full', **kwargs):
        """
        根据用户定义的统计分布和连接类型初始化突触权重。

        参数:
            dist_config (tuple or float): 
                权重值的初始分布。
                如果是float，则所有权重初始化为该值。
                如果是tuple (type, params)，例如 ('uniform', (low, high)) 或 ('normal', (mean, std))。
            connectivity_type (str): 连接类型，例如 'full', 'sparse_prob', 'sparse_num', 'neighborhood'.
            **kwargs: 特定连接类型的参数，例如:
                for 'sparse_prob': prob (float) - 连接概率。
                for 'sparse_num': num_connections (int) - 每个突触后神经元的连接数。
                for 'neighborhood': radius (int) - 邻域半径 (假设一维排列)。
                                    allow_self_connections (bool) - 是否允许自连接 (如果pre=post)。
        """
        num_post = len(self.post_pop)
        num_pre = len(self.pre_pop)

        # 1. 生成权重值
        if isinstance(dist_config, (int, float)):
            initial_weights = np.full((num_post, num_pre), float(dist_config))
        elif isinstance(dist_config, tuple) and len(dist_config) == 2:
            dist_type, params = dist_config
            if dist_type == 'uniform':
                initial_weights = np.random.uniform(params[0], params[1], size=(num_post, num_pre))
            elif dist_type == 'normal':
                initial_weights = np.random.normal(params[0], params[1], size=(num_post, num_pre))
            else:
                raise ValueError(f"不支持的权重分布类型: {dist_type}")
        else:
            raise ValueError(f"无效的权重分布配置: {dist_config}")

        # 2. 根据连接类型应用权重并设置连接掩码
        if connectivity_type == 'full':
            self.weights = initial_weights
            self.connection_mask = np.ones_like(self.weights, dtype=bool)
        elif connectivity_type == 'sparse_prob':
            prob = kwargs.get('prob', 0.1)
            self.connection_mask = np.random.rand(num_post, num_pre) < prob
            self.weights = np.where(self.connection_mask, initial_weights, 0)
        elif connectivity_type == 'sparse_num':
            num_connections = kwargs.get('num_connections', int(0.1 * num_pre))
            if num_connections > num_pre:
                num_connections = num_pre 
            self.connection_mask.fill(False)
            for i in range(num_post):
                if num_pre > 0 :
                    chosen_indices = np.random.choice(num_pre, num_connections, replace=False)
                    self.connection_mask[i, chosen_indices] = True
            self.weights = np.where(self.connection_mask, initial_weights, 0)
        elif connectivity_type == 'neighborhood':
            # 仅当 pre_population 和 post_population 是同一个群体时才有意义，或者它们有某种空间映射
            # 这里简化为 pre_pop 和 post_pop 具有相同的神经元数量和一维排列
            if num_pre != num_post:
                # 对于不同群体间的邻域连接，需要更复杂的定义，这里仅支持群体内或大小相同的群体
                print("警告: 邻域连接目前主要针对相同大小的群体或群体内连接进行简化实现。")
            
            radius = kwargs.get('radius', 1)
            allow_self = kwargs.get('allow_self_connections', False)
            self.connection_mask.fill(False)
            for i in range(num_post):
                for j in range(num_pre):
                    if abs(i - j) <= radius:
                        if not allow_self and i == j and self.pre_pop is self.post_pop:
                            continue
                        self.connection_mask[i, j] = True
            self.weights = np.where(self.connection_mask, initial_weights, 0)
        else:
            raise ValueError(f"不支持的连接类型: {connectivity_type}")
        
        # 根据兴奋性/抑制性调整权重（如果需要，可以在此强制权重符号）
        # 例如，如果 is_excitatory=False，则 self.weights = -np.abs(self.weights)
        if not self.is_excitatory:
             self.weights[self.connection_mask] = -np.abs(self.weights[self.connection_mask])
        else:
             self.weights[self.connection_mask] = np.abs(self.weights[self.connection_mask])


    def set_excitatory(self, is_excitatory):
        """ 设置突触是兴奋性还是抑制性，并相应调整权重符号 """
        if self.is_excitatory == is_excitatory:
            return
        self.is_excitatory = is_excitatory
        if is_excitatory:
            self.weights[self.connection_mask] = np.abs(self.weights[self.connection_mask])
        else:
            self.weights[self.connection_mask] = -np.abs(self.weights[self.connection_mask])

    def get_input_currents(self, pre_spikes):
        """
        根据突触前脉冲计算传递给突触后神经元的总输入电流。

        参数:
            pre_spikes (np.array): 布尔数组，指示突触前神经元群体中哪些神经元发放了脉冲。

        返回:
            np.array: 长度为 num_post_neurons 的数组，包含每个突触后神经元接收到的总电流。
        """
        if len(pre_spikes) != self.weights.shape[1]:
            raise ValueError("突触前脉冲数组的长度与权重矩阵的列数不匹配。")
        
        # 仅考虑实际存在的连接 (通过 connection_mask)
        effective_weights = self.weights * self.connection_mask
        
        # 电流 = 权重 * 脉冲 (这里脉冲是0或1)
        # I_post = W_post_pre * S_pre
        input_currents = np.dot(effective_weights, pre_spikes.astype(float))
        return input_currents

    def apply_learning_rule(self, pre_spikes, post_spikes, dt, current_time):
        """
        如果附加了学习规则，则应用它来更新权重。
        """
        if self.learning_rule:
            self.learning_rule.update_weights(
                synapse_collection=self,
                pre_spikes=pre_spikes,
                post_spikes=post_spikes,
                dt=dt,
                current_time=current_time
            )

    def set_learning_rule(self, learning_rule_instance):
        self.learning_rule = learning_rule_instance

    def get_weights(self):
        return self.weights
    
    def get_connection_mask(self):
        return self.connection_mask

    def __repr__(self):
        return "SynapseCollection(name='%s', pre='%s', post='%s', shape=%s, connections=%d)" % \
               (self.name, self.pre_pop.name, self.post_pop.name, 
                repr(self.weights.shape), #确保shape是标准的repr形式
                np.sum(self.connection_mask))

# 可以在这里添加更具体的连接管理类，如 ConnectionManager，如果需要更复杂的连接管理逻辑
# 但 SynapseCollection 目前已涵盖了 README 2.2 的主要要求。 