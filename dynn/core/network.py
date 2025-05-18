# Placeholder for the NeuralNetwork object 

import numpy as np
# Attempt to import BaseProbe, assuming it's in a sibling directory utils
# This kind of relative import works when dynn is treated as a package.
from ..utils.probes import BaseProbe 

class NeuralNetwork:
    """
    顶层对象，表示和管理整个SNN，包括所有的神经元群体、突触连接及其学习规则。
    """
    def __init__(self, name="SNN"):
        self.name = name
        self.populations = {} # 存储神经元群体, key: name, value: NeuronPopulation
        self.synapses = {}    # 存储突触连接, key: name, value: SynapseCollection
        self.input_population_names = [] # 指定作为输入的神经元群体名称列表
        self.output_population_names = []# 指定作为输出的神经元群体名称列表
        
        self.probes = {} # 修改回字典：存储BaseProbe实例, key: probe_name

    def add_population(self, population):
        """
        将创建的神经元群体注册到网络对象中。
        """
        if population.name in self.populations:
            raise ValueError(f"名为 '{population.name}' 的神经元群体已存在。")
        self.populations[population.name] = population

    def add_synapses(self, synapse_collection):
        """
        将创建的突触连接注册到网络对象中。
        """
        if synapse_collection.name in self.synapses:
            raise ValueError(f"名为 '{synapse_collection.name}' 的突触连接已存在。")
        if synapse_collection.pre_pop.name not in self.populations:
            raise ValueError(f"突触前群体 '{synapse_collection.pre_pop.name}' 未在网络中注册。")
        if synapse_collection.post_pop.name not in self.populations:
            raise ValueError(f"突触后群体 '{synapse_collection.post_pop.name}' 未在网络中注册。")
        self.synapses[synapse_collection.name] = synapse_collection

    def set_input_populations(self, population_names):
        """
        指定网络中的哪些神经元群体作为外部输入的接收端。
        参数:
            population_names (list of str): 神经元群体名称的列表。
        """
        for name in population_names:
            if name not in self.populations:
                raise ValueError(f"名为 '{name}' 的神经元群体不存在于网络中，无法设为输入。")
        self.input_population_names = list(set(population_names)) # 去重并存储

    def set_output_populations(self, population_names):
        """
        指定网络中的哪些神经元群体作为产生行为输出的信号源。
        参数:
            population_names (list of str): 神经元群体名称的列表。
        """
        for name in population_names:
            if name not in self.populations:
                raise ValueError(f"名为 '{name}' 的神经元群体不存在于网络中，无法设为输出。")
        self.output_population_names = list(set(population_names)) # 去重并存储

    def get_population(self, name):
        if name not in self.populations:
            raise KeyError(f"网络中不存在名为 '{name}' 的神经元群体。")
        return self.populations[name]

    def get_synapses(self, name):
        if name not in self.synapses:
            raise KeyError(f"网络中不存在名为 '{name}' 的突触连接。")
        return self.synapses[name]

    # --- Probe Management --- 
    def add_probe(self, probe_instance):
        """
        向网络添加一个探针实例。
        参数:
            probe_instance (BaseProbe): BaseProbe的子类的实例。
        """
        if not isinstance(probe_instance, BaseProbe):
            raise TypeError("probe_instance 必须是 BaseProbe 的一个实例。")
        if probe_instance.name in self.probes: # 检查名称冲突
            raise ValueError(f"名为 '{probe_instance.name}' 的探针已存在.")
        self.probes[probe_instance.name] = probe_instance # 修改为字典赋值
        # print(f"探针 '{probe_instance.name}' 已添加到网络 '{self.name}'.")

    def _record_probes(self, current_time):
        """在每个仿真步骤中尝试记录所有已注册探针的数据。"""
        for probe in self.probes.values(): # 修改为遍历字典的值
            probe.attempt_record(self, current_time) # 传递网络自身和当前时间

    def get_probe_data(self, probe_name):
        """
        获取指定名称探针的已记录数据。
        参数:
            probe_name (str): 要获取数据的探针名称。
        返回:
            dict: 探针的数据，格式由探针的 get_data() 方法定义。
        异常:
            KeyError: 如果找不到具有指定名称的探针。
        """
        if probe_name in self.probes:
            return self.probes[probe_name].get_data()
        raise KeyError(f"名为 '{probe_name}' 的探针未在网络中找到。")
    
    def get_all_probes(self):
        """返回所有已注册探针的字典。""" # 修改文档字符串
        return self.probes # 直接返回字典
    # --- End Probe Management ---

    def step(self, input_currents_map, dt, current_time):
        """
        执行单个仿真步骤。

        参数:
            input_currents_map (dict): 
                一个字典，键是输入神经元群体的名称，值是注入该群体的电流向量 (np.array)。
                例如: {"input_pop1": np.array([...]), "input_pop2": np.array([...])}
            dt (float): 仿真时间步长。
            current_time (float): 当前仿真时间。

        返回:
            dict: 一个字典，键是输出神经元群体的名称，值是该群体当前的脉冲状态 (布尔数组)。
        """
        # 0. 准备: 获取所有群体的当前脉冲状态 (用于学习规则)
        #    实际上，学习规则可能需要的是上一步或当前正在计算的脉冲
        #    这里假设学习规则会在神经元更新后，使用新的脉冲状态

        all_current_inputs = {pop_name: np.zeros(len(pop)) for pop_name, pop in self.populations.items()}

        # 1. 应用外部输入电流
        for pop_name, currents in input_currents_map.items():
            if pop_name in self.input_population_names and pop_name in self.populations:
                if len(currents) != len(self.populations[pop_name]):
                    raise ValueError(f"输入电流向量长度与群体 '{pop_name}' 的神经元数量不匹配。")
                all_current_inputs[pop_name] += currents
            # else: 忽略不在指定输入群体列表中的输入，或者抛出警告/错误

        # 2. 计算突触传递的电流
        #    需要先获取所有突触前群体的脉冲状态 (通常是上一步的脉冲，或者在这一步开始时确定)
        #    简单起见，我们使用这一步开始时的脉冲（即神经元更新前的状态，但这在逻辑上不完全正确，
        #    因为脉冲产生后才应有电流传递）。
        #    更标准的做法是：
        #    a. 神经元更新 -> 产生脉冲 S_t
        #    b. 脉冲 S_t 通过突触传播 -> 产生突触后电流 I_syn_t (影响下一时间步)
        #    c. 学习规则应用，使用 S_t (pre) 和 S_t (post)
        #    或者，如果电流是瞬时作用的：
        #    a. 获取上一时间步的脉冲 S_{t-1}
        #    b. S_{t-1} 通过突触传播 -> 产生突触后电流 I_syn_t (影响当前时间步的神经元更新)
        #    c. 神经元根据 I_ext_t + I_syn_t 更新 -> 产生脉冲 S_t
        #    d. 学习规则应用，使用 S_t (pre) and S_t (post) (或者 S_{t-1} (pre) 和 S_t (post))
        #    README 中对 STDP 的描述是基于"突触前和突触后神经元脉冲迹"，暗示了需要知道脉冲发生的时间。

        # 假设我们有一个两阶段过程：
        # 阶段1: 收集所有上一时间步的脉冲 (或当前步骤开始时的状态，如果无延迟)
        # 这里我们先直接获取神经元群体在被更新前的 `fired`状态，如果这代表了上一步的脉冲。 
        # 但是 neuron.fired 是在 neuron.update() 内部被设置的。
        # 因此，我们需要一种方式获取上一步的脉冲。
        # 为简单起见，这里假设 `SynapseCollection.get_input_currents` 期望的是
        # *导致* 这些电流的突触前脉冲。在一个严格的事件驱动或时序模型中，这会是 *上一个* 时间步的脉冲。
        
        # 先收集所有群体上一时间步的脉冲状态，这里我们暂时假设在群体更新前，
        # .get_spikes() 返回的是上一步的结果，或者在simulator中管理脉冲的传递。
        # 为了简化，在这一步，我们先让所有神经元更新，然后计算突触电流，然后应用学习规则。
        # 这意味着突触电流是基于本轮更新后产生的脉冲，这会导致一个时间步的延迟，或者解释为电流在本步内传递。

        # --- 重新思考步骤 --- 
        # 1. (外部) 施加外部输入电流 `I_ext` 到输入神经元。
        # 2. (内部) 对于每个突触连接，从突触前群体获取脉冲，计算并累加突触后电流 `I_syn` 到突触后神经元。
        #    这需要突触前神经元的脉冲。如果这是一个迭代过程，我们应该用 *上一步* 的脉冲来计算 *当前步* 的 `I_syn`。
        #    或者，如果脉冲是即时传播的，并且网络中有循环，这会变得复杂。
        #    标准方法： S[t-1] -> W -> I_syn[t] -> V[t] update -> S[t]
        
        # 简化方案: 假设在 `network.step` 开始时，所有 population 的 `fired` 状态是来自 t-1
        # 那么 `syn_currents` 是基于 t-1 的脉冲
        
        # 收集所有群体当前的脉冲状态 (假设是 t-1 时刻的)
        # (这个状态应该在 Simulator 的循环中被正确管理和传递)
        # 这里我们直接获取，并假设它们代表了 *即将影响* 当前神经元更新的脉冲活动
        
        # 我们需要神经元更新前的脉冲状态来计算突触电流，
        # 然后用总电流更新神经元，得到新的脉冲状态，
        # 然后用新的脉冲状态（pre 和 post）来更新学习规则。

        # 存储所有群体在本轮更新前的脉冲 (作为突触前脉冲)
        pre_update_spikes_map = {pop_name: pop.get_spikes() for pop_name, pop in self.populations.items()}

        # 2. 计算由网络内部连接产生的突触电流
        for syn_name, syn_collection in self.synapses.items():
            pre_pop_name = syn_collection.pre_pop.name
            post_pop_name = syn_collection.post_pop.name
            
            # 使用更新前的脉冲状态作为突触前脉冲
            pre_spikes = pre_update_spikes_map[pre_pop_name]
            
            synaptic_currents = syn_collection.get_input_currents(pre_spikes)
            all_current_inputs[post_pop_name] += synaptic_currents

        # 3. 更新所有神经元群体状态
        #    并收集本轮更新后产生的新脉冲
        current_step_spikes_map = {}
        for pop_name, pop in self.populations.items():
            total_input_for_pop = all_current_inputs[pop_name]
            fired_mask = pop.update(total_input_for_pop, dt, current_time)
            current_step_spikes_map[pop_name] = fired_mask # fired_mask 即 pop.get_spikes() 的结果

        # 4. 应用学习规则 (使用本轮产生的突触前和突触后脉冲)
        for syn_name, syn_collection in self.synapses.items():
            if syn_collection.learning_rule:
                pre_pop_name = syn_collection.pre_pop.name
                post_pop_name = syn_collection.post_pop.name
                
                # STDP 通常基于 pre 和 post 在相近时间发生的脉冲
                # 所以使用 current_step_spikes (即本轮更新后新产生的脉冲)
                pre_spikes_for_stdp = current_step_spikes_map[pre_pop_name]
                post_spikes_for_stdp = current_step_spikes_map[post_pop_name]
                
                syn_collection.apply_learning_rule(
                    pre_spikes=pre_spikes_for_stdp,
                    post_spikes=post_spikes_for_stdp,
                    dt=dt,
                    current_time=current_time
                )
        
        # 5. 收集输出
        output_spikes_map = {}
        for pop_name in self.output_population_names:
            if pop_name in self.populations:
                output_spikes_map[pop_name] = current_step_spikes_map[pop_name]
        
        # 6. (可选) 更新探针数据 (在此处或 simulator 中完成)
        self._record_probes(current_time)

        return output_spikes_map

    def reset(self):
        """
        重置网络中所有神经元群体的状态，以及学习规则的内部状态（如果适用），还有探针数据。
        """
        for pop in self.populations.values():
            pop.reset_states() # 使用其默认或配置的初始分布重置
        
        # 重置学习规则内部状态 (例如，某些自适应学习率机制)
        for syn_collection in self.synapses.values():
            if hasattr(syn_collection.learning_rule, 'reset'):
                syn_collection.learning_rule.reset()
        
        # 重置探针数据
        for probe in self.probes.values(): # 修改为遍历字典的值
            probe.reset()
        
        # print(f"网络 '{self.name}' 已重置.")

    def __repr__(self):
        return f"NeuralNetwork(name='{self.name}', populations={len(self.populations)}, synapses={len(self.synapses)}, probes={len(self.probes)})" 