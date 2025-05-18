# Placeholder for data recording and monitoring probes 

import numpy as np
import csv

class BaseProbe:
    """探针的基类，用于记录仿真过程中的数据。"""
    def __init__(self, name, record_interval=1):
        """
        初始化基础探针。

        参数:
            name (str): 探针的唯一名称。
            record_interval (int): 每隔多少个仿真步记录一次数据。
        """
        self.name = name
        if not isinstance(record_interval, int) or record_interval <= 0:
            raise ValueError("record_interval 必须是一个正整数。")
        self.record_interval = record_interval
        self.data = {}  # 存储不同变量的时间序列数据
        self.time_data = []  # 存储记录数据点的时间戳
        self._steps_since_last_record = 0

    def attempt_record(self, network, current_time_ms):
        """
        尝试记录数据。只有在达到 record_interval 时才实际记录。
        由 Simulator 或 Network 在每个时间步调用。
        """
        self._steps_since_last_record += 1
        if self._steps_since_last_record >= self.record_interval:
            self._steps_since_last_record = 0
            self.time_data.append(current_time_ms)
            self._collect_data(network, current_time_ms)
            return True
        return False

    def _collect_data(self, network, current_time_ms):
        """
        实际的数据收集逻辑。子类必须实现此方法。
        当满足记录条件时，由 attempt_record 调用。

        参数:
            network (NeuralNetwork): 神经网络实例，用于从中获取数据。
            current_time_ms (float): 当前的仿真时间（毫秒）。
        """
        raise NotImplementedError("子类必须实现 _collect_data 方法。")

    def get_data(self):
        """
        返回记录的数据的副本。

        返回:
            dict: 包含 'time' (时间戳列表) 和 'data' (变量数据字典) 的字典。
                  其中 'data' 的每个值都是一个列表，对应时间戳。
        """
        # 返回数据的深拷贝以防止外部修改，特别是对于包含numpy数组的列表
        data_copy = {}
        for key, val_list in self.data.items():
            data_copy[key] = [v.copy() if isinstance(v, np.ndarray) else v for v in val_list]
        
        return {'time': list(self.time_data), 
                'data': data_copy}

    def reset(self):
        """
        重置探针的内部状态和所有已记录的数据。
        """
        self.data = {k: [] for k in self.data} # 清空列表，但保留键结构
        self.time_data = []
        self._steps_since_last_record = 0
        # print(f"探针 '{self.name}' 已重置.")

    def export_to_csv(self, filepath, delimiter=','):
        """
        将记录的数据导出到CSV文件。

        参数:
            filepath (str): 要保存CSV文件的路径。
            delimiter (str): CSV文件的分隔符。
        """
        all_recorded_data = self.get_data() # 获取数据的副本
        times = all_recorded_data['time']
        
        data_keys = list(all_recorded_data['data'].keys())
        
        if not times:
            print(f"探针 '{self.name}' 没有数据点可导出到CSV。")
            # 创建一个空的带有表头的CSV，或者什么都不做
            if data_keys: # 如果有定义的数据变量，至少写表头
                 with open(filepath, 'w', newline='') as csvfile:
                    header = ['time'] + data_keys
                    writer = csv.writer(csvfile, delimiter=delimiter)
                    writer.writerow(header)
            return

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter)
            
            # 构建表头
            header = ['time']
            # 检查第一个数据点以确定每个变量的列数 (用于向量/数组数据)
            first_data_values = {key: all_recorded_data['data'][key][0] for key in data_keys if all_recorded_data['data'][key]}

            structured_keys = [] # (key_name, num_elements_or_1_if_scalar)
            for key in data_keys:
                val = first_data_values.get(key)
                if isinstance(val, (list, np.ndarray)):
                    num_elements = len(val)
                    for i in range(num_elements):
                        header.append(f"{key}_{i}")
                    structured_keys.append((key, num_elements))
                else: # 标量
                    header.append(key)
                    structured_keys.append((key, 1))
            
            writer.writerow(header)

            # 写入数据行
            for i, t in enumerate(times):
                row = [t]
                for key, num_elements in structured_keys:
                    value_at_t_list = all_recorded_data['data'][key]
                    if i < len(value_at_t_list):
                        value_at_t = value_at_t_list[i]
                        if value_at_t is None: # 处理可能的None值 (如果记录失败)
                            row.extend([None] * num_elements)
                        elif num_elements > 1: # 列表或数组
                            row.extend(value_at_t)
                        else: # 标量
                            row.append(value_at_t)
                    else: # 数据缺失的情况（理论上不应发生，如果time_data和data同步）
                        row.extend([None] * num_elements)
                writer.writerow(row)
        print(f"探针 '{self.name}' 数据已导出到 {filepath}")

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', interval={self.record_interval}, records={len(self.time_data)})"


class PopulationProbe(BaseProbe):
    """记录神经元群体状态的探针。"""
    def __init__(self, name, population_name, state_vars, record_interval=1):
        """
        参数:
            population_name (str): 要探测的神经元群体名称。
            state_vars (list of str): 要记录的状态变量列表 (例如 ['v', 'fired'])。
        """
        super().__init__(name, record_interval)
        if not population_name or not isinstance(population_name, str):
            raise ValueError("population_name 必须是一个非空字符串。")
        if not state_vars or not isinstance(state_vars, (list, tuple)):
            raise ValueError("state_vars 必须是一个包含变量名的列表/元组。")
            
        self.population_name = population_name
        self.state_vars = list(state_vars)
        # 初始化data字典的键
        for var in self.state_vars:
            self.data[var] = []

    def _collect_data(self, network, current_time_ms):
        if self.population_name not in network.populations:
            print(f"警告 (探针 '{self.name}'): 群体 '{self.population_name}' 在网络中未找到。跳过此时间点记录。")
            # 为了保持数据结构的完整性，可以为每个变量记录None或空占位符
            for var in self.state_vars:
                self.data[var].append(None) # 或者一个与群体大小匹配的None数组
            return

        pop = network.populations[self.population_name]
        collected_states = pop.get_all_states(self.state_vars)
        
        for var in self.state_vars:
            if var in collected_states:
                # .copy() 对于numpy数组很重要
                self.data[var].append(collected_states[var].copy() if isinstance(collected_states[var], np.ndarray) else collected_states[var])
            else:
                print(f"警告 (探针 '{self.name}'): 状态变量 '{var}' 在群体 '{self.population_name}' 中未找到。")
                self.data[var].append(None) # 或者一个与群体大小匹配的None数组
    
    def __repr__(self):
        return f"PopulationProbe(name='{self.name}', pop='{self.population_name}', vars={self.state_vars}, interval={self.record_interval}, records={len(self.time_data)})"


class SynapseProbe(BaseProbe):
    """记录突触集合状态的探针，例如权重。"""
    def __init__(self, name, synapse_collection_name, record_weights=True, record_interval=1):
        super().__init__(name, record_interval)
        if not synapse_collection_name or not isinstance(synapse_collection_name, str):
            raise ValueError("synapse_collection_name 必须是一个非空字符串。")
            
        self.synapse_collection_name = synapse_collection_name
        self.record_weights = record_weights
        if self.record_weights:
            self.data['weights'] = []
        # 可以扩展到记录其他突触相关变量

    def _collect_data(self, network, current_time_ms):
        if self.synapse_collection_name not in network.synapses:
            print(f"警告 (探针 '{self.name}'): 突触集合 '{self.synapse_collection_name}' 在网络中未找到。跳过记录。")
            if self.record_weights: self.data['weights'].append(None)
            return

        syn_collection = network.synapses[self.synapse_collection_name]
        if self.record_weights:
            self.data['weights'].append(syn_collection.get_weights().copy()) # 存储副本
            
    def __repr__(self):
        return f"SynapseProbe(name='{self.name}', syn='{self.synapse_collection_name}', weights={self.record_weights}, interval={self.record_interval}, records={len(self.time_data)})"


class CustomDataProbe(BaseProbe):
    """
    一个通用探针，用于记录由用户提供的函数在每个记录点返回的数据。
    """
    def __init__(self, name, data_provider_fn, data_keys, record_interval=1):
        """
        参数:
            name (str): 探针名称。
            data_provider_fn (callable): 一个函数，签名 fn(network, current_time_ms)，
                                       应返回一个字典，其键与 data_keys 匹配。
            data_keys (list of str): 期望从 data_provider_fn 返回的字典中的键。
            record_interval (int): 记录间隔。
        """
        super().__init__(name, record_interval)
        if not callable(data_provider_fn):
            raise TypeError("data_provider_fn 必须是一个可调用对象。")
        if not data_keys or not isinstance(data_keys, (list, tuple)):
            raise ValueError("data_keys 必须是一个包含键名的列表/元组。")
            
        self.data_provider_fn = data_provider_fn
        self.data_keys = list(data_keys)
        for key in self.data_keys:
            self.data[key] = []

    def _collect_data(self, network, current_time_ms):
        try:
            custom_data = self.data_provider_fn(network, current_time_ms)
            if not isinstance(custom_data, dict):
                print(f"警告 (探针 '{self.name}'): data_provider_fn 未返回字典。跳过记录。")
                for key in self.data_keys: self.data[key].append(None)
                return
            
            for key in self.data_keys:
                if key in custom_data:
                    val = custom_data[key]
                    self.data[key].append(val.copy() if isinstance(val, np.ndarray) else val)
                else:
                    print(f"警告 (探针 '{self.name}'): data_provider_fn 返回的数据中缺少键 '{key}'。")
                    self.data[key].append(None)
        except Exception as e:
            print(f"错误 (探针 '{self.name}'): 调用 data_provider_fn 时出错: {e}")
            for key in self.data_keys: self.data[key].append(None)
            
    def __repr__(self):
        return f"CustomDataProbe(name='{self.name}', keys={self.data_keys}, interval={self.record_interval}, records={len(self.time_data)})"

# 可以在 dynn.utils 中提供一些预定义的 data_provider_fn，
# 例如：
# def get_reward_mod_factor(network, current_time_ms, learning_rule_name_map):
#     # learning_rule_name_map = {"probe_key_name": ("synapse_collection_name", "learning_rule_attr_on_syncoll")}
#     # ... 实现 ...
#     pass 