# Placeholder for system configuration management 

import yaml # PyYAML 需要安装: pip install PyYAML
import copy # 修正：正确的导入
from types import SimpleNamespace # 新增导入

class ConfigManager:
    """
    管理 DyNN 框架的所有可配置参数。
    支持从字典、YAML文件加载配置，并允许参数覆盖。
    提供点分路径访问配置项。
    """
    def __init__(self, default_config=None):
        """
        初始化配置管理器。

        参数:
            default_config (dict, optional): 一个包含默认配置参数的字典。
        """
        self._config = copy.deepcopy(default_config) if default_config else {} # 修正：使用正确的copy.deepcopy

    def get(self, path, default=None):
        """
        使用点分路径获取配置项的值。
        例如: get('snn.neurons.izhikevich.a')

        参数:
            path (str): 配置项的点分路径。
            default (any, optional): 如果路径不存在，则返回此默认值。

        返回:
            any: 配置项的值，如果不存在则返回默认值。
        """
        keys = path.split('.')
        current_level = self._config
        try:
            for key in keys:
                if isinstance(current_level, dict):
                    current_level = current_level[key]
                # elif isinstance(current_level, list) and key.isdigit(): # 可选支持列表索引
                #    current_level = current_level[int(key)]
                else:
                    # print(f"警告: 路径 '{path}' 在键 '{key}' 处无效 (当前层类型: {type(current_level)}) ")
                    return default # 路径无效
            return current_level
        except (KeyError, IndexError, TypeError):
            # print(f"信息: 路径 '{path}' 未找到，返回默认值。")
            return default

    def set(self, path, value):
        """
        使用点分路径设置配置项的值。
        如果路径中的中间字典不存在，则会创建它们。
        例如: set('snn.neurons.izhikevich.a', 0.02)

        参数:
            path (str): 配置项的点分路径。
            value (any): 要设置的值。
        """
        keys = path.split('.')
        current_level = self._config
        for i, key in enumerate(keys):
            if i == len(keys) - 1: # 到达最后一个键
                current_level[key] = value
            else:
                if key not in current_level or not isinstance(current_level[key], dict):
                    current_level[key] = {} # 如果不存在或不是字典，则创建新字典
                current_level = current_level[key]

    def update_from_dict(self, update_dict, merge_nested=True):
        """
        从另一个字典更新配置。

        参数:
            update_dict (dict): 包含要更新的参数的字典。
            merge_nested (bool): 如果为True (默认)，则递归合并嵌套字典。
                                如果为False，则顶层键的值将被直接替换。
        """
        if merge_nested:
            self._merge_dicts(self._config, update_dict)
        else:
            self._config.update(update_dict)

    def _merge_dicts(self, target, source):
        """ 辅助函数，递归合并源字典到目标字典。"""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._merge_dicts(target[key], value)
            else:
                target[key] = value

    def load_from_yaml(self, filepath, merge_nested=True):
        """
        从 YAML 文件加载配置并更新当前配置。

        参数:
            filepath (str): YAML 配置文件的路径。
            merge_nested (bool): 是否递归合并，同 update_from_dict。
        
        注意: 需要安装 PyYAML (pip install PyYAML)。
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            if yaml_config and isinstance(yaml_config, dict):
                self.update_from_dict(yaml_config, merge_nested=merge_nested)
                # print(f"配置已从 {filepath} 加载并合并。")
            else:
                print(f"警告: YAML 文件 {filepath} 为空或格式不正确。未加载配置。")
        except FileNotFoundError:
            print(f"错误: 找不到配置文件 {filepath}。")
        except yaml.YAMLError as e:
            print(f"错误: 解析 YAML 文件 {filepath} 失败: {e}")
        except Exception as e:
            print(f"错误: 从 {filepath} 加载配置时发生意外错误: {e}")

    def save_to_yaml(self, filepath):
        """
        将当前配置保存到 YAML 文件。

        参数:
            filepath (str): 要保存 YAML 文件的路径。
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            # print(f"当前配置已保存到 {filepath}")
        except Exception as e:
            print(f"错误: 保存配置到 {filepath} 失败: {e}")

    def get_all_config(self):
        """
        返回整个配置字典的深拷贝。
        """
        return copy.deepcopy(self._config) # 修正：使用正确的copy.deepcopy

    def __repr__(self):
        # 为了简洁，不打印整个配置字典
        return f"ConfigManager(config_keys_L1={list(self._config.keys())})"

    def to_namespace(self):
        """将配置字典转换为一个嵌套的 SimpleNamespace 对象，以便通过属性访问。"""
        return self._dict_to_namespace(self._config)

    def _dict_to_namespace(self, d):
        if isinstance(d, dict):
            # 首先递归转换所有子字典
            converted_dict = {k: self._dict_to_namespace(v) for k, v in d.items()}
            return SimpleNamespace(**converted_dict)
        elif isinstance(d, list):
            return [self._dict_to_namespace(item) for item in d]
        return d

# 示例默认配置 (可以非常详细)
def get_default_dynn_config():
    """返回一个包含 DyNN 框架默认参数的字典。"""
    config = {
        'simulation': {
            'dt': 1.0,  # ms, 时间步长
            'snn_run_duration_ms_per_env_step': 20.0, # ms, 每个环境步骤SNN运行的时长
            'num_episodes': 100, # 增加轮次数以进行更长时间的训练
            'max_steps_per_episode': 200, # MountainCar-v0 默认是200
            'random_seed': None
        },
        'environment': {
            'name': 'MountainCar-v0',
            'params': {},
        },
        'snn': { # 将SNN核心相关的配置移到这里，与实验脚本的 config.snn.xxx 对应
            'input_neurons_count': 64, # 示例值
            'output_neurons_count': 3, # 对应 MountainCar 的3个动作
            'populations': [
                {
                    'name': 'InputPopulation',
                    'model_type': 'IzhikevichNeuron', # 新增：指定神经元模型类名
                    'params': {'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0}, # RS神经元参数
                    'initial_conditions': { # 新增：结构化初始条件
                        'v': {'dist': 'uniform', 'low': -70.0, 'high': -50.0},
                        'u': {'dist': 'scalar', 'value': -14.0} # d*v_reset_approx
                    }
                },
                {
                    'name': 'OutputPopulation',
                    'model_type': 'IzhikevichNeuron',
                    'params': {'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0},
                    'initial_conditions': {
                        'v': {'dist': 'uniform', 'low': -70.0, 'high': -50.0},
                        'u': {'dist': 'scalar', 'value': -14.0}
                    }
                }
            ],
            'synapses': [
                {
                    'name': 'InputToOutputConnections',
                    'pre_population_name': 'InputPopulation',
                    'post_population_name': 'OutputPopulation',
                    'connectivity_type': 'full', # 'full', 'sparse_prob', etc.
                    # 'connectivity_params': {'prob': 0.1}, # 如果是稀疏连接
                    'initial_weights': {
                        'strategy': 'normal', # 'normal', 'uniform', scalar
                        'mean': 0.5, # for normal
                        'std': 0.1,   # for normal
                        # 'low': 0.0, 'high': 1.0, # for uniform
                        'value': 1.0 # for scalar
                    },
                    'weight_limits': {'min': 0.0, 'max': 10.0} # 示例权重限制
                }
            ]
        },
        'input_encoder': {
            'type': 'GaussianEncoder', # 'GaussianEncoder' or 'CurrentInjector'
            'gaussian_sigma_scale': 0.1, # 重命名自 gaussian_sigma，现在代表 sigma_scale
            'gaussian_amplitude': 10.0, # 注入电流的幅度
            'current_injector_gain': 1.0
        },
        'output_decoder': {
            'type': 'WinnerTakesAllDecoder',
            # SpikeRateDecoder 参数 (如果使用)
            # 'spike_rate_window_ms': 50.0
        },
        'learning_rules': {
            'stdp': { # STDP 基础参数
                'tau_plus': 20.0,  # ms
                'tau_minus': 20.0, # ms
                'a_plus': 0.01,   # 学习率 (LTP)
                'a_minus': 0.01,  # 学习率 (LTD)
                'dependency_type': 'multiplicative' # 'additive' or 'multiplicative'
            },
            'reward_modulation': { # 奖励调制参数
                'reward_tau': 50.0, # ms, 奖励信号平滑的时间常数 (用于RewardModulatedSTDP内部平滑)
                                     # 注意: 这与外部 SlidingWindowSmoother 的 window_size 不同
                'strength': 0.1 # 奖励对学习率的调制强度
            }
        },
        'reward_processor': {
            'type': 'SlidingWindowSmoother',
            'smoothing_window_size': 50 # 多少个环境步骤的奖励进行平滑
        },
        'probes': {
            'record_interval_ms': 100.0, # ms
            'output_dir': 'results/mountain_car_v0/',
            'save_to_csv': True
        },
        'experiment': {
            'name': 'mountain_car_v0_default',
            'log_level': 'INFO',
        }
    }
    return config

# 全局默认配置实例 (ConfigManager)
# default_config_manager = ConfigManager(get_default_dynn_config()) # 不在这里实例化，而是在load_config中

def load_config(default_config_dict=None, yaml_config_path=None):
    """
    加载配置。首先加载默认配置字典，然后（如果提供）从YAML文件加载并覆盖。
    返回一个可以通过属性访问的配置对象 (SimpleNamespace)。

    参数:
        default_config_dict (dict, optional): 包含默认配置的字典。
                                            如果为None，则使用 get_default_dynn_config()。
        yaml_config_path (str, optional): YAML配置文件的路径。

    返回:
        SimpleNamespace: 一个嵌套的 SimpleNamespace 对象，包含最终的配置。
    """
    if default_config_dict is None:
        default_config_dict = get_default_dynn_config()

    manager = ConfigManager(default_config_dict)

    if yaml_config_path:
        print(f"尝试从 YAML 文件加载配置: {yaml_config_path}")
        manager.load_from_yaml(yaml_config_path)
    else:
        print("未提供 YAML 配置文件路径，使用默认配置。")

    return manager.to_namespace()


if __name__ == '__main__':
    # 示例用法
    # 1. 加载默认配置
    cfg = load_config()
    print("--- 默认配置 (通过属性访问) ---")
    print(f"模拟步长 (dt): {cfg.simulation.dt} ms")
    print(f"输入神经元数量: {cfg.snn.input_neurons_count}")
    print(f"第一个突触连接的学习规则 (如果存在): {cfg.snn.synapses[0].learning_rule if hasattr(cfg.snn.synapses[0], 'learning_rule') else 'N/A'}")
    print(f"STDP a_plus: {cfg.learning_rules.stdp.a_plus}")

    # 2. 模拟从文件加载配置 (创建一个临时的YAML文件来测试)
    temp_yaml_content = """
simulation:
  dt: 0.5 # 覆盖默认的 dt
  num_episodes: 5

snn:
  input_neurons_count: 32 # 覆盖

experiment:
  name: 'my_custom_experiment'
"""
    temp_yaml_path = 'temp_test_config.yaml'
    with open(temp_yaml_path, 'w', encoding='utf-8') as f:
        f.write(temp_yaml_content)

    cfg_from_file = load_config(yaml_config_path=temp_yaml_path)
    print("\n--- 从 YAML 加载并覆盖后的配置 ---")
    print(f"模拟步长 (dt): {cfg_from_file.simulation.dt} ms")
    print(f"输入神经元数量: {cfg_from_file.snn.input_neurons_count}")
    print(f"实验名称: {cfg_from_file.experiment.name}")
    print(f"STDP a_plus (未被覆盖，应为默认值): {cfg_from_file.learning_rules.stdp.a_plus}")

    # 清理临时文件
    import os
    try:
        os.remove(temp_yaml_path)
    except OSError:
        pass

    # 示例：直接使用 ConfigManager 来获取特定值 (如果不想用 SimpleNamespace)
    manager = ConfigManager(get_default_dynn_config())
    if os.path.exists(temp_yaml_path):
        manager.load_from_yaml(temp_yaml_path)
    else:
        print(f"信息: 临时配置文件 {temp_yaml_path} 已被删除或不存在，跳过加载到 manager。")

    print("\n--- 使用 ConfigManager.get --- ")
    print(f"模拟步长 (dt): {manager.get('simulation.dt')}")

    # 展示保存配置
    # default_config_for_saving = ConfigManager(get_default_dynn_config())
    # default_config_for_saving.save_to_yaml("default_dynn_config_output.yaml")
    # print("\n默认配置已保存到 default_dynn_config_output.yaml")

    # 测试不存在的路径返回None (对于SimpleNamespace，我们会期望AttributeError)
    try:
        _ = cfg_from_file.this_attribute_should_not_exist
        # 如果上面没有抛出 AttributeError，说明它意外地存在了
        # 或者 SimpleNamespace 的行为与预期不同 (例如，如果它返回 None 而不是抛出错误)
        # 但标准的 SimpleNamespace 会在属性不存在时抛出 AttributeError
        # 如果我们想测试 ConfigManager.get() 的行为，那应该用 manager 对象
        assert manager.get('this.deep.path.does.not.exist') is None, \
               "ConfigManager.get() 未对不存在的深层路径返回 None"
        print("测试 ConfigManager.get() 对不存在路径返回 None 成功。")

        # 对于SimpleNamespace，我们可以检查一个顶层属性是否不存在
        # (注意：SimpleNamespace(**{}) 创建后，访问任何属性都会 AttributeError)
        # 如果 'this_attribute_should_not_exist' 不在 cfg_from_file 的顶层，访问会报错
        # 这个测试点有点微妙，取决于 `_dict_to_namespace` 如何处理空字典或不存在的键
        # 假设 `load_config` 返回的 namespace 中，未定义的顶层属性不会存在
        # 更好的测试是尝试访问并期望 AttributeError
        raised_error = False
        try:
            _ = cfg_from_file.non_existent_top_level_attribute 
        except AttributeError:
            raised_error = True
        assert raised_error, "访问 SimpleNamespace 上不存在的顶层属性时未引发 AttributeError"
        print("测试 SimpleNamespace 对不存在的顶层属性引发 AttributeError 成功。")

    except AttributeError:
        # 这是访问SimpleNamespace上不存在属性时的预期行为
        print("测试 SimpleNamespace 对不存在属性引发 AttributeError 成功 (通过捕获异常)。")
        # 额外确认 ConfigManager.get() 的行为
        assert manager.get('this.deep.path.does.not.exist') is None, \
               "ConfigManager.get() 未对不存在的深层路径返回 None (在异常块中检查)"
        print("测试 ConfigManager.get() 对不存在路径返回 None 成功 (在异常块中检查)。")

    # 测试设置深层路径 (这部分是针对 ConfigManager 的)
    temp_manager_for_set = ConfigManager({})
    temp_manager_for_set.set("a.b.c.d.e", 100)
    assert temp_manager_for_set.get("a.b.c.d.e") == 100
    print("测试设置深层路径成功。") 