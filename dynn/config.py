# Placeholder for system configuration management 

import yaml # PyYAML 需要安装: pip install PyYAML
import copy # 修正：正确的导入

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

    def __str__(self):
        # 以更可读的格式打印配置 (例如，YAML 格式)
        try:
            return yaml.dump(self._config, default_flow_style=False, allow_unicode=True, sort_keys=False)
        except Exception:
            return repr(self)

# 示例默认配置 (可以非常详细)
def get_default_dynn_config():
    """返回一个包含 DyNN 框架默认参数的字典。"""
    config = {
        'simulation': {
            'dt': 1.0,  # ms, 时间步长
            'total_duration': 1000.0, # ms, 总仿真时长 (示例)
            'random_seed': None # None 表示不设置特定种子，或使用整数值
        },
        'snn_core': {
            'populations': [], # 列表，每个元素是一个群体配置字典
            'synapses': [],    # 列表，每个元素是一个突触连接配置字典
        },
        'environment': {
            'name': 'MountainCar-v0', # 示例 Gym 环境
            'params': {},
        },
        'learning': {
            'reward_processor': {
                'type': 'SlidingWindowSmoother', # 或 None
                'params': {
                    'window_size': 100
                }
            },
            'global_reward_modulation': True # 是否所有学习规则都受同一个处理后的奖励信号调制
        },
        'experiment': {
            'name': 'default_experiment',
            'log_level': 'INFO',
            'results_dir': 'results/'
        }
    }
    return config

# 可以在这里预定义一些特定模型/实验的配置模板函数
# 例如: def get_mountain_car_config(): ...


if __name__ == '__main__':
    # 示例用法
    default_cfg = get_default_dynn_config()
    config_manager = ConfigManager(default_cfg)

    print("--- 初始配置 ---")
    print(config_manager)

    # 获取配置
    print(f"\n获取 dt: {config_manager.get('simulation.dt')}")
    print(f"获取不存在的路径 (带默认值): {config_manager.get('snn.nonexistent.param', default=42)}")

    # 设置配置
    config_manager.set('snn_core.populations', [
        {
            'name': 'input_pop', 'num_neurons': 10, 'model': 'IzhikevichNeuron',
            'params': {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 2},
            'initial_v': ('normal', -70, 5), # (distribution_type, mean, std) or scalar
            'initial_u': ('uniform', -14, -10) # (distribution_type, low, high) or scalar
        },
        {
            'name': 'output_pop', 'num_neurons': 3, 'model': 'IzhikevichNeuron',
            'params': {'a': 0.1, 'b': 0.25, 'c': -60, 'd': 2}
        }
    ])
    config_manager.set('snn_core.synapses', [
        {
            'name': 'input_to_output',
            'pre_pop_name': 'input_pop',
            'post_pop_name': 'output_pop',
            'connectivity': {
                'type': 'sparse_prob', # 'full', 'sparse_prob', 'sparse_num', 'neighborhood'
                'prob': 0.5 # 连接概率
            },
            'weight_init': {
                'dist': ('normal', 0.5, 0.1), # (type, mean, std) or scalar
                'w_min': 0.0,
                'w_max': 1.0
            },
            'is_excitatory': True,
            'learning_rule': {
                'type': 'TraceSTDP',
                'params': {
                    'lr_ltp': 0.005, 'lr_ltd': 0.005,
                    'tau_pre': 20.0, 'tau_post': 20.0,
                    # w_min, w_max 从 weight_init 获取或在此处覆盖
                }
            }
        }
    ])

    print("\n--- 修改后的配置 (部分) ---")
    print(f"输入群体配置: {config_manager.get('snn_core.populations.0')}")
    print(f"第一个突触的学习规则类型: {config_manager.get('snn_core.synapses.0.learning_rule.type')}")

    # 从字典更新
    updates = {
        'simulation': {'dt': 0.5, 'random_seed': 12345},
        'experiment': {'name': 'mountain_car_test'}
    }
    config_manager.update_from_dict(updates)
    print("\n--- 从字典更新后的配置 ---")
    print(f"dt: {config_manager.get('simulation.dt')}, seed: {config_manager.get('simulation.random_seed')}")
    print(f"实验名称: {config_manager.get('experiment.name')}")

    # 保存到 YAML
    config_manager.save_to_yaml('./dynn_config_example.yaml')
    print("\n配置已保存到 dynn_config_example.yaml")

    # 从 YAML 加载 (创建新实例或覆盖现有实例)
    new_config_manager = ConfigManager()
    new_config_manager.load_from_yaml('./dynn_config_example.yaml')
    print("\n--- 从 YAML 文件加载的配置 ---")
    # print(new_config_manager)
    assert new_config_manager.get('simulation.dt') == 0.5
    assert new_config_manager.get('snn_core.populations.0.name') == 'input_pop'
    print("YAML 加载和断言成功。")

    # 测试不存在的路径返回None
    assert new_config_manager.get('does.not.exist') is None
    print("测试不存在路径返回 None 成功。")

    # 测试设置深层路径
    config_manager.set("a.b.c.d.e", 100)
    assert config_manager.get("a.b.c.d.e") == 100
    print("测试设置深层路径成功。") 