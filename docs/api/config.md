# `dynn.config` API 文档

## 模块概览

`dynn.config` 模块负责管理 DyNN 框架的所有可配置参数。它提供了一个中心化的方式来加载、访问、修改和保存配置，支持从字典和 YAML 文件初始化配置，并通过点分路径轻松访问嵌套的配置项。

主要组件包括：

*   `ConfigManager`: 一个类，用于封装和操作配置数据。
*   `get_default_dynn_config()`: 一个函数，返回 DyNN 框架的默认配置字典。

## 类详细说明

### `ConfigManager`

#### 类名及其构造函数签名

`ConfigManager(default_config=None)`

#### 类描述

`ConfigManager` 类是 DyNN 框架中配置管理的核心。它允许从多种来源（如Python字典、YAML文件）加载配置参数，并提供了一种结构化的方式来访问和修改这些参数。配置项可以通过点分路径字符串进行寻址 (例如 `'snn_core.neurons.params.a'`)。

#### 构造函数参数

*   `default_config` (`dict`, 可选): 一个可选的Python字典，包含初始的默认配置参数。如果提供，该字典会被深拷贝到配置管理器中。

#### 主要方法

*   `get(self, path, default=None)`:
    *   使用点分路径获取配置项的值。
    *   **参数**:
        *   `path` (`str`): 配置项的点分路径 (例如 `'simulation.dt'`)。
        *   `default` (`any`, 可选): 如果指定的路径在配置中不存在，则返回此默认值。
    *   **返回值**: `any` - 找到的配置项的值；如果未找到，则为指定的 `default` 值。

*   `set(self, path, value)`:
    *   使用点分路径设置配置项的值。如果路径中的中间层级（字典）不存在，则会自动创建它们。
    *   **参数**:
        *   `path` (`str`): 要设置的配置项的点分路径。
        *   `value` (`any`): 要为该配置项设置的新值。

*   `update_from_dict(self, update_dict, merge_nested=True)`:
    *   从一个Python字典更新当前配置。
    *   **参数**:
        *   `update_dict` (`dict`): 包含要更新或添加的参数的字典。
        *   `merge_nested` (`bool`, 可选, 默认值: `True`): 如果为 `True`，则递归地合并嵌套字典。如果 `update_dict` 中的键也存在于当前配置中且两者都是字典，则它们的内容会被合并。如果为 `False`，则对于顶层键，`update_dict` 中的值会直接替换当前配置中的值。

*   `load_from_yaml(self, filepath, merge_nested=True)`:
    *   从指定的 YAML 文件加载配置，并将其与当前配置合并。
    *   **参数**:
        *   `filepath` (`str`): YAML 配置文件的路径。
        *   `merge_nested` (`bool`, 可选, 默认值: `True`): 合并行为同 `update_from_dict`。
    *   **注意**: 此方法需要 `PyYAML` 库 (`pip install PyYAML`)。如果文件不存在或解析失败，会打印错误信息。

*   `save_to_yaml(self, filepath)`:
    *   将当前的完整配置保存到指定的 YAML 文件。
    *   **参数**:
        *   `filepath` (`str`): 要保存配置的 YAML 文件的路径。

*   `get_all_config(self)`:
    *   返回当前整个配置字典的一个深拷贝。
    *   **返回值**: `dict` - 当前配置的完整深拷贝。

*   `__repr__(self)`:
    *   返回 `ConfigManager` 实例的简洁字符串表示，通常显示顶级配置键。

*   `__str__(self)`:
    *   返回当前配置的更可读的字符串表示，通常为 YAML 格式。

## 函数详细说明

### `get_default_dynn_config()`

#### 函数签名

`get_default_dynn_config()`

#### 函数描述

此函数返回一个Python字典，其中包含了 DyNN 框架的一组预定义默认配置参数。这些参数涵盖了仿真的基本设置、SNN核心组件的结构、环境信息、学习参数以及实验管理等方面。这个默认配置可以作为构建具体实验配置的基础。

#### 返回值

*   `dict`: 一个包含 DyNN 默认配置的字典。

#### 默认配置结构概览 (示例)

```python
{
    'simulation': {
        'dt': 1.0,
        'total_duration': 1000.0,
        'random_seed': None
    },
    'snn_core': {
        'populations': [], # 列表，每个元素是群体配置
        'synapses': []     # 列表，每个元素是突触连接配置
    },
    'environment': {
        'name': 'MountainCar-v0',
        'params': {}
    },
    'learning': {
        'reward_processor': {
            'type': 'SlidingWindowSmoother',
            'params': {'window_size': 100}
        },
        'global_reward_modulation': True
    },
    'experiment': {
        'name': 'default_experiment',
        'log_level': 'INFO',
        'results_dir': 'results/'
    }
}
```

## (可选) 使用示例

```python
from dynn.config import ConfigManager, get_default_dynn_config

# 1. 获取默认配置并创建 ConfigManager 实例
default_config = get_default_dynn_config()
config = ConfigManager(default_config)

# 2. 获取配置项
simulation_dt = config.get('simulation.dt')
print(f"Simulation dt: {simulation_dt}")

# 获取不存在的项，使用默认值
learning_rate = config.get('snn_core.learning_rules.stdp.lr', default=0.001)
print(f"STDP Learning rate (defaulted): {learning_rate}")

# 3. 设置配置项
config.set('simulation.total_duration', 2000.0)
config.set('snn_core.populations', [
    {'name': 'input_pop', 'num_neurons': 50, 'model': 'IzhikevichNeuron'},
    {'name': 'output_pop', 'num_neurons': 10, 'model': 'IzhikevichNeuron'}
])
print(f"Updated total_duration: {config.get('simulation.total_duration')}")

# 4. 从字典更新配置
updates = {
    'simulation': {'random_seed': 12345},
    'experiment': {'name': 'custom_run_01'}
}
config.update_from_dict(updates)
print(f"Updated random_seed: {config.get('simulation.random_seed')}")
print(f"Experiment name: {config.get('experiment.name')}")

# 5. 保存当前配置到 YAML 文件 (假设有一个 'my_config.yaml')
# config.save_to_yaml('my_config.yaml')
# print("Configuration saved to my_config.yaml")

# 6. 从 YAML 文件加载配置 (假设有一个 'existing_config.yaml')
# config.load_from_yaml('existing_config.yaml')
# print("Configuration loaded from existing_config.yaml")

# 7. 获取所有配置
all_params = config.get_all_config()
# print("
Full configuration:")
# print(config) # 使用 __str__ 打印 YAML 格式
``` 