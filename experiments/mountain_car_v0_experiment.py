'''
## 1. 项目概述与目标

本项目旨在从零开始设计并实现一个名为 DyNN (Dynamic Neural Networks) 的脉冲神经网络 (SNN) 仿真框架。
该框架的首要应用目标是支持研究和开发基于SNN的强化学习智能体，具体将以控制 OpenAI Gym 中的 "MountainCar-v0" 环境作为初始验证场景。

核心学习范式将围绕尖峰时间依赖可塑性 (STDP) 构建，并要求该学习机制能够根据外部奖励信号进行有效的动态调制。
框架设计需强调模块化、灵活性和可配置性，以适应未来对不同神经元模型、学习规则、网络拓扑及应用场景的探索与扩展。
'''
# Placeholder for MountainCar-v0 experiment script 

import gymnasium as gym
import numpy as np
import time
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm # 导入 tqdm

# 添加 Matplotlib 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

from dynn.core.neurons import NeuronPopulation, IzhikevichNeuron
from dynn.core.synapses import SynapseCollection
from dynn.core.network import NeuralNetwork
from dynn.core.simulator import Simulator
from dynn.core.learning_rules import STDP, RewardModulatedSTDP
from dynn.io.input_encoders import GaussianEncoder, DirectCurrentInjector
from dynn.io.output_decoders import InstantaneousSpikeCountDecoder, MountainCarActionDecoder
from dynn.io.reward_processors import SlidingWindowSmoother
from dynn.utils.probes import PopulationProbe, SynapseProbe
from dynn.config import load_config # Updated import

def run_mountain_car_experiment(config_path=None):
    """
    运行 MountainCar-v0 实验的主函数。
    """
    # 1. 加载配置
    config = load_config(yaml_config_path=config_path) # Use the new load_config function
    print("使用配置:")
    # A more structured way to print the config, e.g. a few key items
    print(f"  Simulation dt: {config.simulation.dt} ms")
    print(f"  SNN Input Neurons: {config.snn.input_neurons_count}")
    print(f"  SNN Output Neurons: {config.snn.output_neurons_count}")
    print(f"  Environment: {config.environment.name}")

    # 设置随机种子 (如果提供)
    if config.simulation.random_seed is not None:
        np.random.seed(config.simulation.random_seed)
        # Note: gym environments might also need seeding if deterministic behavior is critical
        # env.seed(config.simulation.random_seed) # or env.reset(seed=...) for newer gym
        print(f"设置随机种子: {config.simulation.random_seed}")


    # 2. 初始化 Gym 环境
    env = gym.make(config.environment.name)
    # For newer gym versions, reset() returns (observation, info)
    # and seeding is done via env.reset(seed=config.simulation.random_seed)
    # For compatibility, let's assume older gym or handle it:
    try:
        observation, info = env.reset(seed=config.simulation.random_seed if config.simulation.random_seed is not None else None)
    except TypeError: # Older gym might not support seed in reset or returns only obs
        observation = env.reset()
        if config.simulation.random_seed is not None:
             env.seed(config.simulation.random_seed) # Deprecated
        info = {} # Placeholder for info

    print(f"初始观测: 位置={observation[0]}, 速度={observation[1]}")

    # 3. 配置输入编码器
    num_input_neurons = config.snn.input_neurons_count
    position_range = (env.observation_space.low[0], env.observation_space.high[0])
    
    # 确保目标群体名称在配置中可用或硬编码，这里假设为 'InputPopulation'
    # 这与后面创建 NeuronPopulation 时使用的名称一致。
    input_population_name = "InputPopulation" 

    if config.input_encoder.type == 'GaussianEncoder':
        # 确保从配置中获取的参数名称与 GaussianEncoder 构造函数匹配
        # GaussianEncoder(target_pop_name, num_neurons, min_val, max_val, sigma_scale=0.1, current_amplitude=10.0)
        input_encoder = GaussianEncoder(
            target_pop_name=input_population_name,
            num_neurons=num_input_neurons,
            min_val=position_range[0],
            max_val=position_range[1],
            sigma_scale=config.input_encoder.gaussian_sigma_scale, # 假设配置中有 gaussian_sigma_scale
            current_amplitude=config.input_encoder.gaussian_amplitude
        )
    elif config.input_encoder.type == 'CurrentInjector':
        # DirectCurrentInjector(target_pop_name, num_neurons, observation_slice=None, scale_factor=1.0)
        # MountainCar 观察值是 [position, velocity]。我们只用 position。
        # 因此，observation_slice 应该是 slice(0, 1) 如果输入是向量，但这里 encode() 只接收位置
        # 如果 encode() 只接收位置标量，那么 DirectCurrentInjector 需要 num_neurons=1 (除非它内部处理扩展)
        # 根据 DirectCurrentInjector 的实现，它期望输入向量的长度与 num_neurons 匹配。
        # 而 MountainCar 的实验脚本后面是这样调用: input_currents = input_encoder.encode(position)
        # 这意味着 DirectCurrentInjector 的 encode 会收到一个标量。这与其设计不符。
        # 因此，如果使用 CurrentInjector (DirectCurrentInjector)，它应该适用于标量输入并扩展到 num_neurons，
        # 或者 MountainCar 的 config 应该指定 num_input_neurons=1 for this type, and use gain.

        # 假设：如果类型是 CurrentInjector，配置文件意图是将单个位置值通过增益应用到所有输入神经元，
        # 或者配置文件中的 current_injector_gain 意味着每个神经元接收此固定电流（如果位置满足条件）。
        # DirectCurrentInjector 的当前实现是将输入向量直接映射。
        # 为了使代码能运行，我们需要一个能处理标量输入并应用到所有神经元的 Injector，
        # 或者假设 config.snn.input_neurons_count 为1当使用此类编码器时。

        # *** 暂时保留实验脚本原来的意图，并假设一个简化的 CurrentInjector 行为 ***
        # *** 这可能需要 input_encoders.py 中的 DirectCurrentInjector 被修改或被一个新的类替代 ***
        # *** 为了让脚本继续运行，我们先按原实验脚本的参数名（如 gain）进行映射 ***
        # *** 并假设 config.input_encoder.current_injector_gain 将作为 scale_factor ***
        # *** 并且输入给 encode 的将是单个值，然后该值乘以 gain 广播到所有神经元。***
        # *** DirectCurrentInjector 本身不支持这种广播，它期望输入向量维度匹配 num_neurons ***
        
        # 这是一个临时的妥协，需要回顾编码器设计与实验脚本的实际用法是否匹配。
        # 如果 config.snn.input_neurons_count > 1，而我们传递标量给 DirectCurrentInjector，它会报错。
        # 我们先假设配置文件会确保 num_input_neurons=1，如果用的是 'CurrentInjector' 类型，
        # 并且 gain 就是 scale_factor。
        print(f"警告: 使用 '{config.input_encoder.type}' 时，假设输入神经元数量 ({num_input_neurons}) 与编码器期望的输入维度匹配。")
        print(f"警告: DirectCurrentInjector 通常期望一个与神经元数量等长的输入向量。如果 num_input_neurons > 1，这可能会在 encode 时失败。")

        input_encoder = DirectCurrentInjector( # Changed class name
            target_pop_name=input_population_name,
            num_neurons=num_input_neurons, # 如果 > 1，encode(scalar) 会失败
            # observation_slice=None, # 对于标量输入，slice不适用
            scale_factor=config.input_encoder.current_injector_gain # 映射 gain 到 scale_factor
        )
    else:
        raise ValueError(f"未知的输入编码器类型: {config.input_encoder.type}")


    # 4. 配置输出解码器
    num_output_neurons = config.snn.output_neurons_count
    output_population_name = "OutputPopulation" # Consistent with neuron population creation

    output_decoder = None
    if config.output_decoder.type == 'WinnerTakesAllDecoder': 
        print("信息: 使用 WinnerTakesAllDecoder (实际上是 InstantaneousSpikeCountDecoder 的一个配置)。")
        # 此分支的参数处理可能与 InstantaneousSpikeCountDecoder 不同，但目前不使用默认配置
        output_decoder = InstantaneousSpikeCountDecoder(
            name="OutputDecoder", 
            source_pop_name=output_population_name,
            num_actions=env.action_space.n,
            default_action=getattr(config.output_decoder, 'default_action_if_none', 1) 
        )
    elif config.output_decoder.type == 'MountainCarActionDecoder':
        print("信息: 使用 MountainCarActionDecoder。")
        output_decoder = MountainCarActionDecoder(
             source_pop_name=output_population_name, 
             default_action_idx=getattr(config.output_decoder, 'default_action', 1)
        )
    elif config.output_decoder.type == 'InstantaneousSpikeCountDecoder': # 默认配置会进入此分支
        print(f"信息: 使用 InstantaneousSpikeCountDecoder (类型: {config.output_decoder.type})。")
        
        cfg_default_action_val = getattr(config.output_decoder, 'default_action', None)
        print(f"  从配置加载的 default_action: {cfg_default_action_val} (类型: {type(cfg_default_action_val)})")

        effective_default_action = 1 # 默认设为1 (no-op for MountainCar)
        if cfg_default_action_val is not None:
            try:
                effective_default_action = int(cfg_default_action_val)
            except (ValueError, TypeError):
                print(f"  警告: 配置的 default_action '{cfg_default_action_val}' 无法转换为整数。使用硬编码默认值 1。")
                effective_default_action = 1
        else:
            # 如果配置中 default_action 是 None 或未设置，也使用硬编码的 1
            print(f"  信息: 配置的 default_action 未设置或为 None。使用硬编码默认值 1。")
            effective_default_action = 1 # 确保有一个值
            
        print(f"  将用于解码器的有效 default_action: {effective_default_action}")

        # 确保 num_actions 也从配置中获取，如果存在，否则从环境中获取
        num_actions_for_decoder = getattr(config.output_decoder, 'num_actions', None)
        if num_actions_for_decoder is None:
            print(f"  信息: 配置中未找到 num_actions，将使用来自环境的值: {env.action_space.n}")
            num_actions_for_decoder = env.action_space.n
        else:
            try:
                num_actions_for_decoder = int(num_actions_for_decoder)
                print(f"  信息: 从配置中使用 num_actions: {num_actions_for_decoder}")
            except (ValueError, TypeError):
                print(f"  警告: 配置的 num_actions '{num_actions_for_decoder}' 非整数。将使用来自环境的值: {env.action_space.n}")
                num_actions_for_decoder = env.action_space.n

        output_decoder = InstantaneousSpikeCountDecoder(
            source_pop_name=output_population_name,
            num_actions=num_actions_for_decoder, 
            default_action=effective_default_action
        )
    else:
        raise ValueError(f"未知的输出解码器类型: {config.output_decoder.type}")

    if output_decoder is None:
        raise ValueError("未知的输出解码器类型: None")


    # 5. 构建 SNN 网络
    network = NeuralNetwork(name="MountainCarSNN") # 移除dt，可以给网络一个名字

    # Helper function to generate initial condition values or lambdas from config
    def _get_initial_condition_config_obj(condition_config_from_yaml):
        """直接返回从YAML解析的条件配置对象 (SimpleNamespace) 或 None。"""
        # condition_config_from_yaml 是类似 config.snn.populations[0].initial_conditions.v 的对象
        if not condition_config_from_yaml: 
            return None
        # 验证它至少有 'dist' 属性，如果不是None
        if not hasattr(condition_config_from_yaml, 'dist'):
            raise ValueError(f"初始条件配置对象缺少 'dist' 属性: {condition_config_from_yaml}")
        return condition_config_from_yaml

    # 5.1 创建神经元群体
    # Helper to find population config by name
    def get_pop_config(name):
        for pop_cfg in config.snn.populations:
            if pop_cfg.name == name:
                return pop_cfg
        raise ValueError(f"在配置中未找到名为 '{name}' 的神经元群体")

    input_pop_config = get_pop_config('InputPopulation')
    
    # Dynamically select neuron model class based on config (example)
    neuron_model_cls = None # Renamed variable to avoid conflict with parameter name
    if input_pop_config.model_type == 'IzhikevichNeuron':
        neuron_model_cls = IzhikevichNeuron
    else:
        raise ValueError(f"未知的神经元模型类型: {input_pop_config.model_type}")

    # Convert SimpleNamespace params to dict before passing to NeuronPopulation
    input_neuron_params_dict = vars(input_pop_config.params) if input_pop_config.params else {}

    input_pop = NeuronPopulation(
        neuron_model_class=neuron_model_cls, 
        num_neurons=num_input_neurons,
        neuron_params=input_neuron_params_dict, 
        initial_v_dist=_get_initial_condition_config_obj(getattr(input_pop_config.initial_conditions, 'v', None)), 
        initial_u_dist=_get_initial_condition_config_obj(getattr(input_pop_config.initial_conditions, 'u', None)), 
        name=input_pop_config.name
    )
    network.add_population(input_pop)

    output_pop_config = get_pop_config('OutputPopulation')
    output_neuron_model_cls = None 
    if output_pop_config.model_type == 'IzhikevichNeuron':
        output_neuron_model_cls = IzhikevichNeuron 
    else:
        raise ValueError(f"未知的神经元模型类型: {output_pop_config.model_type}")

    # Convert SimpleNamespace params to dict for output population as well
    output_neuron_params_dict = vars(output_pop_config.params) if output_pop_config.params else {}

    output_pop = NeuronPopulation(
        neuron_model_class=output_neuron_model_cls, 
        num_neurons=num_output_neurons,
        neuron_params=output_neuron_params_dict, 
        initial_v_dist=_get_initial_condition_config_obj(getattr(output_pop_config.initial_conditions, 'v', None)), 
        initial_u_dist=_get_initial_condition_config_obj(getattr(output_pop_config.initial_conditions, 'u', None)), 
        name=output_pop_config.name
    )
    network.add_population(output_pop)

    # 5.2 创建突触连接
    # Helper to find synapse config by name
    def get_syn_config(name):
        for syn_cfg in config.snn.synapses:
            if syn_cfg.name == name:
                return syn_cfg
        raise ValueError(f"在配置中未找到名为 '{name}' 的突触连接")

    syn_config = get_syn_config('InputToOutputConnections')

    input_to_output_synapses = SynapseCollection(
        pre_population=input_pop,
        post_population=output_pop,
        name=syn_config.name
        # initial_weights 参数已移除，将在下面通过 initialize_weights 方法设置
    )

    # 从配置准备 initialize_weights 所需的参数
    init_weights_cfg = syn_config.initial_weights
    dist_config_for_synapses = None
    if init_weights_cfg.strategy == 'normal':
        dist_config_for_synapses = ('normal', (init_weights_cfg.mean, init_weights_cfg.std))
    elif init_weights_cfg.strategy == 'uniform':
        dist_config_for_synapses = ('uniform', (init_weights_cfg.low, init_weights_cfg.high))
    elif init_weights_cfg.strategy == 'scalar':
        dist_config_for_synapses = float(init_weights_cfg.value) # 直接传递标量值
    else:
        raise ValueError(f"未知的权重初始化策略: {init_weights_cfg.strategy}")

    # 获取连接类型和参数 (如果存在)
    connectivity_type = getattr(syn_config, 'connectivity_type', 'full') # 默认为 'full'
    connectivity_params = getattr(syn_config, 'connectivity_params', {}) # 默认为空字典
    # 将 connectivity_params 中的 SimpleNamespace 转换为字典 (如果它是 SimpleNamespace)
    if hasattr(connectivity_params, '__dict__'): # 检查是否像 SimpleNamespace
        connectivity_params = vars(connectivity_params)
    
    # 调用 initialize_weights 方法
    input_to_output_synapses.initialize_weights(
        dist_config=dist_config_for_synapses,
        connectivity_type=connectivity_type,
        **connectivity_params # 解包连接参数
    )

    # 应用权重限制 (如果已在突触配置中指定并且 SynapseCollection 支持)
    weight_limits_config = getattr(syn_config, 'weight_limits', None)

    if weight_limits_config is not None and hasattr(weight_limits_config, 'min') and hasattr(weight_limits_config, 'max'):
        # 配置中指定了权重限制，并且其结构似乎正确。
        if hasattr(input_to_output_synapses, 'set_weight_limits'):
            print(f"信息: 为突触 '{syn_config.name}' 应用配置的权重限制: "
                  f"最小={weight_limits_config.min}, 最大={weight_limits_config.max}")
            input_to_output_synapses.set_weight_limits(
                weight_limits_config.min,
                weight_limits_config.max
            )
            # 注意: 关于 SynapseCollection 内部处理或其构造函数参数是否应包含权重限制的原始 TODO，
            # 仍然是 DyNN 库设计者需要考虑的问题。
        else:
            print(f"警告: 突触 '{syn_config.name}' 的配置中指定了权重限制 (最小={weight_limits_config.min}, 最大={weight_limits_config.max}), "
                  "但是 SynapseCollection 类没有 'set_weight_limits' 方法。这些限制将不会被应用。")
    elif hasattr(input_to_output_synapses, 'set_weight_limits'):
        # SynapseCollection 支持限制方法，但在该突触的配置中未指定或未正确指定权重限制。
        print(f"信息: SynapseCollection 支持通过 'set_weight_limits' 方法设置权重限制, "
              f"但在突触 '{syn_config.name}' 的配置中未指定或未正确指定权重限制。 "
              "将使用 SynapseCollection 的默认限制行为 (如有)。")
    # 如果 weight_limits_config 为 None (或格式不正确) 并且 SynapseCollection 没有 set_weight_limits 方法,
    # 则不执行任何与权重限制相关的操作，并且不会针对此组合打印消息。

    network.add_synapses(input_to_output_synapses)

    # 5.3 配置学习规则
    stdp_params = config.learning_rules.stdp
    reward_modulation_params = config.learning_rules.reward_modulation

    learning_rule = RewardModulatedSTDP(
        synapse_collection=input_to_output_synapses, # Learning rule is associated with a synapse collection
        dt=config.simulation.dt, 
        tau_plus=stdp_params.tau_plus,
        tau_minus=stdp_params.tau_minus,
        a_plus=stdp_params.a_plus,
        a_minus=stdp_params.a_minus,
        dependency_type=stdp_params.dependency_type, 
        reward_tau=reward_modulation_params.reward_tau,
        learning_rate_modulation_strength=reward_modulation_params.strength
    )
    # network.add_learning_rule(learning_rule) # Incorrect: NeuralNetwork does not have this method
    input_to_output_synapses.set_learning_rule(learning_rule) # Correct: Set on the SynapseCollection

    # 6. 配置奖励处理器
    if config.reward_processor.type == 'SlidingWindowSmoother':
        reward_smoother = SlidingWindowSmoother(window_size=config.reward_processor.smoothing_window_size)
    elif config.reward_processor.type is None:
        reward_smoother = None # No smoothing
    else:
        raise ValueError(f"未知的奖励处理器类型: {config.reward_processor.type}")


    # 7. 配置数据探针
    probes_data = {} # To store data from probes if needed for saving later

    # 初始化用于存储所有轮次数据的列表
    all_episodes_input_spikes_data = []
    all_episodes_output_spikes_data = []
    all_episodes_weights_data = []
    all_episodes_raw_rewards_data = []
    all_episodes_smoothed_rewards_data = []
    all_episodes_actions_data = []

    # 计算以仿真步数为单位的记录间隔
    record_interval_steps = 1 # 默认值为1，即每步都记录 (如果 record_interval_ms <= 0)
    if config.probes.record_interval_ms > 0:
        if config.simulation.dt > 0:
            record_interval_steps = int(config.probes.record_interval_ms / config.simulation.dt)
            if record_interval_steps <= 0: # 确保至少为1，如果除法结果小于1
                record_interval_steps = 1
        else: # dt 为0或负数，则无法计算，默认每步记录
            print(f"警告: 仿真 dt ({config.simulation.dt}) 无效，探针将每步记录。")
            record_interval_steps = 1
    else: # record_interval_ms 配置为0或负数，也表示不按时间间隔记录（或禁用）
          # 但为了让代码继续，我们可能仍然希望有探针，只是不经常记录
          # 或者，如果 record_interval_ms <= 0 意味着禁用探针，下面的 if 判断会处理
          pass # record_interval_steps 保持其初始值，下面的 if 会判断是否创建探针

    if config.probes.record_interval_ms > 0: # 仅当配置的记录间隔（毫秒）为正时才添加探针
        input_spike_probe = PopulationProbe(
            name="input_spikes", # Provide a name
            population_name=input_pop.name, # Pass population name (string)
            state_vars=['fired'],             # Pass state_vars as a list
            record_interval=record_interval_steps # Pass interval in steps
        )
        network.add_probe(input_spike_probe)

        output_spike_probe = PopulationProbe(
            name="output_spikes",
            population_name=output_pop.name,
            state_vars=['fired'],
            record_interval=record_interval_steps
        )
        network.add_probe(output_spike_probe)

        weights_probe = SynapseProbe(
            name="input_output_weights",
            synapse_collection_name=input_to_output_synapses.name, # Pass synapse collection name
            record_weights=True, # Default is True, can be explicit
            record_interval=record_interval_steps
        )
        network.add_probe(weights_probe)

        # Store data for custom logging
        # These lists will be populated by the custom logger
        logged_raw_rewards = [] # List to store (time_ms, raw_reward) for the CURRENT episode
        logged_smoothed_rewards = [] # List to store (time_ms, smoothed_reward) for the CURRENT episode
        logged_actions = [] # List to store (time_ms, action) for the CURRENT episode
        # Removed CustomDataProbe and its callback

    # 8. 创建仿真器
    simulator = Simulator(network=network)

    # 9. 仿真循环
    num_episodes = config.simulation.num_episodes # 假设配置
    max_steps_per_episode = config.simulation.max_steps_per_episode # 假设配置
    total_rewards_per_episode = []

    print(f"开始 {num_episodes} 轮实验...")
    start_time = time.time()

    # 修改 tqdm 循环以允许更新 postfix
    episode_iterator = tqdm(range(num_episodes), desc="运行轮次", unit="轮")
    for episode in episode_iterator: 
        observation, info = env.reset()
        network.reset() # Correct method name to reset network, populations, and probes
        if hasattr(reward_smoother, 'reset'): reward_smoother.reset()
        
        # 重置探针数据 和 自定义日志列表 (network.reset() 应该已经处理了探针的 reset)
        # for probe in network.probes.values(): # network.probes is a list, not dict
        #     probe.reset()
        # 上面的循环不再需要，因为 network.reset() 会调用所有探针的 reset 方法。
        # 但是，自定义的 logged_X 列表仍然需要在这里清除。
        if config.probes.record_interval_ms > 0: 
            logged_raw_rewards.clear()
            logged_smoothed_rewards.clear()
            logged_actions.clear()

        episode_reward = 0
        
        # 初始编码一次观察值，得到初始输入电流/脉冲
        # 注意: MountainCar环境在reset后就给出了第一个observation
        # 我们需要在第一个仿真步之前就编码它
        position = observation[0]
        input_currents = input_encoder.encode(position) # 假设返回与输入神经元数量一致的数组
        
        # 为了简化，我们假设在每个 Gym step (dt_env) 中，SNN 运行 N 个内部步骤 (dt_snn)
        # dt_env 通常是 20ms (0.02s)
        # dt_snn 通常是 1ms (0.001s)
        # 所以 N = dt_env / dt_snn
        # 这里为了简化，我们假设 Gym step 和 SNN step 之间有一定的倍数关系
        # 或者更简单，每个 Gym step, SNN 运行固定的毫秒数
        
        # 例如，让SNN运行10ms来处理一个观察值并决定动作
        snn_run_duration_ms_per_env_step = config.simulation.snn_run_duration_ms_per_env_step # 假设配置

        for step in range(max_steps_per_episode):
            # 9.1 应用当前编码的输入到网络
            # 在SNN的第一个时间步之前或之中设置输入
            # 这里我们假设在运行SNN之前，通过某种方式将input_currents设置到输入神经元
            # 例如，可以有一个 network.set_external_current(population_name, currents) 的方法
            # 或者神经元群体有 receive_external_input(currents)
            # 为了简单起见，我们假设在simulator.run_for_duration内部处理了
            # input_pop.external_current = input_currents # 这是一个简化的概念
            
            # 9.2 运行SNN一小段时间来处理输入并产生输出脉冲
            # 我们需要获取在 simulator.run_for_duration 期间 output_pop 的脉冲
            # 传递 input_currents 到 simulator，让它在第一步设置

            # 定义一个输入生成器函数，它在每次被调用时都返回当前的 input_currents
            # 忽略参数 current_time, dt, previous_step_outputs，因为输入是固定的
            current_input_map = {'InputPopulation': input_currents}
            def static_input_generator(ct, sim_dt, prev_out):
                return current_input_map

            simulator.run_for_duration(
                total_duration=snn_run_duration_ms_per_env_step, # Corrected param name
                input_generator_fn=static_input_generator # Pass the generator
                # external_inputs 参数已被移除，通过 input_generator_fn 提供
            )

            # 9.3 从SNN输出解码动作
            # output_decoder 需要知道 output_pop 在最近一段时间的脉冲
            # 这通常通过查询 output_pop.get_spikes_since_last_query() 或类似方法实现
            # 或者，更简单的方式是，output_decoder 直接访问 output_pop 的当前脉冲状态
            # (假设在 WinnerTakesAllDecoder 内部实现)
            output_spikes_array = output_pop.get_spikes() # Correct method to get current spike states
            
            # The decoder expects a map {population_name: spike_array}
            spike_map_for_decoder = {output_pop.name: output_spikes_array}
            action = output_decoder.decode(spike_map_for_decoder)
            
            # 调试: 打印动作的类型和值
            # print(f"解码得到的动作: {action} (类型: {type(action)})")
            if action is None:
                print("错误: 解码器返回的动作为 None！检查解码器逻辑和默认动作设置。")
                # 可以选择在这里设置一个默认动作，或者抛出错误以停止
                # action = env.action_space.sample() # 例如，随机动作
                # action = 1 # 或者一个固定的默认动作，如"不推"
                # Forcing an error here if None, to ensure env.step() doesn't get None
                assert action is not None, "解码后的动作为 None，无法传递给 env.step()"

            # 9.4 执行动作并获得环境反馈
            observation, reward, terminated, truncated, info = env.step(action)

            # 9.5 处理奖励信号
            current_smoothed_reward = reward_smoother.process(reward) if reward_smoother else reward

            # 9.6 将平滑奖励信号传递给学习规则 (STDP调制)
            if learning_rule and current_smoothed_reward is not None: 
                learning_rule.update_reward_signal(current_smoothed_reward)

            # Log raw reward, smoothed reward, and action for this environment step
            # Use the network time at the end of this SNN run block as the timestamp
            current_network_time_ms = simulator.current_time # Corrected attribute name
            if config.probes.record_interval_ms > 0: # Log only if probes are generally active
                logged_raw_rewards.append((current_network_time_ms, reward))
                logged_smoothed_rewards.append((current_network_time_ms, current_smoothed_reward))
                logged_actions.append((current_network_time_ms, action))

            # 9.7 编码新的观测值，准备下一次SNN运行
            next_position = observation[0]
            input_currents = input_encoder.encode(next_position)

            episode_reward += reward
            observation = next_position, observation[1]

            if terminated or truncated:
                break
        
        total_rewards_per_episode.append(episode_reward)
        # print(f"轮次 {episode + 1}/{num_episodes} 完成: 总奖励 = {episode_reward}, 步数 = {step + 1}")
        episode_iterator.set_postfix(轮次=f"{episode + 1}/{num_episodes}", 总奖励=episode_reward, 步数=step + 1, refresh=True)

        # (可选) 在每轮结束后保存或分析探针数据 -> 修改为在每轮结束后收集数据
        if config.probes.record_interval_ms > 0 and config.probes.save_to_csv:
            # output_dir = config.probes.output_dir # output_dir将在最后使用
            # os.makedirs(output_dir, exist_ok=True) # 也移到最后
            
            # 收集探针数据
            if 'input_spikes' in network.probes:
                probe_obj = network.probes['input_spikes']
                probe_full_data = probe_obj.get_data() # {'time': [...], 'data': {'fired': [...]}}
                timestamps = probe_full_data.get('time', [])
                fired_arrays_list = probe_full_data.get('data', {}).get('fired', [])
                
                processed_spike_data_for_ep = []
                for t_idx, fired_array_at_t in enumerate(fired_arrays_list):
                    if t_idx < len(timestamps): # 确保时间戳和数据对齐
                        current_timestamp = timestamps[t_idx]
                        if fired_array_at_t is not None: # 检查数据是否有效
                            firing_neuron_indices = np.where(fired_array_at_t)[0]
                            for neuron_id in firing_neuron_indices:
                                processed_spike_data_for_ep.append((current_timestamp, neuron_id))
                all_episodes_input_spikes_data.append({'episode': episode + 1, 'data': processed_spike_data_for_ep})

            if 'output_spikes' in network.probes:
                probe_obj = network.probes['output_spikes']
                probe_full_data = probe_obj.get_data()
                timestamps = probe_full_data.get('time', [])
                fired_arrays_list = probe_full_data.get('data', {}).get('fired', [])
                
                processed_spike_data_for_ep = []
                for t_idx, fired_array_at_t in enumerate(fired_arrays_list):
                    if t_idx < len(timestamps):
                        current_timestamp = timestamps[t_idx]
                        if fired_array_at_t is not None:
                            firing_neuron_indices = np.where(fired_array_at_t)[0]
                            for neuron_id in firing_neuron_indices:
                                processed_spike_data_for_ep.append((current_timestamp, neuron_id))
                all_episodes_output_spikes_data.append({'episode': episode + 1, 'data': processed_spike_data_for_ep})

            if 'input_output_weights' in network.probes:
                # SynapseProbe.get_data()返回 {'time': [...], 'data': {'weights': [(w_arr1_at_t1), (w_arr2_at_t2), ...]}}
                # SynapseProbe 内部 self.data['weights'] 存储的是 (weights_array) 列表。
                # 之前的逻辑是直接访问 self.data['weights']，但它的时间戳在哪里？
                # SynapseProbe._collect_data appends syn_collection.get_weights().copy()，这里没有时间戳。
                # BaseProbe.attempt_record appends current_time_ms to self.time_data.
                # So, SynapseProbe.get_data() should correctly pair them.
                
                probe_obj = network.probes['input_output_weights']
                probe_full_data = probe_obj.get_data() # {'time': [...], 'data': {'weights': [...]}}
                timestamps = probe_full_data.get('time', [])
                weights_arrays_list = probe_full_data.get('data', {}).get('weights', [])
                
                processed_weights_data_for_ep = []
                for t_idx, weights_array_at_t in enumerate(weights_arrays_list):
                    if t_idx < len(timestamps):
                        current_timestamp = timestamps[t_idx]
                        if weights_array_at_t is not None:
                             processed_weights_data_for_ep.append((current_timestamp, weights_array_at_t))
                all_episodes_weights_data.append({'episode': episode + 1, 'data': processed_weights_data_for_ep})
            
            # 收集自定义日志数据
            all_episodes_raw_rewards_data.append({'episode': episode + 1, 'data': list(logged_raw_rewards)})
            all_episodes_smoothed_rewards_data.append({'episode': episode + 1, 'data': list(logged_smoothed_rewards)})
            all_episodes_actions_data.append({'episode': episode + 1, 'data': list(logged_actions)})
            
            # 注释掉原来的每轮次文件写入操作
            # input_spike_probe.export_to_csv(os.path.join(output_dir, f"ep{episode+1}_input_spikes.csv"))
            # output_spike_probe.export_to_csv(os.path.join(output_dir, f"ep{episode+1}_output_spikes.csv"))
            # weights_probe.export_to_csv(os.path.join(output_dir, f"ep{episode+1}_weights.csv"))
            
            # header_reward = "network_time_ms,raw_reward"
            # header_action = "network_time_ms,action"
            
            # with open(os.path.join(output_dir, f"ep{episode+1}_raw_rewards.csv"), 'w') as f_reward:
            #     f_reward.write(header_reward + '\n')
            #     for entry in logged_raw_rewards:
            #         f_reward.write(f"{entry[0]},{entry[1]}\n")
            
            # with open(os.path.join(output_dir, f"ep{episode+1}_smoothed_rewards.csv"), 'w') as f_s_reward:
            #     f_s_reward.write("network_time_ms,smoothed_reward\n")
            #     for entry in logged_smoothed_rewards:
            #         f_s_reward.write(f"{entry[0]},{entry[1]}\n")

            # with open(os.path.join(output_dir, f"ep{episode+1}_actions.csv"), 'w') as f_action:
            #     f_action.write(header_action + '\n')
            #     for entry in logged_actions:
            #         f_action.write(f"{entry[0]},{entry[1]}\n")
            # # print(f"探针数据已保存到 {output_dir} 目录 (轮次 {episode+1})") # 注释掉此行

    end_time = time.time()
    
    # 诊断：在写入文件前打印收集到的数据信息
    if config.probes.record_interval_ms > 0 and config.probes.save_to_csv:
        print(f"诊断: all_episodes_weights_data 收集到的轮次数: {len(all_episodes_weights_data)}")
        if all_episodes_weights_data:
            print(f"诊断: 第一个轮次的权重数据条目数: {len(all_episodes_weights_data[0]['data'])}")
            if all_episodes_weights_data[0]['data']:
                print(f"诊断: 第一个轮次的第一条权重数据样本: {all_episodes_weights_data[0]['data'][0]}")
        
        print(f"诊断: all_episodes_raw_rewards_data 收集到的轮次数: {len(all_episodes_raw_rewards_data)}")
        if all_episodes_raw_rewards_data:
            print(f"诊断: 第一个轮次的原始奖励数据条目数: {len(all_episodes_raw_rewards_data[0]['data'])}")
            if all_episodes_raw_rewards_data[0]['data']:
                print(f"诊断: 第一个轮次的第一条原始奖励数据样本: {all_episodes_raw_rewards_data[0]['data'][0]}")

    # 在实验总结前，执行所有数据文件的写入操作
    if config.probes.record_interval_ms > 0 and config.probes.save_to_csv:
        output_dir = config.probes.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"开始将所有轮次的探针和日志数据保存到 {output_dir} 目录...")

        # 保存 Input Spikes 数据
        for item in all_episodes_input_spikes_data:
            ep_num = item['episode']
            data_to_write = item['data']
            filepath = os.path.join(output_dir, f"ep{ep_num}_input_spikes.csv")
            with open(filepath, 'w') as f:
                f.write("timestamp,neuron_id\n") # PopulationProbe (fired) header
                for timestamp, neuron_id in data_to_write:
                    f.write(f"{timestamp},{neuron_id}\n")

        # 保存 Output Spikes 数据
        for item in all_episodes_output_spikes_data:
            ep_num = item['episode']
            data_to_write = item['data']
            filepath = os.path.join(output_dir, f"ep{ep_num}_output_spikes.csv")
            with open(filepath, 'w') as f:
                f.write("timestamp,neuron_id\n") # PopulationProbe (fired) header
                for timestamp, neuron_id in data_to_write:
                    f.write(f"{timestamp},{neuron_id}\n")

        # 保存 Weights 数据
        for item in all_episodes_weights_data:
            ep_num = item['episode']
            # data is list of (timestamp, weights_array)
            data_to_write = item['data'] 
            filepath = os.path.join(output_dir, f"ep{ep_num}_weights.csv")
            with open(filepath, 'w') as f:
                if data_to_write: # Ensure there's data to determine num_weights
                    num_weights = data_to_write[0][1].size
                    header = "timestamp," + ",".join([f"weight_{i}" for i in range(num_weights)])
                    f.write(header + "\n")
                    for timestamp, weights_array in data_to_write:
                        weights_str = ",".join(map(str, weights_array))
                        f.write(f"{timestamp},{weights_str}\n")
                else:
                    f.write("timestamp\n") # Empty file with header if no weights data for an episode

        # 保存 Raw Rewards 数据
        header_reward = "network_time_ms,raw_reward"
        for item in all_episodes_raw_rewards_data:
            ep_num = item['episode']
            data_to_write = item['data'] # list of (time_ms, value)
            filepath = os.path.join(output_dir, f"ep{ep_num}_raw_rewards.csv")
            with open(filepath, 'w') as f:
                f.write(header_reward + '\n')
                for time_ms, value in data_to_write:
                    f.write(f"{time_ms},{value}\n")
        
        # 保存 Smoothed Rewards 数据
        header_smoothed_reward = "network_time_ms,smoothed_reward"
        for item in all_episodes_smoothed_rewards_data:
            ep_num = item['episode']
            data_to_write = item['data']
            filepath = os.path.join(output_dir, f"ep{ep_num}_smoothed_rewards.csv")
            with open(filepath, 'w') as f:
                f.write(header_smoothed_reward + '\n')
                for time_ms, value in data_to_write:
                    f.write(f"{time_ms},{value}\n")

        # 保存 Actions 数据
        header_action = "network_time_ms,action"
        for item in all_episodes_actions_data:
            ep_num = item['episode']
            data_to_write = item['data']
            filepath = os.path.join(output_dir, f"ep{ep_num}_actions.csv")
            with open(filepath, 'w') as f:
                f.write(header_action + '\n')
                for time_ms, value in data_to_write:
                    f.write(f"{time_ms},{value}\n")
        
        print(f"所有轮次的探针和日志数据已保存到 {output_dir} 目录。")


    # 在实验总结前，添加一条关于探针数据保存的总提示 -> 已被上面的详细保存过程替代和覆盖
    # if config.probes.record_interval_ms > 0 and config.probes.save_to_csv:
    #     output_dir_summary = config.probes.output_dir # 获取一次目录，避免重复访问
    #     print(f"所有轮次的探针数据已保存到 {output_dir_summary} 目录。")


    print(f"实验完成，总耗时: {end_time - start_time:.2f} 秒")
    print(f"每轮平均奖励: {np.mean(total_rewards_per_episode)}")

    # 结果可视化
    if config.probes.save_to_csv: # Only plot if data was likely saved/meaningful
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(total_rewards_per_episode)
            plt.xlabel("轮次 (Episode)")
            plt.ylabel("总奖励 (Total Reward)")
            plt.title(f"MountainCar-v0 实验奖励 ({config.experiment.name})")
            plt.grid(True)
            # Save the plot to the same directory as other results
            plot_filename = os.path.join(config.probes.output_dir, f"rewards_per_episode_{config.experiment.name}.png")
            plt.savefig(plot_filename)
            print(f"奖励曲线图已保存到: {plot_filename}")
            # plt.show() # Optionally show plot if running interactively
        except Exception as e:
            print(f"无法生成或保存奖励曲线图: {e}")

    env.close()
    return total_rewards_per_episode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 DyNN MountainCar-v0 实验.")
    parser.add_argument(
        '--config', 
        type=str, 
        help='可选的配置文件路径 (YAML)。如果未提供，则使用默认配置。'
    )
    args = parser.parse_args()
    
    try:
        run_mountain_car_experiment(config_path=args.config)
    except Exception as e:
        print("实验运行期间发生未捕获的异常:")
        import traceback
        traceback.print_exc() 