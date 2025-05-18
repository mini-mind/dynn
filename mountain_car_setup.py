"""
Setup functions for MountainCar experiment using DyNN ExperimentSession framework.
Also includes plotting utilities.
"""
import gym
import numpy as np
import os
import matplotlib.pyplot as plt

# 添加 Matplotlib 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from dynn.core.network import NeuralNetwork
from dynn.core.neurons import NeuronPopulation, IzhikevichNeuron, LIFNeuron
from dynn.core.synapses import SynapseCollection
from dynn.core.learning_rules import RewardModulatedSTDP, VoltageTripletSTDP, STDP
from dynn.io import GaussianEncoder, InstantaneousSpikeCountDecoder, BaseRewardProcessor, SlidingWindowSmoother, BidirectionalThresholdDecoder

def get_env_properties(env):
    obs_space = env.observation_space
    act_space = env.action_space
    return {
        'observation_low': obs_space.low,
        'observation_high': obs_space.high,
        'observation_shape': obs_space.shape,
        'action_n': act_space.n if hasattr(act_space, 'n') else None,
        'action_type': 'discrete' if hasattr(act_space, 'n') else 'continuous'
    }

def create_mountain_car_env(env_config):
    """
    Creates the MountainCar-v0 environment.
    'env_config' (dict): Configuration for the environment, e.g., {'id': 'MountainCar-v0'}.
    """
    env_id = env_config.get("id", "MountainCar-v0")
    try:
        env = gym.make(env_id)
        print(f"Environment '{env_id}' created successfully.")
        return env
    except gym.error.Error as e:
        print(f"Error creating Gym environment '{env_id}': {e}")
        raise

def build_mountain_car_network(network_config, env_props):
    """
    Builds the SNN for the MountainCar experiment.
    'network_config' (dict): Configuration for the network.
    'env_props' (dict): Properties of the environment (obs/action space).
    """
    print(f"Building SNN with config: {network_config}")
    net = NeuralNetwork(name=network_config.get("name", "MountainCarSNN"))

    neuron_model_type = network_config.get("neuron_model", "izhikevich").lower()
    common_neuron_params = network_config.get("neuron_params", {})

    input_pop_config = None
    output_pop_config = None
    
    if 'populations' in network_config and network_config['populations']:
        print(f"DEBUG build_mountain_car_network: Found 'populations' key. Num pops: {len(network_config['populations'])}")
        for i, pop_conf in enumerate(network_config['populations']):
            current_pop_name = pop_conf.get('name')
            print(f"DEBUG build_mountain_car_network: Iterating pop {i}, name: '{current_pop_name}', conf: {pop_conf}")
            if current_pop_name == "InputPopulation":
                input_pop_config = pop_conf
                print(f"DEBUG build_mountain_car_network: Assigned input_pop_config")
            elif current_pop_name == "OutputPopulation":
                output_pop_config = pop_conf
                print(f"DEBUG build_mountain_car_network: Assigned output_pop_config")
    else:
        print("DEBUG build_mountain_car_network: 'populations' key missing or empty.")
    
    if not input_pop_config:
        raise ValueError("Network config missing 'input_population' details.")

    if neuron_model_type == "lif":
        lif_params_for_pop = common_neuron_params.get('lif', {})
        input_neurons = NeuronPopulation(name=input_pop_config['name'],
                                   num_neurons=network_config['input_neurons_count'],
                                   neuron_model_class=LIFNeuron, 
                                   neuron_params=lif_params_for_pop)
    elif neuron_model_type == "izhikevich":
        default_izh_params = {'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0}
        izh_model_params = {**default_izh_params, **common_neuron_params.get('izhikevich', {})}
        input_neurons = NeuronPopulation(name=input_pop_config['name'], 
                                        num_neurons=network_config['input_neurons_count'],
                                        neuron_model_class=IzhikevichNeuron,
                                        neuron_params=izh_model_params)
    else:
        raise ValueError(f"Unsupported neuron model type: {neuron_model_type}")
    net.add_population(input_neurons)
    net.set_input_populations([input_neurons.name])

    if not output_pop_config:
        raise ValueError("Network config missing 'output_population' details.")

    if neuron_model_type == "lif":
        lif_params_for_pop = common_neuron_params.get('lif', {})
        output_neurons = NeuronPopulation(name=output_pop_config['name'],
                                   num_neurons=network_config['output_neurons_count'],
                                   neuron_model_class=LIFNeuron,
                                   neuron_params=lif_params_for_pop)
    elif neuron_model_type == "izhikevich":
        default_izh_params = {'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0}
        izh_model_params = {**default_izh_params, **common_neuron_params.get('izhikevich', {})}
        output_neurons = NeuronPopulation(name=output_pop_config['name'], 
                                         num_neurons=network_config['output_neurons_count'],
                                         neuron_model_class=IzhikevichNeuron,
                                         neuron_params=izh_model_params)
    net.add_population(output_neurons)
    net.set_output_populations([output_neurons.name])

    synapses_configs = network_config.get("synapses", [])
    for syn_conf in synapses_configs:
        pre_pop = net.get_population(syn_conf['pre_population_name'])
        post_pop = net.get_population(syn_conf['post_population_name'])

        weight_init_conf = syn_conf.get('weight_init', {'method': 'normal', 'mean': 0.5, 'std': 0.1})
        weights = None
        if weight_init_conf['method'] == 'normal':
            weights = np.random.normal(weight_init_conf['mean'], weight_init_conf['std'],
                                       size=(len(pre_pop), len(post_pop)))
        elif weight_init_conf['method'] == 'uniform':
             weights = np.random.uniform(weight_init_conf['low'], weight_init_conf['high'],
                                       size=(len(pre_pop), len(post_pop)))
        elif weight_init_conf['method'] == 'fixed':
            weights = np.full((len(pre_pop), len(post_pop)), weight_init_conf['value'])
        else:
            raise ValueError(f"Unsupported weight init method: {weight_init_conf['method']}")
        weights = np.clip(weights, syn_conf.get('weight_min', 0), syn_conf.get('weight_max', 1.0))

        lr_name = syn_conf.get('learning_rule')
        learning_rule_instance = None
        if lr_name:
            lr_params = syn_conf.get('lr_params', {})
            if lr_name == "RewardModulatedSTDP":
                learning_rule_instance = RewardModulatedSTDP(**lr_params)
            elif lr_name == "VoltageTripletSTDP":
                learning_rule_instance = VoltageTripletSTDP(**lr_params)
            elif lr_name == "STDP":
                learning_rule_instance = STDP(**lr_params)
            else:
                raise ValueError(f"Unsupported learning rule: {lr_name}")

        syn_collection = SynapseCollection(
            name=syn_conf['name'],
            pre_population=pre_pop,
            post_population=post_pop
        )

        weight_init_method = syn_conf.get('initial_weights', {}).get('strategy', 'normal')
        connectivity_type = syn_conf.get('connectivity_type', 'full')
        
        dist_config_for_init = None
        if weight_init_method == 'normal':
            mean = syn_conf.get('initial_weights', {}).get('mean', 0.5)
            std = syn_conf.get('initial_weights', {}).get('std', 0.1)
            dist_config_for_init = ('normal', (mean, std))
        elif weight_init_method == 'uniform':
            low = syn_conf.get('initial_weights', {}).get('low', 0.0)
            high = syn_conf.get('initial_weights', {}).get('high', 1.0)
            dist_config_for_init = ('uniform', (low, high))
        elif weight_init_method == 'fixed':
            value = syn_conf.get('initial_weights', {}).get('value', 0.5)
            dist_config_for_init = float(value)
        else:
            print(f"Warning: Unsupported weight init strategy '{weight_init_method}'. Using default zero weights.")

        if dist_config_for_init:
            syn_collection.initialize_weights(dist_config=dist_config_for_init, connectivity_type=connectivity_type)
        
        if learning_rule_instance:
            syn_collection.set_learning_rule(learning_rule_instance)
            if hasattr(learning_rule_instance, 'w_min'):
                learning_rule_instance.w_min = syn_conf.get('weight_limits', {}).get('min', 0.0)
            if hasattr(learning_rule_instance, 'w_max'):
                learning_rule_instance.w_max = syn_conf.get('weight_limits', {}).get('max', 1.0)

        net.add_synapses(syn_collection)

    print(f"Network '{net.name}' built with {len(net.populations)} populations and {len(net.synapses)} synapse groups.")
    return net

def create_mountain_car_input_encoder(encoder_config, input_pop_instance, env_props):
    """
    Creates an input encoder for MountainCar.
    'encoder_config' (dict): Configuration for the encoder.
    'input_pop_instance' (NeuronPopulation): Target input population instance.
    'env_props' (dict): Environment properties (e.g., observation space for min/max).
    """
    encoder_type = encoder_config.get("type", "GaussianEncoder")
    obs_idx_to_encode = encoder_config.get("observation_index_to_encode", 0)

    min_val = encoder_config.get("min_val", env_props['observation_low'][obs_idx_to_encode])
    max_val = encoder_config.get("max_val", env_props['observation_high'][obs_idx_to_encode])

    if encoder_type == "GaussianEncoder":
        actual_encoder = GaussianEncoder(
            target_pop_name=input_pop_instance.name,
            num_neurons=len(input_pop_instance),
            min_val=min_val,
            max_val=max_val,
            sigma_scale=encoder_config.get("sigma_scale", 0.15),
            current_amplitude=encoder_config.get("current_amplitude", 10.0)
        )
    else:
        raise ValueError(f"Unsupported input encoder type: {encoder_type}")

    class ObservationSelectiveEncoder:
        def __init__(self, core_encoder, observation_index):
            self.core_encoder = core_encoder
            self.observation_index = observation_index
            self.target_pop_name = core_encoder.target_pop_name

        def encode(self, observation, dt=None, current_time=None):
            # Ensure value_to_encode is a Python scalar float
            value_to_encode = float(observation[self.observation_index])
            return self.core_encoder.encode(value_to_encode, dt=dt, current_time=current_time)

        def reset(self):
            if hasattr(self.core_encoder, 'reset'):
                self.core_encoder.reset()

        def __repr__(self):
            return f"ObservationSelectiveEncoder(idx={self.observation_index}, core={self.core_encoder!r})"

    print(f"Input encoder created: ObservationSelectiveEncoder wrapping {encoder_type}")
    return ObservationSelectiveEncoder(actual_encoder, obs_idx_to_encode)

def create_mountain_car_output_decoder(decoder_config, output_pop_instance, env_props):
    """
    Creates an output decoder for MountainCar.
    'decoder_config' (dict): Configuration for the decoder.
    'output_pop_instance' (NeuronPopulation): Source output population instance.
    'env_props' (dict): Environment properties (e.g., action space size).
    """
    decoder_type = decoder_config.get("type", "InstantaneousSpikeCountDecoder")
    num_actions = decoder_config.get("num_actions", env_props['action_n'])

    if len(output_pop_instance) != num_actions:
        print(f"Warning: Output population size ({len(output_pop_instance)}) does not match num_actions ({num_actions}).")

    if decoder_type == "InstantaneousSpikeCountDecoder":
        decoder = InstantaneousSpikeCountDecoder(
            source_pop_name=output_pop_instance.name,
            num_actions=num_actions,
            default_action=decoder_config.get("default_action", 1)
        )
    elif decoder_type == "BidirectionalThresholdDecoder":
        decoder = BidirectionalThresholdDecoder(
            source_pop_name=output_pop_instance.name,
            num_actions=num_actions,
            action_threshold=decoder_config.get("action_threshold", 0),
            default_action=decoder_config.get("default_action", 1),
            action_left=decoder_config.get("action_left", 0),
            action_right=decoder_config.get("action_right", 2)
        )
    else:
        raise ValueError(f"Unsupported output decoder type: {decoder_type}")

    print(f"Output decoder created: {decoder_type}")
    return decoder

class MountainCarShapedRewardProcessor(BaseRewardProcessor):
    def __init__(self, position_weight=0.1, velocity_weight=0.0, goal_reward_bonus=0.0,
                 height_bonus_factor=0.0, min_pos_for_height_bonus=-0.4, **kwargs):
        super().__init__(**kwargs)
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.goal_reward_bonus = goal_reward_bonus
        self.height_bonus_factor = height_bonus_factor
        self.min_pos_for_height_bonus = min_pos_for_height_bonus
        print(f"ShapedReward: pos_w={position_weight}, vel_w={velocity_weight}, goal_bonus={goal_reward_bonus}, height_bonus={height_bonus_factor}")

    def process(self, reward, observation, action, next_observation, done, **kwargs):
        pos, vel = next_observation[0], next_observation[1]

        shaped_r = 0.0
        shaped_r += self.position_weight * pos
        shaped_r += self.velocity_weight * vel

        if pos > self.min_pos_for_height_bonus:
            shaped_r += self.height_bonus_factor * (pos - self.min_pos_for_height_bonus)

        final_reward = reward + shaped_r

        if done and pos >= 0.5: # Goal condition for MountainCar
            final_reward += self.goal_reward_bonus

        return final_reward

    def reset(self):
        pass

def create_mountain_car_reward_processor(rp_config):
    """
    Creates a reward processor, possibly with shaping and/or smoothing.
    'rp_config' (dict): Configuration for the reward processor.
    Returns a BaseRewardProcessor instance or None.
    """
    if not rp_config:
        return None

    processor_to_use = None
    if rp_config.get("use_shaped_reward", False):
        processor_to_use = MountainCarShapedRewardProcessor(
            position_weight=rp_config.get("position_weight", 0.1),
            velocity_weight=rp_config.get("velocity_weight", 0.0),
            goal_reward_bonus=rp_config.get("goal_reward_bonus", 0.0),
            height_bonus_factor=rp_config.get("height_bonus_factor", 0.0),
            min_pos_for_height_bonus=rp_config.get("min_pos_for_height_bonus", -0.4)
        )
        print("Using shaped reward processor.")

    smoother_window = rp_config.get("smoother_window_size")
    if smoother_window and smoother_window > 0:
        smoother = SlidingWindowSmoother(window_size=smoother_window)
        print(f"Reward smoothing enabled with window size: {smoother_window}")
        if processor_to_use is None:
            return smoother
        else:
            class ChainedRewardProcessor(BaseRewardProcessor):
                def __init__(self, primary_processor, secondary_processor):
                    self.primary = primary_processor
                    self.secondary = secondary_processor
                def process(self, reward, observation, action, next_observation, done, **kwargs):
                    processed_reward = self.primary.process(reward, observation, action, next_observation, done, **kwargs)
                    return self.secondary.process(processed_reward)
                def reset(self):
                    if hasattr(self.primary, 'reset'): self.primary.reset()
                    if hasattr(self.secondary, 'reset'): self.secondary.reset()
                def __repr__(self):
                    return f"Chained({self.primary!r} -> {self.secondary!r})"
            return ChainedRewardProcessor(processor_to_use, smoother)

    return processor_to_use

def plot_experiment_results(config, total_rewards_per_episode_list):
    """绘制实验结果（每轮总奖励）并保存图像。"""
    probes_config = getattr(config, 'probes', None)
    if not probes_config or \
       not getattr(probes_config, 'record_interval_ms', 0) > 0 or \
       not getattr(probes_config, 'save_to_csv', False) or \
       not total_rewards_per_episode_list:
        print("信息: 未配置CSV保存、探针未激活或没有奖励数据，跳过绘图。")
        print(f"Debug: probes_config={probes_config}, record_interval_ms={getattr(probes_config, 'record_interval_ms', None)}, save_to_csv={getattr(probes_config, 'save_to_csv', None)}, list_empty={not total_rewards_per_episode_list}")
        return

    try:
        plt.figure(figsize=(10, 6))
        plt.plot(total_rewards_per_episode_list)
        plt.xlabel("轮次 (Episode)")
        plt.ylabel("总奖励 (Total Reward)")

        experiment_config = getattr(config, 'experiment', None)
        experiment_name = getattr(experiment_config, 'name', 'experiment') if experiment_config else 'experiment'

        plt.title(f"MountainCar-v0 实验奖励 ({experiment_name})")
        plt.grid(True)

        output_dir = getattr(probes_config, 'output_dir', '.')
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = os.path.join(output_dir, f"rewards_per_episode_{experiment_name}.png")
        plt.savefig(plot_filename)
        print(f"奖励曲线图已保存到: {plot_filename}")
    except Exception as e:
        print(f"无法生成或保存奖励曲线图: {e}")
        import traceback
        traceback.print_exc() 