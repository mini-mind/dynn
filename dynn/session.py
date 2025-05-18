class ExperimentSession:
    """
    管理整个实验流程，从设置到执行和数据记录。

    该类通过用户定义的创建者函数进行配置，负责协调环境、网络、Agent、
    模拟器以及数据记录等组件的交互。

    Attributes:
        config (dict): 实验的全局配置，通常从YAML文件加载。
        env (any): 实验中使用的环境实例 (例如 Gym 环境)。
        agent (dynn.Agent): 负责决策和学习的Agent实例。
        network (dynn.core.NeuralNetwork): SNN模型实例。
        simulator (dynn.core.Simulator): 驱动SNN的模拟器实例。
        data_recorder (any, optional): 用于记录实验数据的记录器实例。
        _is_setup_complete (bool): 标志实验是否完成设置。
    """
    def __init__(self, config):
        """
        初始化 ExperimentSession。

        Args:
            config (dict): 实验的全局配置字典。
        """
        self.config = config
        self.env = None
        self.agent = None
        self.network = None # Store network instance
        self.simulator = None # Store simulator instance
        self.data_recorder = None
        self._is_setup_complete = False

    def setup(self, env_creator_fn, network_builder_fn, input_encoder_creator_fn, 
              output_decoder_creator_fn, reward_processor_creator_fn=None, 
              agent_creator_fn=None, simulator_creator_fn=None):
        """
        设置实验环境，初始化所有必要的组件。

        通过传递用户定义的创建者/构建者函数来配置实验的各个部分。

        Args:
            env_creator_fn (callable): 用户提供的函数，无参数，返回一个初始化的环境实例。
            network_builder_fn (callable): 用户提供的函数，接收网络配置 (dict)，
                返回一个 `dynn.core.NeuralNetwork` 实例。
            input_encoder_creator_fn (callable): 用户提供的函数，接收编码器配置 (dict)
                和目标输入群体实例，返回一个 `dynn.io.BaseInputEncoder` 实例。
            output_decoder_creator_fn (callable): 用户提供的函数，接收解码器配置 (dict)
                和源输出群体实例，返回一个 `dynn.io.BaseOutputDecoder` 实例。
            reward_processor_creator_fn (callable, optional): 用户提供的函数，接收奖励处理器
                配置 (dict)，返回一个 `dynn.io.BaseRewardProcessor` 实例。默认为 None。
            agent_creator_fn (callable, optional): 用户提供的函数，接收Agent配置 (dict)、
                网络、模拟器、编码器、解码器和奖励处理器实例，返回一个 `dynn.Agent` 实例。
                如果为 None，则会实例化默认的 `dynn.Agent`。默认为 None。
            simulator_creator_fn (callable, optional): 用户提供的函数，接收模拟器配置 (dict)
                和网络实例，返回一个 `dynn.core.Simulator` 实例。如果为 None，
                则会实例化默认的 `dynn.core.Simulator`。默认为 None。
        """
        
        self.env = env_creator_fn()

        network_config = self.config.get('network', {}) 
        # Build Network
        if network_builder_fn:
            print(f"DEBUG: network_config in ExperimentSession.setup before calling builder: {network_config}")
            self.network = network_builder_fn(network_config)
            if self.network:
                print(f"Network built successfully. Network name: {self.network.name}")
            else:
                print("Warning: Network not built successfully.")

        # Instantiate Simulator
        # Option 1: Default Simulator instantiation
        if simulator_creator_fn:
            self.simulator = simulator_creator_fn(self.config.get('simulator', {}), self.network)
        else:
            from dynn.core.simulator import Simulator # Assumed path
            simulator_params = self.config.get('simulator', {})
            dt = simulator_params.get('dt', 1.0) # Get dt from config, default to 1.0
            self.simulator = Simulator(network=self.network, dt=dt)

        io_config = self.config.get('io', {})
        input_encoder_config = io_config.get('input_encoder', {})
        
        # Get target input population instance for the encoder
        # The name of the target population for the encoder should be in its config
        target_pop_name_for_encoder = input_encoder_config.get('target_population_name')
        if not target_pop_name_for_encoder:
            # Fallback: if network has designated input populations, use the first one.
            if self.network.input_population_names:
                target_pop_name_for_encoder = self.network.input_population_names[0]
                print(f"Warning: 'target_population_name' not in input_encoder_config. Using first network input pop: {target_pop_name_for_encoder}")
            else:
                raise ValueError("Input encoder target population name not specified in config and network has no designated input populations.")
        
        try:
            input_population_instance = self.network.get_population(target_pop_name_for_encoder)
        except KeyError as e:
            raise ValueError(f"Failed to get input population '{target_pop_name_for_encoder}' for encoder from network: {e}")

        input_encoder = input_encoder_creator_fn(input_encoder_config, input_population_instance)

        output_decoder_config = io_config.get('output_decoder', {})
        # Get source output population instance for the decoder
        source_pop_name_for_decoder = output_decoder_config.get('source_population_name')
        if not source_pop_name_for_decoder:
            if self.network.output_population_names:
                source_pop_name_for_decoder = self.network.output_population_names[0]
                print(f"Warning: 'source_population_name' not in output_decoder_config. Using first network output pop: {source_pop_name_for_decoder}")
            else:
                raise ValueError("Output decoder source population name not specified in config and network has no designated output populations.")
        
        try:
            output_population_instance = self.network.get_population(source_pop_name_for_decoder)
        except KeyError as e:
            raise ValueError(f"Failed to get output population '{source_pop_name_for_decoder}' for decoder from network: {e}")

        output_decoder = output_decoder_creator_fn(output_decoder_config, output_population_instance)

        reward_processor = None
        if reward_processor_creator_fn:
            reward_processor_config = io_config.get('reward_processor', {})
            reward_processor = reward_processor_creator_fn(reward_processor_config)

        if agent_creator_fn:
            self.agent = agent_creator_fn(self.config, self.network, self.simulator, input_encoder, output_decoder, reward_processor)
        else:
            agent_module = __import__('dynn.agent', fromlist=['Agent'])
            Agent = agent_module.Agent
            agent_specific_config = self.config.get('agent', {})
            self.agent = Agent(self.network, self.simulator, input_encoder, output_decoder, 
                               reward_processor, agent_config=agent_specific_config)
        
        self._setup_probes(self.network) # network is now self.network
        self._setup_data_recorder()
        self._is_setup_complete = True

    def run(self):
        """
        运行完整的实验流程。

        循环执行 `self.config.simulation.num_episodes` 次，
        在每次迭代中调用 `self.run_episode()`，记录数据，并在最后保存结果。

        Returns:
            list: 包含每个轮次数据的列表。每个元素是 `run_episode` 返回的字典。
        """
        num_episodes = self.config.get('simulation', {}).get('num_episodes', 1)
        all_episode_data = []
        for episode_num in range(num_episodes):
            print(f"Starting episode {episode_num + 1}/{num_episodes}")
            # Reset simulator at the start of each episode, including its time and network state
            if self.simulator:
                self.simulator.reset() # Resets network and sim time
            else:
                print("Warning: Simulator not available for reset in run().")
            
            episode_data = self.run_episode()
            all_episode_data.append(episode_data)
            print(f"Episode {episode_num + 1} finished. Total reward: {episode_data.get('total_reward', 'N/A')}")
        
        self._finalize_and_save_results(all_episode_data)
        return all_episode_data

    def is_setup_complete(self):
        return self._is_setup_complete

    def run_episode(self):
        """
        运行单个轮次的实验。

        在此方法中，Agent与环境交互，直到轮次结束 (done=True) 或达到最大步数。
        记录步级数据和轮次聚合数据。

        Returns:
            dict: 包含该轮次数据的字典，例如总奖励、步数和探针数据。
                  结构示例: {'steps': [...], 'probes': {...}, 'total_reward': ..., 'num_steps': ...}
        """
        if not self.env or not self.agent or not self.simulator or not self.network:
            raise Exception("Session not fully set up. Call setup() and ensure env, agent, sim, network are initialized.")

        initial_obs_tuple = self.env.reset()
        observation = initial_obs_tuple[0] if isinstance(initial_obs_tuple, tuple) else initial_obs_tuple
        self.agent.reset() # Agent reset its IO components
        # Network and simulator are reset at the start of run() method for each episode.
        # If simulator.reset() was not called, ensure network states are reset here or by Agent.
        # self.network.reset() # Redundant if sim.reset() calls it

        done = False
        total_reward = 0
        step_count = 0
        episode_log = {'steps': [], 'probes': {}} 

        max_steps_per_episode = self.config.get('simulation', {}).get('max_steps_per_episode', float('inf'))

        while not done and step_count < max_steps_per_episode:
            action = self.agent.get_action(observation)
            
            # Handle Gym API differences for env.step()
            try:
                # Modern Gym API (>=0.26.0) returns 5 values
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated 
            except ValueError:
                # Older Gym API might return 4 values (obs, rew, done, info)
                # Or some custom envs might still use 4.
                # This is a fallback, prefer explicit handling if Gym version is known.
                next_observation, reward, done, info = self.env.step(action)
                terminated = done # Assuming 'done' implies termination if truncated is not available
                truncated = False # Assuming not truncated if not explicitly returned

            self.agent.learn(observation, action, reward, next_observation, done)
            
            step_data = {
                'step': step_count,
                'observation': observation, 
                'action': action,
                'reward': reward,
                'next_observation': next_observation,
                'terminated': terminated, # Store individual flags
                'truncated': truncated,   # Store individual flags
                'done': done,             # Store combined done
                'info': info
            }
            episode_log['steps'].append(step_data)

            observation = next_observation
            total_reward += reward
            step_count += 1
            
            if step_count % 100 == 0: 
                 print(f"  Step {step_count}, Total Reward: {total_reward:.2f}, SimTime: {self.simulator.current_time:.2f}ms")

        episode_log['total_reward'] = total_reward
        episode_log['num_steps'] = step_count
        
        # Collect probe data at the end of the episode
        if self.network and hasattr(self.network, 'get_all_probes'):
            all_probes = self.network.get_all_probes()
            for probe_name, probe_instance in all_probes.items():
                episode_log['probes'][probe_name] = probe_instance.get_data()
        
        return episode_log

    def _setup_probes(self, network_instance):
        """
        根据配置在SNN网络中设置探针。

        探针用于记录网络活动，例如脉冲、电压或权重。

        Args:
            network_instance (dynn.core.NeuralNetwork): 要添加探针的网络实例。
        """
        probes_config = self.config.get('probes', [])
        if not probes_config or not network_instance:
            print("No probes configuration found or network not available for probe setup.")
            return

        # Ensure dynn.utils.probes can be imported
        try:
            from dynn.utils import probes as probe_module
        except ImportError:
            print("Error: `dynn.utils.probes` module not found. Cannot set up probes.")
            return

        for probe_conf in probes_config:
            probe_type_str = probe_conf.get('type')
            probe_name = probe_conf.get('name')
            # 'target' in config maps to population_name, synapse_collection_name, etc.
            target_id = probe_conf.get('target') 
            params_from_config = probe_conf.get('params', {})

            if not probe_type_str or not probe_name or not target_id:
                print(f"Skipping probe due to missing type, name, or target in config: {probe_conf}")
                continue
            
            probe_class = getattr(probe_module, probe_type_str, None)
            if not probe_class:
                print(f"Warning: Probe type '{probe_type_str}' not found in `dynn.utils.probes`.")
                continue
            
            try:
                # Prepare arguments for the specific probe class constructor
                probe_init_args = {'name': probe_name}
                probe_init_args.update(params_from_config) # Add common params like record_interval, and specific ones

                # Map target_id to the correct constructor argument based on probe type
                if probe_type_str == "PopulationProbe":
                    probe_init_args['population_name'] = target_id
                    # state_vars should be in params_from_config, e.g., {'state_vars': ['v', 'fired']}
                elif probe_type_str == "SynapseProbe":
                    probe_init_args['synapse_collection_name'] = target_id
                    # record_weights might be in params_from_config, e.g., {'record_weights': True}
                elif probe_type_str == "CustomDataProbe":
                    # For CustomDataProbe, 'target_id' might not be directly used in its constructor
                    # if data_provider_fn and data_keys are primary. 
                    # The config for CustomDataProbe would need to specify 'data_provider_fn_name' 
                    # (if it's a string to be resolved) and 'data_keys'.
                    # This part requires more thought on how data_provider_fn is passed/resolved.
                    # For now, assume params_from_config contains data_provider_fn and data_keys if needed.
                    # If target_id is still relevant (e.g. as a hint to data_provider_fn), it's in params_from_config.
                    pass # CustomDataProbe specific args (data_provider_fn, data_keys) are expected in params_from_config
                else:
                    # For other or generic BaseProbe-derived probes, they might take target_id directly
                    # or handle it via params. This basic mapping covers the known specific probes.
                    # If a new probe type has a different target arg name, it needs to be added here.
                    # Defaulting to not re-assigning target_id if type is unknown, assuming constructor handles it via **kwargs
                    # or it's not a target-specific probe in that way.
                    print(f"Info: Probe type '{probe_type_str}' not specifically handled for target mapping. Ensure constructor accepts relevant args from params or target_id.")
                    # If the probe takes target_id directly, like: SomeProbe(name='n', target_id='t', ...)
                    # probe_init_args['target_id'] = target_id # Uncomment if such probes exist

                probe_instance = probe_class(**probe_init_args)
                network_instance.add_probe(probe_instance)
                print(f"Probe '{probe_name}' ({probe_type_str}) added for target '{target_id}' (mapped to relevant constructor arg). Params: {params_from_config}")
            except TypeError as te:
                 print(f"TypeError setting up probe '{probe_name}': {te}. Check constructor arguments for {probe_type_str} and config params: {probe_init_args}")
            except Exception as e:
                print(f"Error setting up probe '{probe_name}' for target '{target_id}': {e}")

    def _setup_data_recorder(self):
        """
        初始化数据记录机制。

        此方法可以扩展以支持更复杂的数据记录需求，
        目前主要依赖探针数据和轮次日志。
        """
        # This method can be expanded if a more complex, centralized data recorder is needed
        recorder_config = self.config.get('data_recorder', {})
        if recorder_config:
            print(f"Setting up data recorder with config: {recorder_config} (actual implementation pending).")
            # Example: self.data_recorder = CentralDataRecorder(**recorder_config)
        else:
            print("No central data recorder configuration found. Probe data will be primary source.")
        pass 

    def _finalize_and_save_results(self, all_episode_data):
        """
        处理收集到的数据并将其保存到磁盘。

        保存路径、格式等由配置驱动。

        Args:
            all_episode_data (list): 包含所有轮次数据的列表。
        """
        output_config = self.config.get('output', {})
        output_dir = output_config.get('directory', 'results')
        experiment_name = self.config.get('experiment_name', 'dynn_experiment')
        output_format = output_config.get('format', 'pickle') # e.g., pickle, json, csv

        if not all_episode_data:
            print("No episode data to save.")
            return

        import os
        import pickle
        import json
        import datetime

        # Create a timestamped subdirectory for this run
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_output_dir = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
        
        try:
            os.makedirs(run_output_dir, exist_ok=True)
            print(f"Results will be saved to: {run_output_dir}")

            if output_format == 'pickle':
                file_path = os.path.join(run_output_dir, 'results.pkl')
                with open(file_path, 'wb') as f:
                    pickle.dump(all_episode_data, f)
                print(f"Saved all episode data to {file_path}")
            
            elif output_format == 'json':
                file_path = os.path.join(run_output_dir, 'results.json')
                # Need to handle numpy arrays for JSON serialization
                def default_json_serializer(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if hasattr(obj, 'tolist'): # For other array-like from libraries
                         return obj.tolist()
                    if isinstance(obj, (np.generic, np.bool_)):
                        return obj.item()
                    if isinstance(obj, datetime.datetime):
                        return obj.isoformat()
                    # Add more type handlers if necessary
                    try:
                        return str(obj) # Last resort for unknown types
                    except Exception:
                        return f"<Unserializable_object_{type(obj).__name__}>"

                with open(file_path, 'w') as f:
                    json.dump(all_episode_data, f, indent=4, default=default_json_serializer)
                print(f"Saved all episode data to {file_path}")
            
            # TODO: Add CSV saving for aggregated results or step-wise data if appropriate
            # Example: save aggregated per-episode data to a CSV
            if output_format == 'csv' or output_config.get('save_summary_csv', True):
                summary_data = []
                for i, ep_data in enumerate(all_episode_data):
                    summary_data.append({
                        'episode': i + 1,
                        'total_reward': ep_data.get('total_reward'),
                        'num_steps': ep_data.get('num_steps')
                        # Add other summary stats from ep_data as needed
                    })
                if summary_data:
                    import pandas as pd
                    df_summary = pd.DataFrame(summary_data)
                    summary_file_path = os.path.join(run_output_dir, 'summary_results.csv')
                    df_summary.to_csv(summary_file_path, index=False)
                    print(f"Saved summary results to {summary_file_path}")

            # Save the configuration file used for this experiment
            config_file_path = os.path.join(run_output_dir, 'config_used.json')
            with open(config_file_path, 'w') as f:
                json.dump(self.config, f, indent=4, default=str) # Use str for non-serializable
            print(f"Saved configuration to {config_file_path}")

        except Exception as e:
            print(f"Error during finalization and saving of results: {e}")

# Basic test structure (dependencies on dynn.core, dynn.io, dynn.agent, gym are not resolved here)
if __name__ == '__main__':
    print("This is a placeholder for ExperimentSession testing.")
    print("To run a test, you would need to:")
    print("1. Define mock or real creator functions for env, network, IO, agent.")
    print("2. Create a configuration dictionary.")
    print("3. Instantiate ExperimentSession with the config.")
    print("4. Call session.setup(...) with the creator functions.")
    print("5. Call session.run().")
    
    # Example mock config (very basic, assumes components are defined elsewhere)
    mock_config_session_test = {
        'experiment_name': 'mock_test_experiment',
        'simulation': {
            'num_episodes': 1,
            'max_steps_per_episode': 10,
            'dt': 1.0 # dt for simulator
        },
        'network': {'name': 'TestNetwork'}, # Config for network_builder_fn
        'simulator': {'dt': 1.0}, # Config for Simulator (dt matches simulation.dt)
        'io': {
            'input_encoder': {
                'type': 'MockEncoder', 
                'target_population_name': 'input_pop' # Crucial for session.setup
            },
            'output_decoder': {
                'type': 'MockDecoder', 
                'source_population_name': 'output_pop' # Crucial for session.setup
            }
        },
        'agent': {'snn_steps_per_action': 2},
        'probes': [
            # {'type': 'SpikeProbe', 'name': 'input_spikes', 'target': 'input_pop', 'params': {}},
            # {'type': 'VoltageProbe', 'name': 'output_voltage', 'target': 'output_pop', 'params': {'indices': [0,1]}}
        ],
        'output': {
            'directory': 'results/mock_runs',
            'format': 'json' # 'pickle' or 'json' or 'csv' for summary
        }
    }
    
    # Placeholder for actual imports and class definitions
    # class MockEnv: ...
    # def mock_env_creator(): ...
    # class MockNetwork: def __init__(self, config): self.name=config.get('name'); self.populations={'input_pop':MockPop('input_pop',10), 'output_pop':MockPop('output_pop',3)}; self.synapses={}; self.input_population_names=['input_pop']; self.output_population_names=['output_pop']; self.probes={}; def add_probe(self,p):self.probes[p.name]=p; def get_population(self,n):return self.populations[n]; def get_all_probes(self):return self.probes; def reset(self):pass
    # class MockPop: def __init__(self,name,size):self.name=name; self.size=size; 
    # def mock_network_builder(config): return MockNetwork(config)
    # class MockEncoder: def __init__(self,cfg,pop): print(f'MockEnc init for {pop.name}'); self.target_pop_name=pop.name; def encode(self,o,dt,ct): return {self.target_pop_name: [0]*pop.size}; def reset(self):pass
    # def mock_input_encoder_creator(cfg,pop): return MockEncoder(cfg,pop)
    # class MockDecoder: def __init__(self,cfg,pop): print(f'MockDec init for {pop.name}'); self.source_pop_name=pop.name; def decode(self,s,dt,ct): return 0; def reset(self):pass
    # def mock_output_decoder_creator(cfg,pop): return MockDecoder(cfg,pop)
    # from dynn.core.simulator import Simulator
    # from dynn.agent import Agent
    
    # try:
    #     print("\n--- Attempting Mock ExperimentSession Run ---")
    #     session = ExperimentSession(mock_config_session_test)
    #     session.setup(
    #         env_creator_fn=mock_env_creator, # Replace with actual or more detailed mocks
    #         network_builder_fn=mock_network_builder,
    #         input_encoder_creator_fn=mock_input_encoder_creator,
    #         output_decoder_creator_fn=mock_output_decoder_creator,
    #         # reward_processor_creator_fn=None, # Optional
    #         # agent_creator_fn=None, # Optional, will use default dynn.Agent
    #         # simulator_creator_fn=None # Optional, will use default dynn.core.Simulator
    #     )
    #     print("Setup complete.")
    #     # results = session.run()
    #     # print("Run complete. Results:", results)
    # except Exception as e:
    #     print(f"Error during mock session test: {e}")
    #     import traceback
    #     print(traceback.format_exc())
    print("\nMock test structure is present. Full execution requires dynn.core, utils, and other dependencies to be functional.") 