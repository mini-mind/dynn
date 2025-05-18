class Agent:
    """
    封装SNN决策和学习核心的Agent。

    该Agent协调输入编码、SNN仿真、输出解码和学习过程。

    Attributes:
        network (dynn.core.NeuralNetwork): SNN模型。
        simulator (dynn.core.Simulator): 驱动SNN的模拟器。
        input_encoder (dynn.io.BaseInputEncoder): 将观测数据转换为SNN输入的编码器。
        output_decoder (dynn.io.BaseOutputDecoder): 将SNN输出转换为动作的解码器。
        reward_processor (dynn.io.BaseRewardProcessor, optional): 在学习前处理原始奖励的处理器。
        agent_config (dict): Agent特定的配置，例如每个环境动作对应的SNN步数。
        snn_steps_per_action (int): 每个环境动作对应的SNN仿真步数。
    """
    def __init__(self, network, simulator, input_encoder, output_decoder, reward_processor=None, agent_config=None):
        """
        初始化Agent。

        Args:
            network (dynn.core.NeuralNetwork): 要使用的SNN模型。
            simulator (dynn.core.Simulator): 用于驱动SNN的模拟器。
            input_encoder (dynn.io.BaseInputEncoder): 将观测数据转换为SNN输入的编码器。
            output_decoder (dynn.io.BaseOutputDecoder): 将SNN输出转换为动作的解码器。
            reward_processor (dynn.io.BaseRewardProcessor, optional): 用于处理奖励信号的处理器。
                默认为 None。
            agent_config (dict, optional): Agent特定的配置字典。默认为 None。
        """
        self.network = network
        self.simulator = simulator
        self.input_encoder = input_encoder
        self.output_decoder = output_decoder
        self.reward_processor = reward_processor
        self.agent_config = agent_config if agent_config is not None else {}

        # Agent特定配置的示例：每个环境动作对应的SNN步数
        self.snn_steps_per_action = self.agent_config.get('snn_steps_per_action', 1)

    def get_action(self, observation):
        """
        根据给定的观察值获取Agent的动作。

        处理流程:
        1. 使用 `self.input_encoder` 对观察值进行编码。
        2. 运行 `self.simulator` 执行 `self.snn_steps_per_action` 数量的SNN步。
        3. 使用 `self.output_decoder` 解码SNN的输出脉冲以获得动作。

        Args:
            observation (any): 来自环境的当前观察值。其类型和结构取决于环境和输入编码器。

        Returns:
            any: Agent选择的动作。其类型和结构取决于输出解码器。
        """
        # 1. 通过 self.input_encoder 编码 observation
        # input_encoder.encode() 应返回一个 input_currents_map 格式的字典
        # 例如: {'input_pop': np.array([...])}
        snn_input_map = self.input_encoder.encode(observation, 
                                                  dt=self.simulator.dt, 
                                                  current_time=self.simulator.current_time)

        # 2. 运行 self.simulator 执行配置中指定数量的SNN步
        # 这里的输入 snn_input_map 会在每个SNN步骤中施加。
        # 如果输入只应在第一步施加，则需要更复杂的逻辑或 input_generator_fn。
        # 为简单起见，假设输入在所有 snn_steps_per_action 中保持。
        
        last_snn_outputs = None
        for _ in range(self.snn_steps_per_action):
            # simulator.run_step() 内部调用 network.step()，后者应用学习规则并返回输出脉冲
            last_snn_outputs = self.simulator.run_step(input_currents_map=snn_input_map)
            # 注意：学习规则（如STDP）的权重更新是在 network.step() 内部，
            # 如果有 reward_modulation，它会影响这些更新。

        # 3. 通过 self.output_decoder 解码SNN的输出脉冲以获得动作
        # output_decoder.decode() 期望 spike_activities_map，这与 simulator.run_step() 的返回值匹配。
        # 如果 self.snn_steps_per_action 为0或 simulator 未运行，last_snn_outputs可能为None。
        if last_snn_outputs is None:
            # 处理没有SNN输出的情况，可能返回默认动作或引发错误
            # 根据 output_decoder 的实现，它可能能处理 None 输入
            print("警告: Agent.get_action 未从 SNN 获得输出。")
            # 尝试用空的spike_activities_map调用解码器，或者让解码器处理None
            action = self.output_decoder.decode({}, 
                                                dt=self.simulator.dt, 
                                                current_time=self.simulator.current_time)
        else:
            action = self.output_decoder.decode(last_snn_outputs, 
                                                dt=self.simulator.dt, 
                                                current_time=self.simulator.current_time)
        return action

    def learn(self, observation, action, reward, next_observation, done):
        """
        Agent根据经验进行学习。

        主要步骤:
        1. 如果存在 `self.reward_processor`，则使用它处理原始奖励以获得调制信号。
        2. 遍历 `self.network.synapses`，并为每个突触的学习规则设置奖励调制。
           实际的权重更新发生在后续的SNN仿真步骤中。

        Args:
            observation (any): 导致采取动作的观察值。
            action (any): Agent执行的动作。
            reward (float): 执行动作后从环境中获得的原始奖励。
            next_observation (any): 执行动作后得到的新观察值。
            done (bool): 一个标志，指示当前轮次是否结束。
        """
        processed_reward_signal = reward
        # 1. （如果存在 self.reward_processor）处理 reward
        if self.reward_processor:
            processed_reward_signal = self.reward_processor.process(
                reward=reward, 
                observation=observation, 
                action=action, 
                next_observation=next_observation, 
                done=done,
                # 可以传递 dt 和 current_time 如果 reward_processor 需要
                dt=self.simulator.dt, 
                current_time=self.simulator.current_time 
            )

        # 2. 遍历 self.network.synapses 并对每个突触的学习规则调用
        #    learning_rule.set_reward_modulation(processed_reward_signal)
        # 实际的权重更新发生在后续的仿真步骤中 (即 network.step() -> syn_collection.apply_learning_rule())
        if hasattr(self.network, 'synapses') and self.network.synapses:
            # network.synapses 是一个字典 {name: SynapseCollection}
            for synapse_collection in self.network.synapses.values(): 
                 if hasattr(synapse_collection, 'learning_rule') and \
                    synapse_collection.learning_rule is not None and \
                    hasattr(synapse_collection.learning_rule, 'set_reward_modulation'):
                    try:
                        synapse_collection.learning_rule.set_reward_modulation(processed_reward_signal)
                    except Exception as e:
                        print(f"错误：为突触 {synapse_collection.name} 的学习规则设置奖励调制时出错: {e}")
        
        # 注意：此 learn 方法仅设置奖励调制。权重更新发生在下一个 self.simulator.run_step() 期间，
        # 当 self.network.step() 调用 self.synapses[...].apply_learning_rule() 时。
        # 这符合设计文档中 "实际的权重更新发生在后续的仿真步骤中" 的描述。

    def reset(self):
        """
        重置Agent的内部状态。

        这通常涉及重置奖励处理器和IO组件（输入编码器、输出解码器）的内部状态。
        Agent本身通常不负责重置模拟器或网络，这些由 `ExperimentSession` 管理。
        """
        # 重置内部状态
        if self.reward_processor and hasattr(self.reward_processor, 'reset'):
            self.reward_processor.reset()
        
        # Agent 本身通常不需要重置 simulator 或 network，这些由 ExperimentSession 管理。
        # 但 Agent 可能需要重置其 IO 组件的状态。
        if hasattr(self.input_encoder, 'reset'):
            self.input_encoder.reset()
        if hasattr(self.output_decoder, 'reset'):
            self.output_decoder.reset()
        pass 