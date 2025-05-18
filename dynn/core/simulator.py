# Placeholder for the simulation engine 

import time
import numpy as np
# Assuming NeuralNetwork is in .network, NeuronPopulation in .neurons, etc.
# from .network import NeuralNetwork # This will be resolved when the package is used

class Simulator:
    """
    离散时间仿真引擎。
    控制仿真循环，协调神经元状态更新、突触信号传播和学习规则的应用。
    """
    def __init__(self, network, dt=1.0):
        """
        初始化仿真器。

        参数:
            network (NeuralNetwork): 要仿真的神经网络实例。
            dt (float): 仿真时间步长 (单位通常是毫秒)。
        """
        self.network = network
        self.dt = dt
        self.current_time = 0.0
        self.total_steps_run = 0

    def run_step(self, input_currents_map):
        """
        执行单个仿真步骤。

        参数:
            input_currents_map (dict): 
                一个字典，键是输入神经元群体的名称，值是注入该群体的电流向量 (np.array)。
                例如: {"input_pop1": np.array([...]), "input_pop2": np.array([...])}
        
        返回:
            dict: 网络输出的脉冲映射。
        """
        outputs = self.network.step(dt=self.dt, time_elapsed=self.current_time, inputs=input_currents_map)
        self.current_time += self.dt
        self.total_steps_run += 1
        return outputs

    def run_n_steps(self, num_steps, input_generator_fn=None, callback_fn=None, stop_condition_fn=None):
        """
        执行指定数量的仿真步骤。

        参数:
            num_steps (int): 要执行的仿真步数。
            input_generator_fn (callable, optional): 
                一个函数，签名应为 `fn(current_time, dt, previous_step_outputs)`，
                返回一个 `input_currents_map` 字典用于当前步骤。
                `previous_step_outputs` 是上一步 `network.step` 的返回结果。
            callback_fn (callable, optional):
                一个函数，签名应为 `fn(current_time, current_inputs, current_outputs, simulator_instance)`，
                在每个仿真步骤后被调用。可用于记录、与环境交互等。
            stop_condition_fn (callable, optional):
                一个函数，签名应为 `fn(current_time, current_outputs, simulator_instance)`，
                在每个步骤后检查，如果返回True，则仿真提前停止。

        返回:
            dict or None: 最后一个仿真步骤的网络输出，如果仿真未运行则为None。
        """
        last_outputs = None
        for i in range(num_steps):
            if input_generator_fn:
                current_inputs = input_generator_fn(self.current_time, self.dt, last_outputs)
            else:
                # 如果没有提供输入生成器，则假设网络中的输入群体没有外部电流注入
                current_inputs = {name: np.zeros(self.network.get_population(name).size) 
                                  for name in self.network.input_population_names}
            
            last_outputs = self.run_step(current_inputs)

            if callback_fn:
                callback_fn(self.current_time, current_inputs, last_outputs, self)
            
            if stop_condition_fn and stop_condition_fn(self.current_time, last_outputs, self):
                print(f"仿真在时间 {self.current_time:.2f}ms (步骤 {i+1}/{num_steps}) 因满足停止条件而提前结束。")
                break
        return last_outputs

    def run_for_duration(self, total_duration, input_generator_fn=None, callback_fn=None, stop_condition_fn=None):
        """
        执行仿真直到达到指定的总时长。

        参数:
            total_duration (float): 总仿真时长 (与dt单位相同，通常是毫秒)。
            input_generator_fn, callback_fn, stop_condition_fn: 同 run_n_steps。
        
        返回:
            dict or None: 最后一个仿真步骤的网络输出。
        """
        if total_duration <= 0:
            print("警告: 总仿真时长必须为正。")
            return None
        if self.dt <= 0:
            raise ValueError("仿真时间步长 dt 必须为正。")
            
        num_steps = int(round(total_duration / self.dt))
        if num_steps == 0 and total_duration > 0: # 如果总时长小于一个dt，至少运行一步
             print(f"警告: 总时长 {total_duration} 小于时间步长 {self.dt}。将至少尝试运行与当前时间匹配的若干步直到总时长。")
        
        # run_n_steps 将处理 self.current_time 的增加
        # 我们需要确保它从当前时间点开始，运行额外的 num_steps
        # 或者，更简单地，假设 run_for_duration 是从0开始或从当前时间点开始，持续一个指定的时长。
        # 如果是从当前时间点继续，那么 num_steps 的计算是正确的。
        # 如果是重置并运行，则应先调用 reset()。
        
        print(f"计划运行仿真 {num_steps} 步 (总时长: {total_duration}, dt: {self.dt})...")
        return self.run_n_steps(num_steps, input_generator_fn, callback_fn, stop_condition_fn)

    def reset(self, reset_time=True):
        """
        重置仿真器和关联的网络。
        
        参数:
            reset_time (bool): 是否将当前仿真时间重置为0。默认为True。
        """
        self.network.reset() # 重置网络状态 (神经元、学习规则、探针等)
        if reset_time:
            self.current_time = 0.0
        self.total_steps_run = 0
        print("仿真器已重置。")

    def __repr__(self):
        return f"Simulator(network='{self.network.name}', dt={self.dt}, current_time={self.current_time:.2f}ms, total_steps={self.total_steps_run})" 