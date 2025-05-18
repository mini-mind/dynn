'''单元测试代码 for dynn.core.simulator'''
import unittest
from unittest.mock import MagicMock, call
import numpy as np

# 假设 dynn.core.network, dynn.core.neurons, dynn.utils.probes, dynn.core.simulator 路径在PYTHONPATH中
# 或者根据您的项目结构进行调整
from dynn.core.network import NeuralNetwork
from dynn.core.neurons import NeuronPopulation # 仅用于类型提示或模拟实例
from dynn.utils.probes import BaseProbe # 仅用于类型提示或模拟实例
from dynn.core.simulator import Simulator

class MockNeuronPopulation(MagicMock):
    def __init__(self, size, name="mock_pop", **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.name = name
        self._spikes = np.zeros(size, dtype=bool)
        self.v = np.zeros(size, dtype=float)
        self.u = np.zeros(size, dtype=float) # 如果神经元模型需要

    def update(self, dt, current_inputs):
        # 模拟更新逻辑，例如，随机产生一些脉冲
        self._spikes = np.random.rand(self.size) < 0.1 
        return self._spikes

    def reset(self):
        self._spikes = np.zeros(self.size, dtype=bool)
        self.v = np.zeros(self.size, dtype=float)
        self.u = np.zeros(self.size, dtype=float)

    def get_spikes(self):
        return self._spikes

class MockSynapseCollection(MagicMock):
    def __init__(self, pre_pop, post_pop, name="mock_syn", **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.pre_pop = pre_pop
        self.post_pop = post_pop
        self.weights = np.random.rand(pre_pop.size, post_pop.size)
        self.learning_rule = None

    def get_input_currents(self, pre_spikes):
        # 模拟电流计算
        return np.dot(pre_spikes.astype(float), self.weights.T) 

    def apply_learning_rule(self, pre_spikes, post_spikes, dt, reward=None):
        if self.learning_rule:
            self.learning_rule.update(pre_spikes, post_spikes, self.weights, dt, reward)
    
    def reset(self):
        pass # 假设权重不变

class MockLearningRule(MagicMock):
    def __init__(self, name="mock_lr", **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def update(self, pre_spikes, post_spikes, weights, dt, reward=None):
        # 模拟学习规则更新
        pass

    def reset(self):
        pass

class MockProbe(MagicMock):
    def __init__(self, target_obj, target_attr, name="mock_probe", **kwargs):
        # Initialize MagicMock part first, ensure 'name' is passed if used by MagicMock
        super().__init__(name=name, spec=BaseProbe, **kwargs) # Added spec for good measure
        
        self.target_obj = target_obj
        self.target_attr = target_attr
        self.data = {} 
        self.time_data = []
        if isinstance(target_attr, list):
            for attr in target_attr:
                self.data[attr] = []
        else:
            self.data[target_attr] = []
        
        # Crucially, self.name for the mock instance should be set if MagicMock doesn't handle it via constructor
        # For MagicMock, 'name' kwarg in super().__init__ should set its internal name if needed.
        # self.name = name # Already an attribute

        # Replace the 'record' method with a MagicMock that calls the original logic
        # The actual recording logic is moved to a private method _do_record
        self.record = MagicMock(side_effect=self._do_record_logic)
        # Also mock reset if its calls need to be tracked and it has logic
        self.reset = MagicMock(side_effect=self._do_reset_logic) # Changed from reset_data to reset

    def _do_record_logic(self, time):
        self.time_data.append(time)
        if isinstance(self.target_attr, list):
            for attr in self.target_attr:
                current_val = getattr(self.target_obj, attr, None)
                if current_val is not None:
                    self.data[attr].append(np.copy(current_val))
        else:
            current_val = getattr(self.target_obj, self.target_attr, None)
            if current_val is not None:
                self.data[self.target_attr].append(np.copy(current_val))
        return None # Explicit return for side_effect

    def _do_reset_logic(self):
        self.time_data = []
        if isinstance(self.target_attr, list):
            for attr in self.target_attr:
                self.data[attr] = []
        else:
            self.data[self.target_attr] = []
        return None # Explicit return

    def get_data(self):
        return self.data, self.time_data

class TestSimulator(unittest.TestCase):
    def setUp(self):
        # 创建模拟神经网络
        self.mock_network = MagicMock(spec=NeuralNetwork)
        
        # 配置模拟组件
        self.pop1 = MockNeuronPopulation(size=10, name="pop1")
        self.pop2 = MockNeuronPopulation(size=5, name="pop2")
        # self.syn12 = MockSynapseCollection(self.pop1, self.pop2, name="syn12") # Not directly used by simulator tests
        
        self.mock_network.populations = {"pop1": self.pop1, "pop2": self.pop2}
        # self.mock_network.synapses = {"syn12": self.syn12}
        self.mock_network.probes = {}
        self.mock_network.get_population = MagicMock(side_effect=lambda name: self.mock_network.populations.get(name))
        # self.mock_network.get_synapse_collection = MagicMock(side_effect=lambda name: self.mock_network.synapses.get(name))
        self.mock_network.get_probe = MagicMock(side_effect=lambda name: self.mock_network.probes.get(name))
        
        # For Simulator's default input generation in run_n_steps
        self.mock_network.input_population_names = ['pop1'] 

        self.mock_network.step = MagicMock(return_value={}) # Ensure step returns a dict
        self.mock_network.reset = MagicMock()

        self.dt = 0.001 # Define a consistent dt for tests
        self.simulator = Simulator(self.mock_network, dt=self.dt)

    def test_simulator_initialization(self):
        self.assertEqual(self.simulator.network, self.mock_network)
        self.assertEqual(self.simulator.dt, self.dt) 
        self.assertEqual(self.simulator.current_time, 0.0) # Simulator uses current_time

    def test_simulator_initialization_with_custom_dt(self):
        custom_dt = 0.005
        simulator = Simulator(self.mock_network, dt=custom_dt)
        self.assertEqual(simulator.dt, custom_dt)

    def test_run_for_duration_basic(self):
        duration = 0.1 # 100 ms
        expected_steps = int(round(duration / self.dt)) # Simulator uses round()

        self.simulator.run_for_duration(duration)

        self.assertEqual(self.mock_network.step.call_count, expected_steps)
        expected_default_inputs_pop1 = np.zeros(self.pop1.size)

        for i in range(expected_steps):
            call_args = self.mock_network.step.call_args_list[i]
            
            self.assertEqual(call_args.kwargs['dt'], self.dt)
            self.assertAlmostEqual(call_args.kwargs['time_elapsed'], i * self.dt, places=6)
            
            # Check default inputs created by Simulator.run_n_steps
            self.assertIn('pop1', call_args.kwargs['inputs'])
            np.testing.assert_array_equal(call_args.kwargs['inputs']['pop1'], expected_default_inputs_pop1)
            self.assertEqual(len(call_args.kwargs['inputs']), 1) # Only pop1 is an input pop
            
            self.assertIsNone(call_args.kwargs.get('reward'))
        
        self.assertAlmostEqual(self.simulator.current_time, duration, places=5) # Looser precision for final time

    def test_run_for_duration_with_dict_inputs(self):
        duration = 0.01
        num_steps = int(round(duration / self.dt))
        
        inputs_dict_sequence = {"pop1": np.random.rand(num_steps, self.pop1.size)}
        
        # nonlocal index for the generator
        current_step_idx = 0
        def dict_input_generator(current_time, dt_arg, prev_outputs):
            nonlocal current_step_idx
            if current_step_idx < num_steps:
                input_for_this_step = {"pop1": inputs_dict_sequence["pop1"][current_step_idx]}
                current_step_idx += 1
                return input_for_this_step
            return None # Or {} if network expects it for no input

        self.simulator.run_for_duration(total_duration=duration, input_generator_fn=dict_input_generator)
        
        self.assertEqual(self.mock_network.step.call_count, num_steps)
        for i in range(num_steps):
            call_args = self.mock_network.step.call_args_list[i]
            expected_input_for_step = {"pop1": inputs_dict_sequence["pop1"][i]}
            
            self.assertEqual(call_args.kwargs['inputs'].keys(), expected_input_for_step.keys())
            np.testing.assert_array_equal(call_args.kwargs['inputs']['pop1'], expected_input_for_step['pop1'])
            self.assertIsNone(call_args.kwargs.get('reward'))

    def test_run_for_duration_with_callable_inputs(self):
        duration = 0.01
        num_steps = int(round(duration / self.dt))

        def original_input_logic(t_elapsed):
            if 0.002 <= t_elapsed < 0.005:
                return {"pop1": np.ones(self.pop1.size)}
            return None # Or {}

        def adapted_input_generator(current_time, dt_arg, prev_outputs):
            return original_input_logic(current_time)

        self.simulator.run_for_duration(total_duration=duration, input_generator_fn=adapted_input_generator)

        self.assertEqual(self.mock_network.step.call_count, num_steps)
        for i in range(num_steps):
            current_sim_time = i * self.dt
            call_args = self.mock_network.step.call_args_list[i]
            expected_input = original_input_logic(current_sim_time)
            
            if expected_input is None:
                 self.assertIsNone(call_args.kwargs['inputs'])
            else:
                self.assertIsNotNone(call_args.kwargs['inputs']) # Ensure inputs dict exists
                self.assertEqual(call_args.kwargs['inputs'].keys(), expected_input.keys())
                for key in expected_input.keys():
                    np.testing.assert_array_equal(call_args.kwargs['inputs'][key], expected_input[key])
            self.assertIsNone(call_args.kwargs.get('reward'))

    def test_run_for_duration_verifies_reward_is_none(self):
        duration = 0.01
        num_steps = int(round(duration / self.dt))

        # This test now primarily confirms that reward is not unexpectedly passed
        self.simulator.run_for_duration(total_duration=duration) 
        
        self.assertEqual(self.mock_network.step.call_count, num_steps)
        for i in range(num_steps):
            call_args = self.mock_network.step.call_args_list[i]
            self.assertIsNone(call_args.kwargs.get('reward'))
            
    def test_run_for_duration_with_probes(self):
        duration = 0.003 
        num_steps = int(round(duration / self.dt))
        
        mock_probe_pop1_v = MockProbe(self.pop1, 'v', name="probe_pop1_v")
        self.mock_network.probes = {"probe_pop1_v": mock_probe_pop1_v}
        
        # Side effect for network.step; its signature is step(dt, time_elapsed, inputs=None, reward=None)
        def step_side_effect_for_probe_test(dt, time_elapsed, inputs=None, reward=None):
            # Simulate neuron voltage change for probe to record
            self.pop1.v += time_elapsed # Example: v increases with time_elapsed
            # Simulate network's responsibility to call probe.record
            # In a real network, this would be part of network.step's logic after component updates
            if "probe_pop1_v" in self.mock_network.probes:
                 self.mock_network.probes["probe_pop1_v"].record(time_elapsed) # Probe records at current time_elapsed
            return {} # network.step should return outputs dict

        self.mock_network.step.side_effect = step_side_effect_for_probe_test
        
        self.simulator.run_for_duration(total_duration=duration)

        # Check probe's record method calls (indirectly via network.step's side_effect)
        self.assertEqual(mock_probe_pop1_v.record.call_count, num_steps)
        
        expected_times = [i * self.dt for i in range(num_steps)]
        for i, expected_time in enumerate(expected_times):
             # probe.record is called with time_elapsed of the step
             self.assertAlmostEqual(mock_probe_pop1_v.record.call_args_list[i][0][0], expected_time, places=6)

    def test_reset_simulator(self):
        self.simulator.run_for_duration(0.01)
        self.assertNotEqual(self.simulator.current_time, 0.0)
        
        self.simulator.reset()
        
        self.assertEqual(self.simulator.current_time, 0.0)
        self.mock_network.reset.assert_called_once()

    def test_run_for_duration_with_zero_duration(self):
        self.simulator.run_for_duration(0.0)
        self.mock_network.step.assert_not_called() # run_for_duration should return early
        self.assertEqual(self.simulator.current_time, 0.0)

    def test_run_for_duration_with_negative_duration(self):
        self.simulator.run_for_duration(-0.01)
        self.mock_network.step.assert_not_called() # run_for_duration should return early
        self.assertEqual(self.simulator.current_time, 0.0)
        
if __name__ == '__main__':
    unittest.main() 