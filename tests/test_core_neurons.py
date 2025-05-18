# Placeholder for tests of core.neurons module 

'''单元测试代码 for dynn.core.neurons'''
import unittest
import numpy as np
from dynn.core.neurons import IzhikevichNeuron, NeuronPopulation

class TestIzhikevichNeuron(unittest.TestCase):
    def test_neuron_initialization_default(self):
        neuron = IzhikevichNeuron()
        self.assertEqual(neuron.a, 0.02)
        self.assertEqual(neuron.b, 0.2)
        self.assertEqual(neuron.c, -65.0)
        self.assertEqual(neuron.d, 8.0)
        self.assertEqual(neuron.v_thresh, 30.0)
        self.assertEqual(neuron.v, -70.0)
        self.assertEqual(neuron.u, 0.2 * -70.0) # b * initial_v
        self.assertFalse(neuron.fired)
        self.assertEqual(neuron.last_spike_time, -np.inf)

    def test_neuron_initialization_custom(self):
        neuron = IzhikevichNeuron(a=0.03, b=0.25, c=-55.0, d=2.0, v_thresh=25.0, initial_v=-60.0, initial_u=-10.0)
        self.assertEqual(neuron.a, 0.03)
        self.assertEqual(neuron.b, 0.25)
        self.assertEqual(neuron.c, -55.0)
        self.assertEqual(neuron.d, 2.0)
        self.assertEqual(neuron.v_thresh, 25.0)
        self.assertEqual(neuron.v, -60.0)
        self.assertEqual(neuron.u, -10.0)

    def test_neuron_update_no_spike(self):
        neuron = IzhikevichNeuron(initial_v=-70.0)
        # 确保在阈值以下时神经元不发放脉冲
        fired = neuron.update(I_inj=0, dt=1.0)
        self.assertFalse(fired)
        self.assertLess(neuron.v, neuron.v_thresh)

    def test_neuron_update_generates_spike(self):
        # 使用 Regular Spiking (RS) 参数
        neuron = IzhikevichNeuron(a=0.02, b=0.2, c=-65.0, d=8.0, initial_v=-70.0)
        fired = False
        # 施加足够的电流以引发脉冲
        for _ in range(100): # 模拟一些时间步
            if neuron.update(I_inj=10.0, dt=1.0):
                fired = True
                break
        self.assertTrue(fired, "神经元在施加电流后应发放脉冲")
        # 脉冲后，v应重置为c
        self.assertEqual(neuron.v, neuron.c)
        # u应增加d
        # 注意: u的精确值取决于脉冲前的u值和d，这里仅检查v的重置
        # self.assertAlmostEqual(neuron.u, expected_u_after_spike) 

    def test_neuron_reset_state(self):
        neuron = IzhikevichNeuron(initial_v=-60.0, initial_u=-15.0)
        neuron.v = 20.0 # 改变状态
        neuron.u = 5.0
        neuron.fired = True
        neuron.last_spike_time = 10.0
        
        neuron.reset(initial_v=-65.0, initial_u=-12.0)
        self.assertEqual(neuron.v, -65.0)
        self.assertEqual(neuron.u, -12.0)
        self.assertFalse(neuron.fired)
        self.assertEqual(neuron.last_spike_time, -np.inf)

    def test_neuron_reset_state_default_u(self):
        neuron = IzhikevichNeuron(b=0.25)
        neuron.reset(initial_v=-60.0)
        self.assertEqual(neuron.v, -60.0)
        self.assertEqual(neuron.u, 0.25 * -60.0)

    def test_neuron_get_state(self):
        neuron = IzhikevichNeuron(initial_v=-65.0, initial_u=-10.0)
        neuron.last_spike_time = 50.0
        neuron.fired = True # 假设刚发放过
        state = neuron.get_state()
        self.assertEqual(state["v"], -65.0)
        self.assertEqual(state["u"], -10.0)
        self.assertTrue(state["fired"])
        self.assertEqual(state["last_spike_time"], 50.0)

class TestNeuronPopulation(unittest.TestCase):
    def test_population_initialization_default(self):
        pop = NeuronPopulation(num_neurons=10)
        self.assertEqual(pop.num_neurons, 10)
        self.assertEqual(len(pop.neurons), 10)
        self.assertIsInstance(pop.neurons[0], IzhikevichNeuron)
        # 检查所有神经元是否有默认的初始v和u
        for neuron in pop.neurons:
            self.assertEqual(neuron.v, -70.0)
            # IzhikevichNeuron 默认 b=0.2
            self.assertEqual(neuron.u, 0.2 * -70.0) 

    def test_population_initialization_custom_params_shared(self):
        params = {"a": 0.03, "c": -50.0, "initial_v": -60.0}
        pop = NeuronPopulation(num_neurons=5, neuron_params=params)
        self.assertEqual(len(pop.neurons), 5)
        for neuron in pop.neurons:
            self.assertEqual(neuron.a, 0.03)
            self.assertEqual(neuron.c, -50.0)
            self.assertEqual(neuron.v, -60.0)
            # initial_u 会根据 neuron.b 和 initial_v 计算
            # IzhikevichNeuron 默认 b=0.2
            self.assertEqual(neuron.u, 0.2 * -60.0)

    def test_population_initialization_custom_params_individual(self):
        params_list = [
            {"a": 0.02, "initial_v": -70.0},
            {"a": 0.03, "initial_v": -65.0, "initial_u": -10.0}
        ]
        pop = NeuronPopulation(num_neurons=2, neuron_params=params_list)
        self.assertEqual(pop.neurons[0].a, 0.02)
        self.assertEqual(pop.neurons[0].v, -70.0)
        self.assertEqual(pop.neurons[0].u, pop.neurons[0].b * -70.0) # 默认b

        self.assertEqual(pop.neurons[1].a, 0.03)
        self.assertEqual(pop.neurons[1].v, -65.0)
        self.assertEqual(pop.neurons[1].u, -10.0)

    def test_population_initialization_initial_v_fixed(self):
        pop = NeuronPopulation(num_neurons=3, initial_v_dist=-65.0)
        for neuron in pop.neurons:
            self.assertEqual(neuron.v, -65.0)

    def test_population_initialization_initial_v_uniform(self):
        pop = NeuronPopulation(num_neurons=100, initial_v_dist=('uniform', (-70.0, -60.0)))
        all_v = np.array([n.v for n in pop.neurons])
        self.assertTrue(np.all(all_v >= -70.0) and np.all(all_v <= -60.0))
        self.assertGreater(np.std(all_v), 0) # 应有变异

    def test_population_initialization_initial_v_normal(self):
        mean, std = -65.0, 5.0
        pop = NeuronPopulation(num_neurons=100, initial_v_dist=('normal', (mean, std)))
        all_v = np.array([n.v for n in pop.neurons])
        # 简单的统计检查，不期望完全精确
        self.assertAlmostEqual(np.mean(all_v), mean, delta=1.0) 
        self.assertAlmostEqual(np.std(all_v), std, delta=1.0)

    def test_population_initialization_initial_u_fixed(self):
        pop = NeuronPopulation(num_neurons=3, initial_u_dist=-15.0)
        for neuron in pop.neurons:
            self.assertEqual(neuron.u, -15.0)
            # v 应该还是默认的 -70
            self.assertEqual(neuron.v, -70.0)

    def test_population_initialization_initial_u_from_b_and_v_in_params(self):
        # 测试当 initial_u 未指定，但 initial_v 和 b 在 neuron_params 中指定时的情况
        neuron_params = {"b": 0.25, "initial_v": -60.0}
        pop = NeuronPopulation(num_neurons=1, neuron_params=neuron_params)
        self.assertEqual(pop.neurons[0].v, -60.0)
        self.assertEqual(pop.neurons[0].b, 0.25)
        self.assertEqual(pop.neurons[0].u, 0.25 * -60.0)

    def test_population_initialization_initial_u_from_default_b_and_custom_v_dist(self):
        # 测试当 initial_u 未指定，initial_v 通过 dist 指定，b 使用默认值
        pop = NeuronPopulation(num_neurons=1, initial_v_dist=-60.0) # neuron_params 为空
        default_b = IzhikevichNeuron().b # 获取 IzhikevichNeuron 的默认 b
        self.assertEqual(pop.neurons[0].v, -60.0)
        self.assertEqual(pop.neurons[0].u, default_b * -60.0)

    def test_population_update_and_get_spikes(self):
        pop = NeuronPopulation(num_neurons=3, neuron_params={"v_thresh": 20.0, "c": -50.0})
        # 让第一个神经元发放脉冲
        pop.neurons[0].v = 25.0 # 超出阈值
        pop.neurons[1].v = 10.0 # 未超阈值
        pop.neurons[2].v = 22.0 # 超出阈值

        I_inj = np.array([0.0, 5.0, 0.0])
        current_time = 10.0
        fired_mask = pop.update(I_inj, dt=1.0, current_time=current_time)

        expected_fired_mask = np.array([True, False, True])
        np.testing.assert_array_equal(fired_mask, expected_fired_mask)
        
        spikes = pop.get_spikes()
        np.testing.assert_array_equal(spikes, expected_fired_mask)

        # 检查脉冲发放神经元的状态和 last_spike_time
        self.assertEqual(pop.neurons[0].v, -50.0) # c
        self.assertEqual(pop.neurons[0].last_spike_time, current_time)
        self.assertTrue(pop.neurons[0].fired)

        self.assertNotEqual(pop.neurons[1].v, -50.0) # c
        self.assertNotEqual(pop.neurons[1].last_spike_time, current_time)
        self.assertFalse(pop.neurons[1].fired)
        
        self.assertEqual(pop.neurons[2].v, -50.0) # c
        self.assertEqual(pop.neurons[2].last_spike_time, current_time)
        self.assertTrue(pop.neurons[2].fired)


    def test_get_all_states(self):
        params_list = [
            {"initial_v": -70.0, "initial_u": -14.0},
            {"initial_v": -60.0, "initial_u": -12.0}
        ]
        pop = NeuronPopulation(num_neurons=2, neuron_params=params_list)
        pop.neurons[0].fired = True # 手动设置一个发放脉冲
        pop.neurons[0].last_spike_time = 5.0

        states = pop.get_all_states()
        self.assertIn("v", states)
        self.assertIn("u", states)
        self.assertIn("fired", states)
        self.assertIn("last_spike_time", states)

        np.testing.assert_array_equal(states["v"], np.array([-70.0, -60.0]))
        np.testing.assert_array_equal(states["u"], np.array([-14.0, -12.0]))
        np.testing.assert_array_equal(states["fired"], np.array([True, False]))
        np.testing.assert_array_equal(states["last_spike_time"], np.array([5.0, -np.inf]))

    def test_get_all_states_specific_keys(self):
        pop = NeuronPopulation(num_neurons=2)
        states = pop.get_all_states(state_keys=["v", "fired"])
        self.assertIn("v", states)
        self.assertNotIn("u", states)
        self.assertIn("fired", states)
        self.assertEqual(len(states["v"]), 2)
        self.assertEqual(len(states["fired"]), 2)

    def test_set_parameters_single_neuron_single_value(self):
        pop = NeuronPopulation(num_neurons=3)
        pop.set_parameters(neuron_indices=1, param_name="a", param_value=0.05)
        self.assertEqual(pop.neurons[1].a, 0.05)
        self.assertNotEqual(pop.neurons[0].a, 0.05) # 其他神经元不受影响

    def test_set_parameters_multiple_neurons_single_value(self):
        pop = NeuronPopulation(num_neurons=3)
        pop.set_parameters(neuron_indices=[0, 2], param_name="c", param_value=-55.0)
        self.assertEqual(pop.neurons[0].c, -55.0)
        self.assertEqual(pop.neurons[2].c, -55.0)
        self.assertNotEqual(pop.neurons[1].c, -55.0)

    def test_set_parameters_multiple_neurons_multiple_values(self):
        pop = NeuronPopulation(num_neurons=3)
        new_d_values = [2.0, 4.0]
        pop.set_parameters(neuron_indices=[0, 1], param_name="d", param_value=new_d_values)
        self.assertEqual(pop.neurons[0].d, 2.0)
        self.assertEqual(pop.neurons[1].d, 4.0)
        self.assertNotEqual(pop.neurons[2].d, 2.0) # IzhikevichNeuron 默认 d=8.0

    def test_set_parameters_invalid_length(self):
        pop = NeuronPopulation(num_neurons=3)
        with self.assertRaises(ValueError):
            pop.set_parameters(neuron_indices=[0, 1], param_name="b", param_value=[0.1, 0.2, 0.3])

    def test_population_reset_states(self):
        pop = NeuronPopulation(num_neurons=2)
        pop.neurons[0].v = 10.0
        pop.neurons[0].u = 1.0
        pop.neurons[0].fired = True
        pop.neurons[0].last_spike_time = 1.0
        pop.neurons[1].v = 20.0

        pop.reset_states(initial_v_dist=-68.0, initial_u_dist=('uniform', (-15.0, -13.0)))
        
        self.assertEqual(pop.neurons[0].v, -68.0)
        self.assertTrue(-15.0 <= pop.neurons[0].u <= -13.0)
        self.assertFalse(pop.neurons[0].fired)
        self.assertEqual(pop.neurons[0].last_spike_time, -np.inf)
        
        self.assertEqual(pop.neurons[1].v, -68.0)
        self.assertTrue(-15.0 <= pop.neurons[1].u <= -13.0)

    def test_len_and_getitem(self):
        pop = NeuronPopulation(num_neurons=5)
        self.assertEqual(len(pop), 5)
        self.assertIsInstance(pop[0], IzhikevichNeuron)
        self.assertIsInstance(pop[4], IzhikevichNeuron)
        with self.assertRaises(IndexError):
            _ = pop[5]

if __name__ == '__main__':
    unittest.main() 