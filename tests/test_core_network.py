'''单元测试代码 for dynn.core.network'''
import unittest
import numpy as np

from dynn.core.neurons import NeuronPopulation, IzhikevichNeuron
from dynn.core.synapses import SynapseCollection
from dynn.core.learning_rules import BaseLearningRule, TraceSTDP
from dynn.core.network import NeuralNetwork
from dynn.utils.probes import BaseProbe # 假设 BaseProbe 存在且可导入

# --- Mocking/Helper Classes ---
class MockNeuronPopulation(NeuronPopulation):
    def __init__(self, num_neurons, name="mock_pop"):
        super().__init__(num_neurons, neuron_model_class=IzhikevichNeuron, name=name)
        self.update_called_with = None
        self.reset_called = False
        self._spikes = np.zeros(num_neurons, dtype=bool)

    def update(self, I_inj_vector, dt, current_time):
        self.update_called_with = (I_inj_vector, dt, current_time)
        # Simulate some spikes for testing step logic
        if np.sum(I_inj_vector) > 0: # Spike if any current
            self._spikes = np.random.rand(self.num_neurons) > 0.5 
        else:
            self._spikes.fill(False)
        self.fired = self._spikes # NeuronPopulation itself doesn't have .fired
        for i, neuron in enumerate(self.neurons):
            neuron.fired = self._spikes[i]
            if self._spikes[i]:
                neuron.last_spike_time = current_time
        return self._spikes

    def get_spikes(self):
        return self._spikes

    def reset_states(self, initial_v_dist=None, initial_u_dist=None):
        super().reset_states(initial_v_dist, initial_u_dist)
        self.reset_called = True
        self._spikes.fill(False)

class MockSynapseCollection(SynapseCollection):
    def __init__(self, pre_pop, post_pop, name="mock_syn"):
        super().__init__(pre_pop, post_pop, name=name)
        self.get_input_currents_called_with = None
        self.apply_learning_rule_called = False
        self._mock_currents = np.zeros(len(post_pop))

    def get_input_currents(self, pre_spikes):
        self.get_input_currents_called_with = pre_spikes
        # Simulate some current generation
        self._mock_currents = np.ones(len(self.post_pop)) * np.sum(pre_spikes) * 0.1
        return self._mock_currents

    def apply_learning_rule(self, pre_spikes, post_spikes, dt, current_time):
        self.apply_learning_rule_called = True
        if self.learning_rule:
            super().apply_learning_rule(pre_spikes, post_spikes, dt, current_time)

class MockLearningRule(BaseLearningRule):
    def __init__(self, lr_ltp=0.001, lr_ltd=0.001):
        super().__init__(lr_ltp, lr_ltd)
        self.update_weights_called_with = None
        self.reset_called = False

    def update_weights(self, synapse_collection, pre_spikes, post_spikes, dt, current_time):
        self.update_weights_called_with = (synapse_collection.name, pre_spikes.shape, post_spikes.shape, dt, current_time)
    
    def reset(self):
        self.reset_called = True

class MockProbe(BaseProbe):
    def __init__(self, name="mock_probe", record_interval=1, tracked_keys=None):
        super().__init__(name, record_interval)
        self.collect_called_with = None
        self.attempt_record_called = False # Kept for direct check if needed
        self.reset_called = False
        self.tracked_keys = tracked_keys if tracked_keys else ["mock_value"]
        for key in self.tracked_keys:
            self.data[key] = [] # Initialize data storage as per BaseProbe

    def _collect_data(self, network, current_time_ms):
        self.collect_called_with = (network.name, current_time_ms)
        # self.time_data.append(current_time_ms) # This is done by BaseProbe.attempt_record
        for key in self.tracked_keys:
            # For simplicity, store a placeholder or a random value
            if key == "mock_value":
                 self.data[key].append(np.random.rand())
            # If other keys are specified, tests might need to provide specific mock data collection
        return True 
    
    def reset(self):
        super().reset() # This clears self.data lists and self.time_data
        self.reset_called = True
        # Re-initialize data keys if super().reset() clears them completely (it should empty the lists)
        # for key in self.tracked_keys:
        #     if key not in self.data:
        #         self.data[key] = []

# --- End Mocking/Helper Classes ---

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.net = NeuralNetwork(name="TestNet")
        self.pop1 = MockNeuronPopulation(num_neurons=10, name="pop1")
        self.pop2 = MockNeuronPopulation(num_neurons=5, name="pop2")
        self.pop_output = MockNeuronPopulation(num_neurons=3, name="pop_out")

    def test_initialization(self):
        self.assertEqual(self.net.name, "TestNet")
        self.assertEqual(self.net.populations, {})
        self.assertEqual(self.net.synapses, {})
        self.assertEqual(self.net.input_population_names, [])
        self.assertEqual(self.net.output_population_names, [])
        self.assertEqual(self.net.probes, [])

    def test_add_get_population(self):
        self.net.add_population(self.pop1)
        self.assertIn("pop1", self.net.populations)
        self.assertIs(self.net.get_population("pop1"), self.pop1)
        with self.assertRaises(ValueError): # Duplicate name
            self.net.add_population(MockNeuronPopulation(3, name="pop1"))
        with self.assertRaises(KeyError): # Get non-existent
            self.net.get_population("non_existent_pop")

    def test_add_get_synapses(self):
        self.net.add_population(self.pop1)
        self.net.add_population(self.pop2)
        syn12 = MockSynapseCollection(self.pop1, self.pop2, name="syn12")
        self.net.add_synapses(syn12)
        self.assertIn("syn12", self.net.synapses)
        self.assertIs(self.net.get_synapses("syn12"), syn12)

        with self.assertRaises(ValueError): # Duplicate name
            self.net.add_synapses(MockSynapseCollection(self.pop1, self.pop2, name="syn12"))
        with self.assertRaises(KeyError): # Get non-existent
            self.net.get_synapses("non_existent_syn")

    def test_add_synapses_unregistered_populations(self):
        pop_unreg = MockNeuronPopulation(2, name="unreg")
        # Pre-pop not registered
        with self.assertRaises(ValueError):
            self.net.add_synapses(MockSynapseCollection(pop_unreg, self.pop1, name="s_unreg_pre"))
        # Post-pop not registered
        self.net.add_population(self.pop1)
        with self.assertRaises(ValueError):
            self.net.add_synapses(MockSynapseCollection(self.pop1, pop_unreg, name="s_unreg_post"))

    def test_set_input_output_populations(self):
        self.net.add_population(self.pop1)
        self.net.add_population(self.pop2)
        
        self.net.set_input_populations(["pop1"])
        self.assertEqual(self.net.input_population_names, ["pop1"])
        self.net.set_output_populations(["pop2", "pop1"])
        self.assertCountEqual(self.net.output_population_names, ["pop1", "pop2"]) # Order may vary due to set

        with self.assertRaises(ValueError): # Non-existent input pop
            self.net.set_input_populations(["non_existent"])
        with self.assertRaises(ValueError): # Non-existent output pop
            self.net.set_output_populations(["non_existent"])

    def test_add_get_probes(self):
        probe1 = MockProbe(name="p1")
        probe2 = MockProbe(name="p2")
        self.net.add_probe(probe1)
        self.net.add_probe(probe2)
        self.assertCountEqual(self.net.probes, [probe1, probe2])
        self.assertCountEqual(self.net.get_all_probes(), [probe1, probe2])

        # p1 is MockProbe, by default tracks 'mock_value'
        probe1_data = self.net.get_probe_data("p1")
        self.assertIn("time", probe1_data)
        self.assertIn("data", probe1_data)
        self.assertIn("mock_value", probe1_data["data"])
        self.assertEqual(len(probe1_data["time"]), 0)
        self.assertEqual(len(probe1_data["data"]["mock_value"]), 0)
        
        with self.assertRaises(TypeError):
            self.net.add_probe("not_a_probe_instance")
        with self.assertRaises(KeyError):
            self.net.get_probe_data("non_existent_probe")

    def test_network_step(self):
        self.net.add_population(self.pop1) # input
        self.net.add_population(self.pop2) # intermediate
        self.net.add_population(self.pop_output) # output

        self.net.set_input_populations(["pop1"])
        self.net.set_output_populations(["pop_out"])

        syn12 = MockSynapseCollection(self.pop1, self.pop2, name="s12")
        mock_lr = MockLearningRule()
        syn12.set_learning_rule(mock_lr)
        self.net.add_synapses(syn12)
        
        syn2out = MockSynapseCollection(self.pop2, self.pop_output, name="s2out")
        self.net.add_synapses(syn2out) # No learning rule for this one
        
        # probe_pop1_spikes = MockProbe(name="pop1_spikes")
        # Modify probe to actually target something, e.g. pop1 spikes
        probe_pop1_spikes = MockProbe(name="pop1_spikes", tracked_keys=["spikes_data"])
        # self.net.add_probe(probe_pop1_spikes) # Add it first, then override _collect_data if needed
                                            # OR, make MockProbe flexible enough.
                                            # For this test, let's make its _collect_data specific.

        def custom_collect_for_pop1_spikes(probe_instance, network, current_time):
            # This function will be bound to the probe instance if we choose that route
            # or used by a more generic CustomDataProbe.
            # For MockProbe, we can allow it to store under a specific key.
            spikes = network.get_population("pop1").get_spikes()
            probe_instance.data["spikes_data"].append(spikes.copy())
            # probe_instance.time_data.append(current_time) # BaseProbe handles this
            return True
        
        # Instead of overriding _collect_data directly on the instance after creation,
        # let's make MockProbe slightly more flexible or use a more suitable probe type.
        # For now, we will test the _record_probes generically via MockProbe's default.
        # The specific check for pop1 spikes will rely on MockProbe collecting *something*.
        
        # Simplified test: use a standard MockProbe for pop1_spikes
        # and check if its _collect_data was called.
        # The previous custom _collect_data logic was too complex for a simple mock here.
        mock_probe_for_step = MockProbe(name="step_probe")
        self.net.add_probe(mock_probe_for_step)

        dt = 1.0
        current_time = 0.0
        input_currents_map = {"pop1": np.ones(self.pop1.num_neurons) * 5.0}

        # --- First step --- 
        # Simulate pop1 having some spikes from a previous (imaginary) step for syn12 to use
        initial_pop1_spikes_for_synapses = np.array(
            [True, False] * (self.pop1.num_neurons // 2) + [False] * (self.pop1.num_neurons % 2), dtype=bool
        )
        self.pop1._spikes = initial_pop1_spikes_for_synapses.copy() # Set the mock pop's internal state for get_spikes()
        
        output_map = self.net.step(input_currents_map, dt, current_time)

        # 1. Check population updates (pop1 should have been updated with external current)
        self.assertIsNotNone(self.pop1.update_called_with)
        np.testing.assert_array_equal(self.pop1.update_called_with[0], input_currents_map["pop1"]) # Only external current
        
        # 2. Check synapse input current calculation
        # syn12 should use pop1's pre-update spikes
        self.assertIsNotNone(syn12.get_input_currents_called_with)
        # Compare with the state *before* pop1 was updated by the network step
        np.testing.assert_array_equal(syn12.get_input_currents_called_with, initial_pop1_spikes_for_synapses)
        
        # pop2 update should include synaptic current from syn12
        self.assertIsNotNone(self.pop2.update_called_with)
        expected_current_pop2 = syn12._mock_currents # As per MockSynapseCollection
        np.testing.assert_array_equal(self.pop2.update_called_with[0], expected_current_pop2)

        # 3. Check learning rule application
        self.assertTrue(syn12.apply_learning_rule_called)
        self.assertTrue(mock_lr.update_weights_called_with is not None)
        # Learning rule uses post-update (current step) spikes
        self.assertEqual(mock_lr.update_weights_called_with[1], self.pop1.get_spikes().shape) # pop1's new spikes
        self.assertEqual(mock_lr.update_weights_called_with[2], self.pop2.get_spikes().shape) # pop2's new spikes
        self.assertFalse(syn2out.apply_learning_rule_called) # syn2out has no LR

        # 4. Check output collection
        self.assertIn("pop_out", output_map)
        np.testing.assert_array_equal(output_map["pop_out"], self.pop_output.get_spikes())
        self.assertNotIn("pop1", output_map) # pop1 is not in output_population_names for output map

        # 5. Check probe recording
        # self.assertTrue(probe_pop1_spikes.attempt_record_called or probe_pop1_spikes.collect_called_with is not None)
        # probe_data = self.net.get_probe_data("pop1_spikes")
        # self.assertEqual(len(probe_data["time"]), 1)
        # self.assertEqual(probe_data["time"][0], current_time)
        # np.testing.assert_array_equal(probe_data["spikes"][0], self.pop1.get_spikes()) # Spikes after update
        self.assertIsNotNone(mock_probe_for_step.collect_called_with, "Probe's _collect_data should have been called.")
        step_probe_data = self.net.get_probe_data("step_probe")
        self.assertEqual(len(step_probe_data["time"]), 1)
        self.assertEqual(step_probe_data["time"][0], current_time)
        self.assertEqual(len(step_probe_data["data"]["mock_value"]), 1) # MockProbe records 'mock_value'

    def test_network_step_input_mismatch(self):
        self.net.add_population(self.pop1)
        self.net.set_input_populations(["pop1"])
        input_currents_map_wrong = {"pop1": np.ones(self.pop1.num_neurons - 1)} # Wrong length
        with self.assertRaises(ValueError):
            self.net.step(input_currents_map_wrong, 1.0, 0.0)

    def test_network_reset(self):
        self.net.add_population(self.pop1)
        self.net.add_population(self.pop2)
        syn12 = MockSynapseCollection(self.pop1, self.pop2, name="s12")
        mock_lr = MockLearningRule()
        syn12.set_learning_rule(mock_lr)
        self.net.add_synapses(syn12)
        probe1 = MockProbe(name="p1")
        self.net.add_probe(probe1)
        
        # Simulate some activity
        self.pop1.update(np.ones(self.pop1.num_neurons), 1.0, 0.0)
        # probe1.data_buffer.append("some data") # This was the error
        # Manually add some data consistent with BaseProbe structure for reset testing
        probe1.time_data.append(0.0)
        if "mock_value" not in probe1.data:
            probe1.data["mock_value"] = []
        probe1.data["mock_value"].append(0.5) 
        probe1._steps_since_last_record = 0 # Ensure reset clears this too
        
        self.net.reset()
        
        self.assertTrue(self.pop1.reset_called)
        self.assertTrue(self.pop2.reset_called)
        self.assertTrue(mock_lr.reset_called)
        self.assertTrue(probe1.reset_called)
        self.assertEqual(len(probe1.get_data()["time"]), 0)

if __name__ == '__main__':
    unittest.main() 