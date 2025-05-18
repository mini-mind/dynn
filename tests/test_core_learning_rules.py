'''单元测试代码 for dynn.core.learning_rules'''
import unittest
import numpy as np
from dynn.core.neurons import NeuronPopulation
from dynn.core.synapses import SynapseCollection
from dynn.core.learning_rules import BaseLearningRule, TraceSTDP

class TestBaseLearningRule(unittest.TestCase):
    def test_initialization(self):
        rule = BaseLearningRule(lr_ltp=0.01, lr_ltd=0.005)
        self.assertEqual(rule.lr_ltp, 0.01)
        self.assertEqual(rule.lr_ltd, 0.005)
        self.assertEqual(rule.reward_modulation, 1.0)

    def test_reward_modulation(self):
        rule = BaseLearningRule(lr_ltp=0.1, lr_ltd=0.05)
        rule.set_reward_modulation(0.5)
        self.assertEqual(rule.reward_modulation, 0.5)
        self.assertAlmostEqual(rule.get_effective_lr_ltp(), 0.1 * 0.5)
        self.assertAlmostEqual(rule.get_effective_lr_ltd(), 0.05 * 0.5)

        rule.set_reward_modulation(2.0)
        self.assertEqual(rule.reward_modulation, 2.0)
        self.assertAlmostEqual(rule.get_effective_lr_ltp(), 0.1 * 2.0)
        self.assertAlmostEqual(rule.get_effective_lr_ltd(), 0.05 * 2.0)

    def test_update_weights_not_implemented(self):
        rule = BaseLearningRule()
        with self.assertRaises(NotImplementedError):
            rule.update_weights(None, None, None, None, None)

class TestTraceSTDP(unittest.TestCase):
    def setUp(self):
        self.pre_pop = NeuronPopulation(num_neurons=3, name="pre")
        self.post_pop = NeuronPopulation(num_neurons=2, name="post")
        
        self.syn_collection = SynapseCollection(self.pre_pop, self.post_pop, name="s1")
        # Initialize with some weights and full connectivity for easier testing
        self.syn_collection.initialize_weights(dist_config=0.5, connectivity_type='full')
        self.syn_collection.is_excitatory = True # Default, but explicit

        self.stdp_rule = TraceSTDP(lr_ltp=0.1, lr_ltd=0.1, 
                                   tau_pre=20, tau_post=20, 
                                   w_max=1.0, w_min=0.0, trace_increase=1.0)
        self.dt = 1.0 # ms

    def test_initialization(self):
        rule = TraceSTDP(lr_ltp=0.01, lr_ltd=0.005, tau_pre=10, tau_post=15, w_max=0.8, w_min=0.1, trace_increase=0.5)
        self.assertEqual(rule.lr_ltp, 0.01)
        self.assertEqual(rule.lr_ltd, 0.005)
        self.assertEqual(rule.tau_pre, 10)
        self.assertEqual(rule.tau_post, 15)
        self.assertEqual(rule.w_max, 0.8)
        self.assertEqual(rule.w_min, 0.1)
        self.assertEqual(rule.trace_increase, 0.5)

    def test_update_traces_pre_synaptic(self):
        pre_spikes = np.array([True, False, True], dtype=bool)
        initial_trace_pre = np.array([0.1, 0.2, 0.0])
        self.pre_pop.spike_trace_pre = initial_trace_pre.copy()

        self.stdp_rule.update_traces(self.pre_pop, pre_spikes, 'pre', self.dt)
        
        decay_factor = np.exp(-self.dt / self.stdp_rule.tau_pre)
        expected_trace = initial_trace_pre * decay_factor
        expected_trace[pre_spikes] += self.stdp_rule.trace_increase
        
        np.testing.assert_array_almost_equal(self.pre_pop.spike_trace_pre, expected_trace)

    def test_update_traces_post_synaptic(self):
        post_spikes = np.array([False, True], dtype=bool)
        initial_trace_post = np.array([0.3, 0.1])
        self.post_pop.spike_trace_post = initial_trace_post.copy()

        self.stdp_rule.update_traces(self.post_pop, post_spikes, 'post', self.dt)

        decay_factor = np.exp(-self.dt / self.stdp_rule.tau_post)
        expected_trace = initial_trace_post * decay_factor
        expected_trace[post_spikes] += self.stdp_rule.trace_increase
        
        np.testing.assert_array_almost_equal(self.post_pop.spike_trace_post, expected_trace)

    def test_update_weights_ltp_only(self):
        # Post-synaptic spike, pre-synaptic trace non-zero
        self.pre_pop.spike_trace_pre = np.array([0.5, 0.0, 0.0]) # Pre neuron 0 had recent activity
        self.post_pop.spike_trace_post = np.zeros(self.post_pop.num_neurons) # No recent post activity for LTD part

        pre_spikes = np.array([False, False, False], dtype=bool) # No pre-spike in current step
        post_spikes = np.array([True, False], dtype=bool)      # Post neuron 0 spikes
        
        initial_weights = self.syn_collection.weights.copy()
        self.stdp_rule.update_weights(self.syn_collection, pre_spikes, post_spikes, self.dt, 0.0)

        # Expected change for W[0,0] (post_pop[0] from pre_pop[0])
        # dw_ltp = lr_ltp * trace_pre[0] * (w_max - W[0,0])
        delta_w00 = self.stdp_rule.lr_ltp * self.pre_pop.spike_trace_pre[0] * (self.stdp_rule.w_max - initial_weights[0,0])
        
        self.assertAlmostEqual(self.syn_collection.weights[0,0], initial_weights[0,0] + delta_w00)
        # Other weights connected to post_pop[0] but with zero pre-trace should not change from LTP
        self.assertAlmostEqual(self.syn_collection.weights[0,1], initial_weights[0,1]) # pre_trace[1] was 0
        # Weights connected to non-spiking post_pop[1] should not change
        np.testing.assert_array_almost_equal(self.syn_collection.weights[1,:], initial_weights[1,:])

    def test_update_weights_ltd_only(self):
        # Pre-synaptic spike, post-synaptic trace non-zero
        self.pre_pop.spike_trace_pre = np.zeros(self.pre_pop.num_neurons)
        self.post_pop.spike_trace_post = np.array([0.6, 0.0]) # Post neuron 0 had recent activity

        pre_spikes = np.array([True, False, False], dtype=bool) # Pre neuron 0 spikes
        post_spikes = np.array([False, False], dtype=bool)     # No post-spike in current step

        initial_weights = self.syn_collection.weights.copy()
        self.stdp_rule.update_weights(self.syn_collection, pre_spikes, post_spikes, self.dt, 0.0)

        # Expected change for W[0,0] (post_pop[0] from pre_pop[0])
        # dw_ltd = -lr_ltd * trace_post[0] * (W[0,0] - w_min)
        delta_w00 = -self.stdp_rule.lr_ltd * self.post_pop.spike_trace_post[0] * (initial_weights[0,0] - self.stdp_rule.w_min)

        self.assertAlmostEqual(self.syn_collection.weights[0,0], initial_weights[0,0] + delta_w00)
        # Other weights from pre_pop[0] to non-active-trace post_pop[1] should not change from LTD
        self.assertAlmostEqual(self.syn_collection.weights[1,0], initial_weights[1,0]) # post_trace[1] was 0
        # Weights from non-spiking pre_pop[1], pre_pop[2] should not change
        np.testing.assert_array_almost_equal(self.syn_collection.weights[:,1], initial_weights[:,1])
        np.testing.assert_array_almost_equal(self.syn_collection.weights[:,2], initial_weights[:,2])

    def test_weight_clipping_excitatory(self):
        original_lr_ltp = self.stdp_rule.lr_ltp
        original_lr_ltd = self.stdp_rule.lr_ltd

        self.stdp_rule.w_max = 0.7
        self.stdp_rule.w_min = 0.1 
        self.syn_collection.is_excitatory = True
        self.syn_collection.weights.fill(self.stdp_rule.w_max - 0.01) # e.g., 0.69
        self.stdp_rule.lr_ltp = 20.0 # Temporarily increase LR to force overshoot

        # Force LTP
        # Traces will be updated inside update_weights. Set initial traces high.
        self.pre_pop.spike_trace_pre = np.array([1.0, 0.0, 0.0]) 
        self.post_pop.spike_trace_post = np.zeros(self.post_pop.num_neurons)
        pre_spikes = np.array([False, False, False], dtype=bool) # No current pre-spike, relies on existing trace
        post_spikes = np.array([True, False], dtype=bool)      # Post neuron 0 spikes
        
        # Calculation of expected pre_trace[0] after decay:
        # trace_pre_after_decay = 1.0 * np.exp(-self.dt / self.stdp_rule.tau_pre)
        # delta_w_expected_no_clip = self.stdp_rule.lr_ltp * trace_pre_after_decay * (self.stdp_rule.w_max - (self.stdp_rule.w_max - 0.01))
        # self.assertTrue((self.stdp_rule.w_max - 0.01) + delta_w_expected_no_clip > self.stdp_rule.w_max, "LTP should overshoot w_max")

        self.stdp_rule.update_weights(self.syn_collection, pre_spikes, post_spikes, self.dt, 0.0)
        self.assertAlmostEqual(self.syn_collection.weights[0,0], self.stdp_rule.w_max, places=5) 

        # Reset for LTD part
        self.syn_collection.weights.fill(self.stdp_rule.w_min + 0.01) # e.g., 0.11
        self.stdp_rule.lr_ltd = 20.0 # Temporarily increase LR
        self.pre_pop.spike_trace_pre = np.zeros(self.pre_pop.num_neurons)
        self.post_pop.spike_trace_post = np.array([1.0, 0.0]) # Post trace for LTD
        pre_spikes = np.array([True, False, False], dtype=bool) # Pre neuron 0 spikes for LTD
        post_spikes = np.array([False, False], dtype=bool)

        # trace_post_after_decay = 1.0 * np.exp(-self.dt / self.stdp_rule.tau_post)
        # delta_w_expected_no_clip = -self.stdp_rule.lr_ltd * trace_post_after_decay * ((self.stdp_rule.w_min + 0.01) - self.stdp_rule.w_min)
        # self.assertTrue((self.stdp_rule.w_min + 0.01) + delta_w_expected_no_clip < self.stdp_rule.w_min, "LTD should undershoot w_min")

        self.stdp_rule.update_weights(self.syn_collection, pre_spikes, post_spikes, self.dt, 0.0)
        # actual_w_min for excitatory is max(0, self.w_min)
        actual_w_min_exc = max(0, self.stdp_rule.w_min)
        self.assertAlmostEqual(self.syn_collection.weights[0,0], actual_w_min_exc, places=5)
        
        self.stdp_rule.lr_ltp = original_lr_ltp # Restore original LR
        self.stdp_rule.lr_ltd = original_lr_ltd

    def test_weight_clipping_inhibitory(self):
        original_lr_ltp = self.stdp_rule.lr_ltp
        original_lr_ltd = self.stdp_rule.lr_ltd

        self.stdp_rule.w_max = -0.1 
        self.stdp_rule.w_min = -0.7
        self.syn_collection.set_excitatory(False) # This will make weights negative
        # Weights are now -0.5 (from setUp initial 0.5, then set_excitatory(False))
        # Let's set them closer to boundary
        self.syn_collection.weights.fill(self.stdp_rule.w_max - (-0.01)) # e.g., -0.11, which is more negative than w_max (-0.1)
                                                                    # Error in logic: should be w_max + some_negative_val for LTP test, or w_min + some_positive_val for LTD
                                                                    # For LTP (towards w_max = -0.1), current W should be more negative, e.g. -0.11
        self.syn_collection.weights.fill(self.stdp_rule.w_max - 0.01) # So W = -0.1 - 0.01 = -0.11

        self.stdp_rule.lr_ltp = 20.0 # Temporarily increase LR

        # Force LTP (making weight less negative, towards w_max = -0.1)
        self.pre_pop.spike_trace_pre = np.array([1.0, 0.0, 0.0])
        self.post_pop.spike_trace_post = np.zeros(self.post_pop.num_neurons)
        pre_spikes = np.array([False, False, False], dtype=bool)
        post_spikes = np.array([True, False], dtype=bool) 

        self.stdp_rule.update_weights(self.syn_collection, pre_spikes, post_spikes, self.dt, 0.0)
        actual_w_max_inh = min(0, self.stdp_rule.w_max) # = -0.1
        self.assertAlmostEqual(self.syn_collection.weights[0,0], actual_w_max_inh, places=5)

        # Reset for LTD part (making weight more negative, towards w_min = -0.7)
        # Current W should be less negative, e.g., -0.69
        self.syn_collection.weights.fill(self.stdp_rule.w_min + 0.01) # So W = -0.7 + 0.01 = -0.69
        self.stdp_rule.lr_ltd = 20.0 # Temporarily increase LR

        self.pre_pop.spike_trace_pre = np.zeros(self.pre_pop.num_neurons)
        self.post_pop.spike_trace_post = np.array([1.0, 0.0])
        pre_spikes = np.array([True, False, False], dtype=bool)
        post_spikes = np.array([False, False], dtype=bool)

        self.stdp_rule.update_weights(self.syn_collection, pre_spikes, post_spikes, self.dt, 0.0)
        actual_w_min_inh = self.stdp_rule.w_min # = -0.7
        self.assertAlmostEqual(self.syn_collection.weights[0,0], actual_w_min_inh, places=5)

        self.stdp_rule.lr_ltp = original_lr_ltp # Restore original LR
        self.stdp_rule.lr_ltd = original_lr_ltd

    def test_reward_modulation_effect_on_update(self):
        self.stdp_rule.set_reward_modulation(0.5)
        self.pre_pop.spike_trace_pre = np.array([0.5, 0.0, 0.0])
        post_spikes = np.array([True, False], dtype=bool)
        pre_spikes = np.array([False, False, False], dtype=bool)
        
        initial_weights = self.syn_collection.weights.copy()
        self.stdp_rule.update_weights(self.syn_collection, pre_spikes, post_spikes, self.dt, 0.0)
        
        # Expected change for W[0,0] with reward modulation
        # dw_ltp = (lr_ltp * reward_mod) * trace_pre[0] * (w_max - W[0,0])
        effective_lr = self.stdp_rule.lr_ltp * 0.5
        delta_w00 = effective_lr * self.pre_pop.spike_trace_pre[0] * (self.stdp_rule.w_max - initial_weights[0,0])
        self.assertAlmostEqual(self.syn_collection.weights[0,0], initial_weights[0,0] + delta_w00)

    def test_no_update_for_non_connected_synapses(self):
        # Make a sparse connection: only (0,0) is connected
        self.syn_collection.connection_mask.fill(False)
        self.syn_collection.connection_mask[0,0] = True
        self.syn_collection.weights.fill(0.0) # Zero out non-connected weights explicitly
        self.syn_collection.weights[0,0] = 0.5 # Connected weight
        initial_weights = self.syn_collection.weights.copy()

        self.pre_pop.spike_trace_pre = np.ones(self.pre_pop.num_neurons) # All pre traces high
        post_spikes = np.array([True, True], dtype=bool) # All post neurons spike
        pre_spikes = np.array([False, False, False], dtype=bool)

        self.stdp_rule.update_weights(self.syn_collection, pre_spikes, post_spikes, self.dt, 0.0)
        
        # Only W[0,0] should change
        self.assertNotAlmostEqual(self.syn_collection.weights[0,0], initial_weights[0,0])
        # All other weights should remain 0 because mask is False or they were not affected by this spike pattern anyway
        self.assertAlmostEqual(self.syn_collection.weights[0,1], 0.0)
        self.assertAlmostEqual(self.syn_collection.weights[0,2], 0.0)
        self.assertAlmostEqual(self.syn_collection.weights[1,0], 0.0)
        self.assertAlmostEqual(self.syn_collection.weights[1,1], 0.0)
        self.assertAlmostEqual(self.syn_collection.weights[1,2], 0.0)

    def test_repr_method(self):
        representation = repr(self.stdp_rule)
        self.assertIn("TraceSTDP", representation)
        self.assertIn(f"lr_ltp={self.stdp_rule.lr_ltp}", representation)
        self.assertIn(f"tau_pre={self.stdp_rule.tau_pre}", representation)

if __name__ == '__main__':
    unittest.main() 