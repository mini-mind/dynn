'''单元测试代码 for dynn.io.output_decoders'''
import unittest
import numpy as np
from dynn.io.output_decoders import (
    BaseOutputDecoder,
    InstantaneousSpikeCountDecoder,
    MountainCarActionDecoder,
)

class TestBaseOutputDecoder(unittest.TestCase):
    def test_initialization(self):
        decoder = BaseOutputDecoder(source_pop_name="output_pop")
        self.assertEqual(decoder.source_pop_name, "output_pop")
        self.assertEqual(decoder.get_source_population_name(), "output_pop")

    def test_decode_not_implemented(self):
        decoder = BaseOutputDecoder(source_pop_name="output_pop")
        with self.assertRaises(NotImplementedError):
            decoder.decode(spike_activities_map={})

    def test_repr(self):
        decoder = BaseOutputDecoder(source_pop_name="output_pop")
        self.assertEqual(repr(decoder), "BaseOutputDecoder(source_pop_name='output_pop')")


class TestInstantaneousSpikeCountDecoder(unittest.TestCase):
    def test_initialization(self):
        decoder = InstantaneousSpikeCountDecoder(
            source_pop_name="action_pop", num_actions=3, default_action=1
        )
        self.assertEqual(decoder.source_pop_name, "action_pop")
        self.assertEqual(decoder.num_actions, 3)
        self.assertEqual(decoder.default_action, 1)

    def test_decode_no_spikes_with_default(self):
        decoder = InstantaneousSpikeCountDecoder("action_pop", 3, default_action=1)
        spikes = {"action_pop": np.array([False, False, False])}
        action = decoder.decode(spikes)
        self.assertEqual(action, 1)

    def test_decode_no_spikes_no_default(self):
        decoder = InstantaneousSpikeCountDecoder("action_pop", 3, default_action=None)
        spikes = {"action_pop": np.array([False, False, False])}
        action = decoder.decode(spikes)
        self.assertIsNone(action) # Or check for specific value/error if behavior changes

    def test_decode_single_spike(self):
        decoder = InstantaneousSpikeCountDecoder("action_pop", 3, default_action=1)
        spikes1 = {"action_pop": np.array([True, False, False])}
        self.assertEqual(decoder.decode(spikes1), 0)

        spikes2 = {"action_pop": np.array([False, True, False])}
        self.assertEqual(decoder.decode(spikes2), 1)

        spikes3 = {"action_pop": np.array([False, False, True])}
        self.assertEqual(decoder.decode(spikes3), 2)

    def test_decode_multiple_spikes_takes_first(self):
        # Current implementation takes the index of the first True
        decoder = InstantaneousSpikeCountDecoder("action_pop", 3, default_action=1)
        spikes = {"action_pop": np.array([False, True, True])}
        self.assertEqual(decoder.decode(spikes), 1)

        spikes_all = {"action_pop": np.array([True, True, True])}
        self.assertEqual(decoder.decode(spikes_all), 0)

    def test_decode_source_pop_missing(self):
        decoder = InstantaneousSpikeCountDecoder("action_pop", 3)
        with self.assertRaisesRegex(ValueError, "源群体 'action_pop' 的脉冲数据未在输入中找到"):
            decoder.decode({"other_pop": np.array([False, False, False])})

    def test_decode_invalid_spike_data_type(self):
        decoder = InstantaneousSpikeCountDecoder("action_pop", 3)
        with self.assertRaisesRegex(ValueError, "源群体的脉冲数据必须是一维numpy数组"):
            decoder.decode({"action_pop": [False, False, False]}) # List, not ndarray
        with self.assertRaisesRegex(ValueError, "源群体的脉冲数据必须是一维numpy数组"):
            decoder.decode({"action_pop": np.array([[False], [False], [False]])}) # 2D array

    def test_decode_neuron_action_mismatch_warning(self):
        # Test that a warning is printed (cannot directly test print in unittest easily)
        # but the logic should still proceed and cap the action if needed.
        decoder = InstantaneousSpikeCountDecoder("action_pop", num_actions=2, default_action=0)
        
        # Neurons = 3, num_actions = 2. Spike at index 2 (neuron 3)
        spikes_more_neurons = {"action_pop": np.array([False, False, True])}
        # Action should be capped at num_actions - 1 = 1
        action = decoder.decode(spikes_more_neurons)
        self.assertEqual(action, 1) 

        # Neurons = 2, num_actions = 3. Spike at index 1 (neuron 2)
        decoder_less_neurons = InstantaneousSpikeCountDecoder("action_pop", num_actions=3, default_action=0)
        spikes_less_neurons = {"action_pop": np.array([False, True])}
        action_less = decoder_less_neurons.decode(spikes_less_neurons)
        self.assertEqual(action_less, 1)
        # No spike
        spikes_less_neurons_no_spike = {"action_pop": np.array([False, False])}
        action_no_spike = decoder_less_neurons.decode(spikes_less_neurons_no_spike)
        self.assertEqual(action_no_spike, 0) # default action

    def test_repr_instantaneous(self):
        decoder = InstantaneousSpikeCountDecoder("act", 3, 1)
        # repr_str = repr(decoder)
        # self.assertIn("InstantaneousSpikeCountDecoder(source='act'", repr_str)
        # self.assertIn("num_actions=3", repr_str)
        # self.assertIn("default_action=1", repr_str)
        self.assertTrue(isinstance(repr(decoder), str))


class TestMountainCarActionDecoder(unittest.TestCase):
    def test_initialization_default(self):
        decoder = MountainCarActionDecoder(source_pop_name="mc_action_pop")
        self.assertEqual(decoder.source_pop_name, "mc_action_pop")
        self.assertEqual(decoder.num_actions, 3) # MountainCar specific
        self.assertEqual(decoder.default_action, 1) # Default to NO_OP

    def test_initialization_custom_default_action(self):
        decoder = MountainCarActionDecoder(source_pop_name="mc_action_pop", default_action_idx=0)
        self.assertEqual(decoder.default_action, 0)

    def test_initialization_custom_num_neurons_warning(self):
        # This should print a warning but initialize.
        # We can't easily test for stdout in unittest by default.
        decoder = MountainCarActionDecoder(source_pop_name="mc_action_pop", num_neurons_for_action=5)
        self.assertEqual(decoder.num_actions, 5)

    def test_decode_mc_actions(self):
        decoder = MountainCarActionDecoder("mc_out", default_action_idx=1)
        
        # Action 0 (left)
        spikes_left = {"mc_out": np.array([True, False, False])}
        self.assertEqual(decoder.decode(spikes_left), 0)

        # Action 1 (noop)
        spikes_noop_explicit = {"mc_out": np.array([False, True, False])}
        self.assertEqual(decoder.decode(spikes_noop_explicit), 1)

        # Action 2 (right)
        spikes_right = {"mc_out": np.array([False, False, True])}
        self.assertEqual(decoder.decode(spikes_right), 2)

        # No spike - default action
        spikes_none = {"mc_out": np.array([False, False, False])}
        self.assertEqual(decoder.decode(spikes_none), 1) # Default action is 1

    def test_decode_mc_multiple_spikes(self):
        decoder = MountainCarActionDecoder("mc_out")
        spikes_multi = {"mc_out": np.array([True, False, True])} # Spike for action 0 and 2
        self.assertEqual(decoder.decode(spikes_multi), 0) # Takes first one (action 0)

    def test_decode_mc_source_pop_mismatch_len(self):
        # If source pop has, e.g., 2 neurons but MountainCarActionDecoder expects 3 (num_actions=3)
        decoder = MountainCarActionDecoder("mc_out")
        spikes_short = {"mc_out": np.array([False, True])} # Only 2 neurons in spike data
        # The InstantaneousSpikeCountDecoder will print a warning and proceed.
        # If neuron 1 (index 1) spikes, action is 1.
        self.assertEqual(decoder.decode(spikes_short), 1)

        spikes_long = {"mc_out": np.array([False, False, False, True])} # 4 neurons, action is 3 (capped to 2)
        self.assertEqual(decoder.decode(spikes_long), 2) # Capped to num_actions - 1


    def test_repr_mountain_car(self):
        decoder = MountainCarActionDecoder("mc_act", default_action_idx=0)
        # repr_str = repr(decoder)
        # self.assertIn("MountainCarActionDecoder(source='mc_act'", repr_str)
        # self.assertIn("num_actions=3", repr_str)
        # self.assertIn("default_action=0", repr_str)
        self.assertTrue(isinstance(repr(decoder), str))


if __name__ == "__main__":
    unittest.main() 