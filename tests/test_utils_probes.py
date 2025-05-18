'''单元测试代码 for dynn.utils.probes'''
import unittest
from unittest.mock import Mock, patch
import numpy as np
import os
import csv
from collections import OrderedDict

from dynn.utils.probes import (
    BaseProbe,
    PopulationProbe,
    SynapseProbe,
    CustomDataProbe,
)

# Helper mock classes
class MockPopulation:
    def __init__(self, name, size, initial_states=None):
        self.name = name
        self.num_neurons = size
        self.states = initial_states if initial_states else {}
        # Example: self.states = {'v': np.zeros(size), 'fired': np.zeros(size, dtype=bool)}

    def get_all_states(self, state_vars):
        # Only return states that actually exist in self.states
        return {var: self.states[var] for var in state_vars if var in self.states}

class MockSynapseCollection:
    def __init__(self, name, weights):
        self.name = name
        self._weights = weights

    def get_weights(self):
        return self._weights

class MockNetwork:
    def __init__(self):
        self.populations = {}
        self.synapses = {}

    def add_population(self, pop):
        self.populations[pop.name] = pop

    def add_synapses(self, syn):
        self.synapses[syn.name] = syn


class TestBaseProbe(unittest.TestCase):
    def test_initialization(self):
        probe = BaseProbe(name="base_probe", record_interval=5)
        self.assertEqual(probe.name, "base_probe")
        self.assertEqual(probe.record_interval, 5)
        self.assertEqual(probe.data, {})
        self.assertEqual(probe.time_data, [])
        self.assertEqual(probe._steps_since_last_record, 0)

    def test_initialization_invalid_interval(self):
        with self.assertRaisesRegex(ValueError, "record_interval 必须是一个正整数"):
            BaseProbe(name="test", record_interval=0)
        with self.assertRaisesRegex(ValueError, "record_interval 必须是一个正整数"):
            BaseProbe(name="test", record_interval=-1)

    def test_attempt_record_logic(self):
        # Mock _collect_data as it's NotImplemented in BaseProbe
        probe = BaseProbe(name="base_probe", record_interval=2)
        probe._collect_data = Mock()
        mock_network = MockNetwork()

        # Step 1: Should not record
        recorded1 = probe.attempt_record(mock_network, current_time_ms=10.0)
        self.assertFalse(recorded1)
        self.assertEqual(probe._steps_since_last_record, 1)
        probe._collect_data.assert_not_called()
        self.assertEqual(len(probe.time_data), 0)

        # Step 2: Should record
        recorded2 = probe.attempt_record(mock_network, current_time_ms=20.0)
        self.assertTrue(recorded2)
        self.assertEqual(probe._steps_since_last_record, 0)
        probe._collect_data.assert_called_once_with(mock_network, 20.0)
        self.assertEqual(probe.time_data, [20.0])

        # Step 3: Should not record
        probe._collect_data.reset_mock()
        recorded3 = probe.attempt_record(mock_network, current_time_ms=30.0)
        self.assertFalse(recorded3)
        self.assertEqual(probe._steps_since_last_record, 1)
        probe._collect_data.assert_not_called()
        self.assertEqual(len(probe.time_data), 1)

    def test_get_data(self):
        probe = BaseProbe(name="base_probe")
        probe.time_data = [10, 20]
        probe.data = {"var1": [np.array([1,2]), np.array([3,4])], "var2": [0.1, 0.2]}
        
        retrieved_data = probe.get_data()
        self.assertEqual(retrieved_data['time'], [10, 20])
        self.assertTrue(np.array_equal(retrieved_data['data']['var1'][0], np.array([1,2])))
        self.assertTrue(np.array_equal(retrieved_data['data']['var1'][1], np.array([3,4])))
        self.assertEqual(retrieved_data['data']['var2'], [0.1, 0.2])

        # Test deep copy for numpy arrays
        original_np_array = probe.data["var1"][0]
        retrieved_np_array = retrieved_data['data']['var1'][0]
        self.assertNotEqual(id(original_np_array), id(retrieved_np_array))
        retrieved_np_array[0] = 99
        self.assertEqual(probe.data["var1"][0][0], 1) # Original should be unchanged

    def test_reset(self):
        probe = BaseProbe(name="base_probe", record_interval=2)
        probe._collect_data = Mock() # Avoid NotImplementedError
        probe.data = {"var1": []} # Initialize a key for reset to clear
        
        # Perform one non-recording step
        probe.attempt_record(MockNetwork(), 5.0)
        self.assertEqual(probe._steps_since_last_record, 1)
        self.assertEqual(len(probe.time_data), 0)

        # Perform a recording step
        probe.attempt_record(MockNetwork(), 10.0)
        probe.data["var1"].append(100) 
        self.assertNotEqual(len(probe.time_data), 0)
        self.assertNotEqual(len(probe.data["var1"]), 0)
        self.assertEqual(probe._steps_since_last_record, 0) # Should be 0 after recording

        # Reset and check
        probe.reset()
        self.assertEqual(probe.time_data, [])
        self.assertEqual(probe.data["var1"], [])
        self.assertEqual(probe._steps_since_last_record, 0)
    
    def test_repr(self):
        probe = BaseProbe("myprobe", 3)
        self.assertEqual(repr(probe), "BaseProbe(name='myprobe', interval=3, records=0)")
        probe.time_data = [1,2]
        self.assertEqual(repr(probe), "BaseProbe(name='myprobe', interval=3, records=2)")

    def test_export_to_csv_no_data(self):
        probe = BaseProbe(name="empty_probe")
        probe.data = {'voltage': [], 'current': []} # Define keys but no data
        filepath = "empty_probe_output.csv"
        try:
            probe.export_to_csv(filepath)
            self.assertTrue(os.path.exists(filepath))
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertEqual(header, ['time', 'voltage', 'current'])
                with self.assertRaises(StopIteration): # No data rows
                    next(reader)
        finally:
            if os.path.exists(filepath): os.remove(filepath)

    def test_export_to_csv_with_data(self):
        probe = BaseProbe(name="data_probe")
        probe.time_data = [10, 20, 30]
        # Data with scalar and vector types
        probe.data = OrderedDict() # Use OrderedDict for predictable column order in test
        probe.data['scalar_var'] = [1.1, 2.2, 3.3]
        probe.data['vector_var'] = [np.array([1,2]), np.array([3,4]), np.array([5,6])]
        probe.data['missing_later'] = [100, 200] # Shorter data to test None padding

        filepath = "data_probe_output.csv"
        try:
            probe.export_to_csv(filepath)
            self.assertTrue(os.path.exists(filepath))
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertEqual(header, ['time', 'scalar_var', 'vector_var_0', 'vector_var_1', 'missing_later'])
                
                row1 = next(reader)
                self.assertEqual(row1, ['10', '1.1', '1', '2', '100'])
                row2 = next(reader)
                self.assertEqual(row2, ['20', '2.2', '3', '4', '200'])
                row3 = next(reader)
                # missing_later should have None (empty string in CSV) for the last time step
                self.assertEqual(row3, ['30', '3.3', '5', '6', ''])

        finally:
            if os.path.exists(filepath): os.remove(filepath)


class TestPopulationProbe(unittest.TestCase):
    def setUp(self):
        self.mock_network = MockNetwork()
        pop1_states = {
            'v': np.array([-70.0, -65.0]),
            'fired': np.array([False, True], dtype=bool)
        }
        self.pop1 = MockPopulation(name="pop1", size=2, initial_states=pop1_states)
        self.mock_network.add_population(self.pop1)

    def test_initialization(self):
        probe = PopulationProbe(name="p_probe", population_name="pop1", state_vars=['v', 'fired'])
        self.assertEqual(probe.population_name, "pop1")
        self.assertEqual(probe.state_vars, ['v', 'fired'])
        self.assertIn('v', probe.data)
        self.assertIn('fired', probe.data)

    def test_initialization_invalid_args(self):
        with self.assertRaisesRegex(ValueError, "population_name 必须是一个非空字符串"):
            PopulationProbe("p", "", ['v'])
        with self.assertRaisesRegex(ValueError, "state_vars 必须是一个包含变量名的列表/元组"):
            PopulationProbe("p", "pop1", "v")
        with self.assertRaisesRegex(ValueError, "state_vars 必须是一个包含变量名的列表/元组"):
            PopulationProbe("p", "pop1", [])

    @patch('builtins.print') # Mock print to check warnings
    def test_collect_data_valid(self, mock_print):
        probe = PopulationProbe("p_probe", "pop1", ['v', 'fired'], record_interval=1)
        probe.attempt_record(self.mock_network, 10.0)

        self.assertEqual(len(probe.time_data), 1)
        self.assertEqual(probe.time_data[0], 10.0)
        self.assertTrue(np.array_equal(probe.data['v'][0], np.array([-70.0, -65.0])))
        self.assertTrue(np.array_equal(probe.data['fired'][0], np.array([False, True])))
        mock_print.assert_not_called()

    @patch('builtins.print')
    def test_collect_data_population_not_found(self, mock_print):
        probe = PopulationProbe("p_probe", "non_existent_pop", ['v'], record_interval=1)
        probe.attempt_record(self.mock_network, 10.0)
        self.assertEqual(probe.data['v'][0], None)
        mock_print.assert_any_call("警告 (探针 'p_probe'): 群体 'non_existent_pop' 在网络中未找到。跳过此时间点记录。")

    @patch('builtins.print')
    def test_collect_data_state_var_not_found(self, mock_print):
        probe = PopulationProbe("p_probe", "pop1", ['u'], record_interval=1)
        probe.attempt_record(self.mock_network, 10.0)
        self.assertEqual(probe.data['u'][0], None) # Or should be an array of Nones?
        mock_print.assert_any_call("警告 (探针 'p_probe'): 状态变量 'u' 在群体 'pop1' 中未找到。")

    def test_repr_population(self):
        probe = PopulationProbe("p1", "popA", ['v'], 2)
        self.assertEqual(repr(probe), "PopulationProbe(name='p1', pop='popA', vars=['v'], interval=2, records=0)")


class TestSynapseProbe(unittest.TestCase):
    def setUp(self):
        self.mock_network = MockNetwork()
        self.syn_weights = np.array([[0.1, 0.2], [0.3, 0.4]])
        self.syn1 = MockSynapseCollection(name="syn1", weights=self.syn_weights)
        self.mock_network.add_synapses(self.syn1)

    def test_initialization(self):
        probe = SynapseProbe(name="s_probe", synapse_collection_name="syn1", record_weights=True)
        self.assertEqual(probe.synapse_collection_name, "syn1")
        self.assertTrue(probe.record_weights)
        self.assertIn('weights', probe.data)

    def test_initialization_invalid_name(self):
        with self.assertRaisesRegex(ValueError, "synapse_collection_name 必须是一个非空字符串"):
            SynapseProbe("s", "")

    @patch('builtins.print')
    def test_collect_data_valid(self, mock_print):
        probe = SynapseProbe("s_probe", "syn1", record_interval=1)
        probe.attempt_record(self.mock_network, 10.0)
        self.assertEqual(len(probe.time_data), 1)
        self.assertTrue(np.array_equal(probe.data['weights'][0], self.syn_weights))
        mock_print.assert_not_called()

    @patch('builtins.print')
    def test_collect_data_synapse_collection_not_found(self, mock_print):
        probe = SynapseProbe("s_probe", "non_existent_syn", record_interval=1)
        probe.attempt_record(self.mock_network, 10.0)
        self.assertEqual(probe.data['weights'][0], None)
        mock_print.assert_any_call("警告 (探针 's_probe'): 突触集合 'non_existent_syn' 在网络中未找到。跳过记录。")
        
    def test_collect_data_record_weights_false(self):
        probe = SynapseProbe("s_probe", "syn1", record_weights=False, record_interval=1)
        self.assertNotIn('weights', probe.data)
        probe.attempt_record(self.mock_network, 10.0)
        self.assertNotIn('weights', probe.data) # Still should not be there
        self.assertEqual(len(probe.time_data), 1) # Time is still recorded

    def test_repr_synapse(self):
        probe = SynapseProbe("s1", "synA", False, 3)
        self.assertEqual(repr(probe), "SynapseProbe(name='s1', syn='synA', weights=False, interval=3, records=0)")


class TestCustomDataProbe(unittest.TestCase):
    def setUp(self):
        self.mock_network = MockNetwork()
        self.data_provider_mock = Mock()

    def test_initialization(self):
        probe = CustomDataProbe("c_probe", self.data_provider_mock, data_keys=['reward', 'action'])
        self.assertEqual(probe.data_provider_fn, self.data_provider_mock)
        self.assertEqual(probe.data_keys, ['reward', 'action'])
        self.assertIn('reward', probe.data)
        self.assertIn('action', probe.data)

    def test_initialization_invalid_args(self):
        with self.assertRaisesRegex(TypeError, "data_provider_fn 必须是一个可调用对象"):
            CustomDataProbe("c", "not_callable", ['key'])
        with self.assertRaisesRegex(ValueError, "data_keys 必须是一个包含键名的列表/元组"):
            CustomDataProbe("c", self.data_provider_mock, "key")
        with self.assertRaisesRegex(ValueError, "data_keys 必须是一个包含键名的列表/元组"):
            CustomDataProbe("c", self.data_provider_mock, [])

    @patch('builtins.print')
    def test_collect_data_valid(self, mock_print):
        self.data_provider_mock.return_value = {'val1': 10, 'val2': np.array([1,2])}
        probe = CustomDataProbe("c_probe", self.data_provider_mock, ['val1', 'val2'], record_interval=1)
        probe.attempt_record(self.mock_network, 5.0)
        
        self.data_provider_mock.assert_called_once_with(self.mock_network, 5.0)
        self.assertEqual(probe.data['val1'][0], 10)
        self.assertTrue(np.array_equal(probe.data['val2'][0], np.array([1,2])))
        mock_print.assert_not_called()

    @patch('builtins.print')
    def test_collect_data_provider_not_dict(self, mock_print):
        self.data_provider_mock.return_value = "not_a_dict"
        probe = CustomDataProbe("c_probe", self.data_provider_mock, ['val1'], record_interval=1)
        probe.attempt_record(self.mock_network, 5.0)
        self.assertEqual(probe.data['val1'][0], None)
        mock_print.assert_any_call("警告 (探针 'c_probe'): data_provider_fn 未返回字典。跳过记录。")

    @patch('builtins.print')
    def test_collect_data_key_missing(self, mock_print):
        self.data_provider_mock.return_value = {'val1': 10} # Missing 'val2'
        probe = CustomDataProbe("c_probe", self.data_provider_mock, ['val1', 'val2'], record_interval=1)
        probe.attempt_record(self.mock_network, 5.0)
        self.assertEqual(probe.data['val1'][0], 10)
        self.assertEqual(probe.data['val2'][0], None)
        mock_print.assert_any_call("警告 (探针 'c_probe'): data_provider_fn 返回的数据中缺少键 'val2'。")

    @patch('builtins.print')
    def test_collect_data_provider_raises_exception(self, mock_print):
        self.data_provider_mock.side_effect = Exception("Provider error")
        probe = CustomDataProbe("c_probe", self.data_provider_mock, ['val1'], record_interval=1)
        probe.attempt_record(self.mock_network, 5.0)
        self.assertEqual(probe.data['val1'][0], None)
        mock_print.assert_any_call("错误 (探针 'c_probe'): 调用 data_provider_fn 时出错: Provider error")

    def test_repr_custom(self):
        dummy_fn = lambda nw, t: {}
        probe = CustomDataProbe("c1", dummy_fn, ['k1', 'k2'], 4)
        expected_repr = "CustomDataProbe(name='c1', keys=['k1', 'k2'], interval=4, records=0)"
        self.assertEqual(repr(probe), expected_repr)

        probe.time_data = [10, 20] # Add some records
        expected_repr_with_records = "CustomDataProbe(name='c1', keys=['k1', 'k2'], interval=4, records=2)"
        self.assertEqual(repr(probe), expected_repr_with_records)


if __name__ == "__main__":
    unittest.main() 