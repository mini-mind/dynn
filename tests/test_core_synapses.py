'''单元测试代码 for dynn.core.synapses'''
import unittest
import numpy as np
from dynn.core.neurons import NeuronPopulation # 需要 NeuronPopulation 来创建突触
from dynn.core.synapses import SynapseCollection
# 假设有一个简单的学习规则用于测试
class MockLearningRule:
    def __init__(self):
        self.update_called = False
        self.params_received = None

    def update_weights(self, synapse_collection, pre_spikes, post_spikes, dt, current_time):
        self.update_called = True
        self.params_received = {
            "syn_shape": synapse_collection.weights.shape,
            "pre_len": len(pre_spikes),
            "post_len": len(post_spikes),
            "dt": dt,
            "time": current_time
        }

class TestSynapseCollection(unittest.TestCase):
    def setUp(self):
        # 创建一些模拟的神经元群体
        self.pop_pre = NeuronPopulation(num_neurons=5, name="pre")
        self.pop_post = NeuronPopulation(num_neurons=3, name="post")
        self.pop_square = NeuronPopulation(num_neurons=4, name="square")


    def test_synapse_collection_initialization(self):
        syn = SynapseCollection(self.pop_pre, self.pop_post, name="s1")
        self.assertEqual(syn.name, "s1")
        self.assertIs(syn.pre_pop, self.pop_pre)
        self.assertIs(syn.post_pop, self.pop_post)
        self.assertEqual(syn.weights.shape, (len(self.pop_post), len(self.pop_pre))) # (3, 5)
        self.assertEqual(syn.connection_mask.shape, (3, 5))
        self.assertTrue(np.all(syn.weights == 0))
        self.assertTrue(np.all(syn.connection_mask == False)) # 默认应该是全零，除非初始化
        self.assertIsNone(syn.learning_rule)
        self.assertTrue(syn.is_excitatory)

    def test_initialize_weights_fixed_value_full_connectivity(self):
        syn = SynapseCollection(self.pop_pre, self.pop_post)
        syn.initialize_weights(dist_config=0.5, connectivity_type='full')
        self.assertTrue(np.all(syn.weights == 0.5))
        self.assertTrue(np.all(syn.connection_mask == True))

    def test_initialize_weights_uniform_sparse_prob(self):
        syn = SynapseCollection(self.pop_pre, self.pop_post)
        low, high = 0.1, 0.9
        prob = 0.5
        syn.initialize_weights(dist_config=('uniform', (low, high)), 
                               connectivity_type='sparse_prob', prob=prob)
        
        self.assertEqual(syn.weights.shape, (3, 5))
        self.assertTrue(np.sum(syn.connection_mask) > 0) # 应该有一些连接
        self.assertTrue(np.sum(syn.connection_mask) < syn.weights.size) # 不应该是全连接
        
        # 检查权重值是否在范围内且只在连接处非零
        for r in range(syn.weights.shape[0]):
            for c in range(syn.weights.shape[1]):
                if syn.connection_mask[r, c]:
                    self.assertTrue(low <= syn.weights[r, c] <= high)
                else:
                    self.assertEqual(syn.weights[r, c], 0)

    def test_initialize_weights_normal_sparse_num(self):
        syn = SynapseCollection(self.pop_pre, self.pop_post) # 5 pre, 3 post
        mean, std = 0.5, 0.1
        num_connections_per_post = 2 # 每个突触后神经元有2个来自突触前的连接
        
        syn.initialize_weights(dist_config=('normal', (mean, std)), 
                               connectivity_type='sparse_num', num_connections=num_connections_per_post)
        
        self.assertEqual(syn.weights.shape, (3, 5))
        # 每行 (突触后神经元) 应该有 num_connections_per_post 个 True
        for i in range(len(self.pop_post)):
            self.assertEqual(np.sum(syn.connection_mask[i, :]), num_connections_per_post)
        
        active_weights = syn.weights[syn.connection_mask]
        # 简单的统计检查，不期望完全精确，但应接近
        self.assertAlmostEqual(np.mean(active_weights), mean, delta=0.3) # delta较大因为样本少
        self.assertTrue(np.all(syn.weights[~syn.connection_mask] == 0))

    def test_initialize_weights_neighborhood_same_pop(self):
        syn = SynapseCollection(self.pop_square, self.pop_square) # 4x4
        radius = 1
        syn.initialize_weights(dist_config=1.0, connectivity_type='neighborhood', 
                               radius=radius, allow_self_connections=False)
        
        expected_mask = np.array([
            [0,1,0,0], # 0 connects to 1
            [1,0,1,0], # 1 connects to 0, 2
            [0,1,0,1], # 2 connects to 1, 3
            [0,0,1,0]  # 3 connects to 2
        ], dtype=bool)
        np.testing.assert_array_equal(syn.connection_mask, expected_mask)
        self.assertTrue(np.all(syn.weights[expected_mask] == 1.0))
        self.assertTrue(np.all(syn.weights[~expected_mask] == 0))

    def test_initialize_weights_neighborhood_same_pop_with_self(self):
        syn = SynapseCollection(self.pop_square, self.pop_square) # 4x4
        radius = 1
        syn.initialize_weights(dist_config=1.0, connectivity_type='neighborhood', 
                               radius=radius, allow_self_connections=True)
        
        expected_mask = np.array([
            [1,1,0,0], # 0 connects to 0, 1
            [1,1,1,0], # 1 connects to 0, 1, 2
            [0,1,1,1], # 2 connects to 1, 2, 3
            [0,0,1,1]  # 3 connects to 2, 3
        ], dtype=bool)
        np.testing.assert_array_equal(syn.connection_mask, expected_mask)

    def test_initialize_weights_excitatory_inhibitory(self):
        syn_exc = SynapseCollection(self.pop_pre, self.pop_post)
        syn_exc.is_excitatory = True # 应该已经是默认值
        syn_exc.initialize_weights(dist_config=('uniform', (-1.0, -0.1)), connectivity_type='full')
        # 因为是兴奋性，权重应被转为正值
        self.assertTrue(np.all(syn_exc.weights >= 0.1))
        self.assertTrue(np.all(syn_exc.weights <= 1.0))

        syn_inh = SynapseCollection(self.pop_pre, self.pop_post)
        syn_inh.is_excitatory = False # 设置为抑制性
        syn_inh.initialize_weights(dist_config=('uniform', (0.1, 1.0)), connectivity_type='full')
        # 因为是抑制性，权重应被转为负值
        self.assertTrue(np.all(syn_inh.weights <= -0.1))
        self.assertTrue(np.all(syn_inh.weights >= -1.0))
        
    def test_set_excitatory(self):
        syn = SynapseCollection(self.pop_pre, self.pop_post)
        syn.initialize_weights(dist_config=0.5, connectivity_type='full') # weights are 0.5
        self.assertTrue(syn.is_excitatory)
        
        syn.set_excitatory(False) # 转为抑制性
        self.assertFalse(syn.is_excitatory)
        self.assertTrue(np.all(syn.weights == -0.5))
        
        syn.set_excitatory(False) # 再次设置为抑制性，应无变化
        self.assertFalse(syn.is_excitatory)
        self.assertTrue(np.all(syn.weights == -0.5))

        syn.set_excitatory(True) # 转回兴奋性
        self.assertTrue(syn.is_excitatory)
        self.assertTrue(np.all(syn.weights == 0.5))

    def test_get_input_currents(self):
        syn = SynapseCollection(self.pop_pre, self.pop_post) # 5 pre (0-4), 3 post (0-2)
        # weights: post_idx, pre_idx
        # post0 from pre0, pre1
        # post1 from pre2
        # post2 from pre3, pre4
        weights = np.array([
            [1.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.3, 0.7]
        ])
        syn.weights = weights
        syn.connection_mask = (weights != 0) # mask from weights
        
        pre_spikes = np.array([True, False, True, True, False], dtype=bool) # pre0, pre2, pre3 spiked
        
        expected_currents = np.array([
            1.0 * 1 + 0.5 * 0,      # post0: from pre0 (1*1)
            2.0 * 1,                # post1: from pre2 (2*1)
            0.3 * 1 + 0.7 * 0       # post2: from pre3 (0.3*1)
        ])
        
        input_currents = syn.get_input_currents(pre_spikes)
        np.testing.assert_array_almost_equal(input_currents, expected_currents)

    def test_get_input_currents_value_error(self):
        syn = SynapseCollection(self.pop_pre, self.pop_post)
        syn.initialize_weights(0.1)
        wrong_spikes = np.array([True, False]) # 长度错误
        with self.assertRaises(ValueError):
            syn.get_input_currents(wrong_spikes)

    def test_learning_rule_interaction(self):
        syn = SynapseCollection(self.pop_pre, self.pop_post)
        mock_lr = MockLearningRule()
        syn.set_learning_rule(mock_lr)
        self.assertIs(syn.learning_rule, mock_lr)
        
        pre_spikes = np.zeros(len(self.pop_pre), dtype=bool)
        post_spikes = np.zeros(len(self.pop_post), dtype=bool)
        dt = 0.1
        current_time = 1.0
        
        syn.apply_learning_rule(pre_spikes, post_spikes, dt, current_time)
        
        self.assertTrue(mock_lr.update_called)
        self.assertIsNotNone(mock_lr.params_received)
        self.assertEqual(mock_lr.params_received["syn_shape"], syn.weights.shape)
        self.assertEqual(mock_lr.params_received["pre_len"], len(pre_spikes))
        self.assertEqual(mock_lr.params_received["post_len"], len(post_spikes))

    def test_get_weights_and_mask(self):
        syn = SynapseCollection(self.pop_pre, self.pop_post)
        syn.initialize_weights(0.7, connectivity_type='full')
        
        weights = syn.get_weights()
        mask = syn.get_connection_mask()
        
        self.assertIs(weights, syn.weights) # 应该返回内部数组的引用
        self.assertIs(mask, syn.connection_mask)
        np.testing.assert_array_equal(weights, np.full((3,5), 0.7))
        np.testing.assert_array_equal(mask, np.full((3,5), True))
        
    def test_repr_method(self):
        syn = SynapseCollection(self.pop_pre, self.pop_post, name="test_syn")
        syn.initialize_weights(0.1, 'sparse_prob', prob=0.5) # 让连接数不为0
        
        representation = repr(syn)
        self.assertIn("SynapseCollection", representation)
        self.assertIn(f"name='{syn.name}'", representation)
        self.assertIn(f"pre='{self.pop_pre.name}'", representation)
        self.assertIn(f"post='{self.pop_post.name}'", representation)
        self.assertIn(f"shape={syn.weights.shape!r}", representation)
        self.assertIn(f"connections={np.sum(syn.connection_mask)}", representation)


if __name__ == '__main__':
    unittest.main() 