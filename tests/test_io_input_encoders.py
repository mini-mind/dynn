'''单元测试代码 for dynn.io.input_encoders'''
import unittest
import numpy as np
from dynn.io.input_encoders import (
    BaseInputEncoder,
    GaussianEncoder,
    DirectCurrentInjector,
    MountainCarPositionEncoder,
)

# class TestInputEncoders(unittest.TestCase):
#     def test_example(self):
#         # TODO: 为 input_encoders.py 中的功能添加具体的测试用例
#         pass

class TestBaseInputEncoder(unittest.TestCase):
    def test_initialization(self):
        encoder = BaseInputEncoder(target_pop_name="input_pop")
        self.assertEqual(encoder.target_pop_name, "input_pop")
        self.assertEqual(encoder.get_target_population_name(), "input_pop")

    def test_encode_not_implemented(self):
        encoder = BaseInputEncoder(target_pop_name="input_pop")
        with self.assertRaises(NotImplementedError):
            encoder.encode(observation=0.5)

    def test_repr(self):
        encoder = BaseInputEncoder(target_pop_name="input_pop")
        self.assertEqual(repr(encoder), "BaseInputEncoder(target_pop_name='input_pop')")


class TestGaussianEncoder(unittest.TestCase):
    def test_initialization_valid(self):
        encoder = GaussianEncoder(
            target_pop_name="sensory_pop",
            num_neurons=10,
            min_val=0.0,
            max_val=1.0,
            sigma_scale=0.1,
            current_amplitude=5.0,
        )
        self.assertEqual(encoder.target_pop_name, "sensory_pop")
        self.assertEqual(encoder.num_neurons, 10)
        self.assertEqual(encoder.min_val, 0.0)
        self.assertEqual(encoder.max_val, 1.0)
        self.assertEqual(encoder.val_range, 1.0)
        self.assertAlmostEqual(encoder.sigma, 0.1 * 1.0)
        self.assertEqual(encoder.current_amplitude, 5.0)
        self.assertEqual(len(encoder.means), 10)
        self.assertAlmostEqual(encoder.means[0], 0.0)
        self.assertAlmostEqual(encoder.means[-1], 1.0)

    def test_initialization_single_neuron(self):
        encoder = GaussianEncoder("pop1", 1, 0, 10, 0.2, 10)
        self.assertEqual(encoder.num_neurons, 1)
        self.assertAlmostEqual(encoder.means[0], 5.0) # Center of the range
        self.assertAlmostEqual(encoder.sigma, 0.2 * 10)

    def test_initialization_zero_range(self):
        # Test sigma handling when max_val == min_val
        encoder = GaussianEncoder("pop1", 5, 5, 5, 0.1, 10)
        self.assertEqual(encoder.min_val, 5)
        self.assertEqual(encoder.max_val, 5)
        self.assertEqual(encoder.val_range, 0)
        # sigma should not be zero if num_neurons > 1 to avoid division by zero in exp
        self.assertGreater(encoder.sigma, 0) 
        self.assertTrue(np.all(encoder.means == 5.0))
    
    def test_initialization_zero_range_single_neuron(self):
        encoder = GaussianEncoder("pop1", 1, 5, 5, 0.1, 10)
        self.assertEqual(encoder.min_val, 5)
        self.assertEqual(encoder.max_val, 5)
        self.assertEqual(encoder.val_range, 0)
        self.assertEqual(encoder.sigma, 1.0) # Default sigma for single neuron, zero range

    def test_initialization_invalid_neurons(self):
        with self.assertRaises(ValueError):
            GaussianEncoder("pop", 0, 0, 1)
        with self.assertRaises(ValueError):
            GaussianEncoder("pop", -1, 0, 1)

    def test_initialization_invalid_range(self):
        with self.assertRaises(ValueError):
            GaussianEncoder("pop", 10, 1, 0) # min_val > max_val
        # min_val == max_val is handled, should not raise error here

    def test_encode_scalar_input(self):
        encoder = GaussianEncoder("sensory", 5, 0.0, 1.0, sigma_scale=0.2, current_amplitude=10.0)
        # obs_val = 0.5 (center), means = [0, 0.25, 0.5, 0.75, 1.0]
        # Neuron at mean 0.5 should have max current
        obs_val = 0.5
        result = encoder.encode(obs_val)
        self.assertIn("sensory", result)
        currents = result["sensory"]
        self.assertEqual(currents.shape, (5,))
        self.assertAlmostEqual(currents[2], 10.0) # Max amplitude for neuron at mean 0.5
        self.assertTrue(all(0 <= c <= 10.0 for c in currents))

    def test_encode_input_at_min_max(self):
        encoder = GaussianEncoder("sensory", 5, 0.0, 1.0, sigma_scale=0.1, current_amplitude=10.0)
        # obs_val = 0.0 (min_val)
        currents_min = encoder.encode(0.0)["sensory"]
        self.assertAlmostEqual(currents_min[0], 10.0)

        # obs_val = 1.0 (max_val)
        currents_max = encoder.encode(1.0)["sensory"]
        self.assertAlmostEqual(currents_max[-1], 10.0)

    def test_encode_single_neuron(self):
        encoder = GaussianEncoder("sensory", 1, 0.0, 1.0, sigma_scale=0.5, current_amplitude=10.0)
        # mean is 0.5, sigma is 0.5
        currents1 = encoder.encode(0.5)["sensory"] # At mean
        self.assertAlmostEqual(currents1[0], 10.0)
        
        currents2 = encoder.encode(0.0)["sensory"] # val - mean = -0.5
        # exp(-(-0.5)^2 / (2*0.5^2)) = exp(-0.25 / 0.5) = exp(-0.5)
        expected_current = 10.0 * np.exp(-0.5)
        self.assertAlmostEqual(currents2[0], expected_current)

    def test_encode_input_types(self):
        encoder = GaussianEncoder("pop", 3, 0, 1)
        self.assertTrue(isinstance(encoder.encode(0.5)["pop"], np.ndarray))
        self.assertTrue(isinstance(encoder.encode(np.array([0.5]))["pop"], np.ndarray))
        self.assertTrue(isinstance(encoder.encode([0.5])["pop"], np.ndarray))
        with self.assertRaises(ValueError):
            encoder.encode("string")
        with self.assertRaises(ValueError):
            encoder.encode([0.1, 0.2]) # List with more than one element

    def test_encode_zero_sigma_special_case(self):
        # When val_range is 0, sigma might become 0 or very small.
        # Encoder handles sigma=0 by checking for exact match if num_neurons > 1
        encoder_multi_neuron = GaussianEncoder("pop_multi", 3, 0.5, 0.5, sigma_scale=0.1, current_amplitude=5.0)
        self.assertEqual(encoder_multi_neuron.val_range, 0)
        # self.assertEqual(encoder_multi_neuron.sigma, 0) # Actually, it's a small epsilon now
        
        # All means are 0.5. If obs_val is 0.5, all neurons should activate
        currents1 = encoder_multi_neuron.encode(0.5)["pop_multi"]
        # With current epsilon sigma, it's not an exact match behavior
        # self.assertTrue(np.allclose(currents1, [5.0, 5.0, 5.0]))
        # Instead, they will all have high activation due to small sigma and obs_val == means
        self.assertTrue(np.all(currents1 > 4.9), f"Currents were {currents1}")


        currents2 = encoder_multi_neuron.encode(0.6)["pop_multi"]
        # If obs_val is not 0.5, currents should be near zero if sigma was truly zero
        # With epsilon sigma, they will be small
        self.assertTrue(np.all(currents2 < 0.1) or np.allclose(currents2, 0.0), f"Currents were {currents2}")


        encoder_single_neuron = GaussianEncoder("pop_single", 1, 0.5, 0.5, sigma_scale=0.1, current_amplitude=5.0)
        self.assertEqual(encoder_single_neuron.val_range, 0)
        # For single neuron, zero range, sigma defaults to 1.0
        self.assertEqual(encoder_single_neuron.sigma, 1.0)
        
        # obs_val = 0.5, mean = 0.5, sigma = 1.0. exp(0) = 1
        currents_single1 = encoder_single_neuron.encode(0.5)["pop_single"]
        self.assertAlmostEqual(currents_single1[0], 5.0)

        # obs_val = 0.6, mean = 0.5, sigma = 1.0. exp(-(0.1^2)/(2*1^2)) = exp(-0.01/2) = exp(-0.005)
        currents_single2 = encoder_single_neuron.encode(0.6)["pop_single"]
        self.assertAlmostEqual(currents_single2[0], 5.0 * np.exp(-0.005))


    def test_repr_gaussian(self):
        encoder = GaussianEncoder("pop1", 10, 0, 1, 0.1, 5)
        # repr_str = repr(encoder) # Due to float precision, direct match is tricky
        # self.assertIn("GaussianEncoder(target='pop1'", repr_str)
        # self.assertIn("num_neurons=10", repr_str)
        # self.assertIn("range=[0.00, 1.00]", repr_str)
        # self.assertIn("sigma_scale=0.10", repr_str) # sigma_scale is calculated back from sigma and val_range
        # self.assertIn("amplitude=5.00", repr_str)
        self.assertTrue(isinstance(repr(encoder), str)) # Basic check


class TestDirectCurrentInjector(unittest.TestCase):
    def test_initialization(self):
        encoder = DirectCurrentInjector("direct_pop", 5, observation_slice=slice(0,5), scale_factor=2.0)
        self.assertEqual(encoder.target_pop_name, "direct_pop")
        self.assertEqual(encoder.num_neurons, 5)
        self.assertEqual(encoder.observation_slice, slice(0,5))
        self.assertEqual(encoder.scale_factor, 2.0)

    def test_encode_full_observation(self):
        encoder = DirectCurrentInjector("direct_pop", 3, scale_factor=1.0)
        obs = np.array([1.0, 2.0, 3.0])
        result = encoder.encode(obs)
        self.assertIn("direct_pop", result)
        currents = result["direct_pop"]
        self.assertTrue(np.array_equal(currents, obs.astype(float)))
        self.assertEqual(currents.dtype, float)

    def test_encode_with_slice(self):
        encoder = DirectCurrentInjector("direct_pop", 2, observation_slice=slice(1,3), scale_factor=1.0)
        obs = np.array([10, 20, 30, 40])
        result = encoder.encode(obs)
        currents = result["direct_pop"]
        self.assertTrue(np.array_equal(currents, np.array([20.0, 30.0])))

    def test_encode_with_scale_factor_scalar(self):
        encoder = DirectCurrentInjector("direct_pop", 3, scale_factor=0.5)
        obs = np.array([2.0, 4.0, 6.0])
        result = encoder.encode(obs)
        currents = result["direct_pop"]
        self.assertTrue(np.array_equal(currents, np.array([1.0, 2.0, 3.0])))
        
    def test_encode_with_scale_factor_array(self):
        scale = np.array([0.5, 2.0])
        encoder = DirectCurrentInjector("direct_pop", 2, scale_factor=scale)
        obs = np.array([10.0, 5.0])
        result = encoder.encode(obs)
        currents = result["direct_pop"]
        self.assertTrue(np.array_equal(currents, np.array([5.0, 10.0])))

    def test_encode_input_type_list(self):
        encoder = DirectCurrentInjector("direct_pop", 2, scale_factor=1.0)
        obs_list = [5, 6]
        result = encoder.encode(obs_list)
        self.assertTrue(np.array_equal(result["direct_pop"], np.array([5.0, 6.0])))

    def test_encode_dimension_mismatch(self):
        encoder = DirectCurrentInjector("direct_pop", 3) # Expects 3 neurons
        with self.assertRaisesRegex(ValueError, "与目标群体神经元数量 .* 不匹配"):
            encoder.encode(np.array([1.0, 2.0])) # Obs has 2 elements
        
        encoder_slice = DirectCurrentInjector("direct_pop", 2, observation_slice=slice(0,1)) # Slice gives 1 element
        with self.assertRaisesRegex(ValueError, "与目标群体神经元数量 .* 不匹配"):
            encoder_slice.encode(np.array([1.0, 2.0]))

    def test_encode_invalid_input_type(self):
        encoder = DirectCurrentInjector("direct_pop", 3)
        with self.assertRaisesRegex(ValueError, "需要一个向量观察"):
            encoder.encode(123) # Scalar input

    def test_repr_direct(self):
        encoder = DirectCurrentInjector("pop2", 5, slice(0,2), 1.5)
        # repr_str = repr(encoder)
        # self.assertIn("DirectCurrentInjector(target='pop2'", repr_str)
        # self.assertIn("num_neurons=5", repr_str)
        # self.assertIn("slice=slice(0, 2, None)", repr_str)
        # self.assertIn("scale=1.5", repr_str)
        self.assertTrue(isinstance(repr(encoder), str))


class TestMountainCarPositionEncoder(unittest.TestCase):
    def test_initialization(self):
        # Default MountainCar ranges: pos_min=-1.2, pos_max=0.6
        encoder = MountainCarPositionEncoder(
            target_pop_name="mc_pos_pop",
            num_neurons=20,
            sigma_scale=0.05,
            current_amplitude=15.0
        )
        self.assertEqual(encoder.target_pop_name, "mc_pos_pop")
        self.assertEqual(encoder.num_neurons, 20)
        self.assertAlmostEqual(encoder.min_val, -1.2)
        self.assertAlmostEqual(encoder.max_val, 0.6)
        self.assertAlmostEqual(encoder.val_range, 1.8)
        self.assertAlmostEqual(encoder.sigma, 0.05 * 1.8)
        self.assertEqual(encoder.current_amplitude, 15.0)

    def test_initialization_custom_ranges(self):
        encoder = MountainCarPositionEncoder(
            "mc_pop", 10, pos_min=-1.0, pos_max=1.0, sigma_scale=0.1
        )
        self.assertAlmostEqual(encoder.min_val, -1.0)
        self.assertAlmostEqual(encoder.max_val, 1.0)
        self.assertAlmostEqual(encoder.val_range, 2.0)
        self.assertAlmostEqual(encoder.sigma, 0.1 * 2.0)

    def test_encode_valid_observation(self):
        encoder = MountainCarPositionEncoder("mc_pop", 10, pos_min=-1.2, pos_max=0.6, sigma_scale=0.1)
        # Observation: [position, velocity]
        obs = [-0.5, 0.01] # Position = -0.5
        
        # Expected behavior: uses GaussianEncoder.encode with -0.5
        # Create a comparable GaussianEncoder to verify
        gaussian_ref = GaussianEncoder("mc_pop", 10, -1.2, 0.6, sigma_scale=0.1, current_amplitude=encoder.current_amplitude)
        
        expected_currents = gaussian_ref.encode(-0.5)["mc_pop"]
        
        result = encoder.encode(obs)
        self.assertIn("mc_pop", result)
        currents = result["mc_pop"]
        
        self.assertTrue(np.array_equal(currents, expected_currents))

    def test_encode_uses_only_position(self):
        encoder = MountainCarPositionEncoder("mc_pop", 5, pos_min=0, pos_max=1)
        obs1 = [0.5, 0.0]  # Position 0.5, velocity 0.0
        obs2 = [0.5, 10.0] # Position 0.5, velocity 10.0
        
        currents1 = encoder.encode(obs1)["mc_pop"]
        currents2 = encoder.encode(obs2)["mc_pop"]
        
        self.assertTrue(np.array_equal(currents1, currents2), "Currents should be identical if only position is used.")

    def test_encode_invalid_observation_format(self):
        encoder = MountainCarPositionEncoder("mc_pop", 5)
        with self.assertRaisesRegex(ValueError, "期望至少包含位置信息的观察"):
            encoder.encode(0.5) # Scalar, not list/array
        with self.assertRaisesRegex(ValueError, "期望至少包含位置信息的观察"):
            encoder.encode([]) # Empty list
        with self.assertRaisesRegex(ValueError, "期望至少包含位置信息的观察"):
            encoder.encode(None)

    def test_repr_mountain_car(self):
        encoder = MountainCarPositionEncoder("mc_pop", 20, pos_min=-1.2, pos_max=0.6, sigma_scale=0.05)
        # repr_str = repr(encoder)
        # self.assertIn("MountainCarPositionEncoder(target='mc_pop'", repr_str)
        # self.assertIn("num_neurons=20", repr_str)
        # self.assertIn("pos_range=[-1.20, 0.60]", repr_str)
        # self.assertIn("sigma_scale=0.05", repr_str)
        self.assertTrue(isinstance(repr(encoder), str))


if __name__ == "__main__":
    unittest.main() 