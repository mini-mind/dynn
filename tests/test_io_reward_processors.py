'''单元测试代码 for dynn.io.reward_processors'''
import unittest
import numpy as np
from collections import deque
from dynn.io.reward_processors import (
    BaseRewardProcessor,
    SlidingWindowSmoother,
    MountainCarCustomReward,
)

class TestBaseRewardProcessor(unittest.TestCase):
    def test_initialization(self):
        processor = BaseRewardProcessor()
        self.assertIsNotNone(processor)

    def test_process_not_implemented(self):
        processor = BaseRewardProcessor()
        with self.assertRaises(NotImplementedError):
            processor.process(raw_reward=0.0)

    def test_reset(self):
        processor = BaseRewardProcessor()
        try:
            processor.reset()  # Should not raise anything
        except Exception as e:
            self.fail(f"BaseRewardProcessor.reset() raised {e}")
    
    def test_repr(self):
        processor = BaseRewardProcessor()
        self.assertEqual(repr(processor), "BaseRewardProcessor()")


class TestSlidingWindowSmoother(unittest.TestCase):
    def test_initialization(self):
        smoother = SlidingWindowSmoother(window_size=50)
        self.assertEqual(smoother.window_size, 50)
        self.assertIsInstance(smoother.reward_history, deque)
        self.assertEqual(smoother.reward_history.maxlen, 50)
        self.assertEqual(smoother.current_smoothed_reward, 0.0)

    def test_initialization_invalid_window_size(self):
        with self.assertRaisesRegex(ValueError, "窗口大小必须是一个正整数"):
            SlidingWindowSmoother(window_size=0)
        with self.assertRaisesRegex(ValueError, "窗口大小必须是一个正整数"):
            SlidingWindowSmoother(window_size=-1)
        with self.assertRaisesRegex(ValueError, "窗口大小必须是一个正整数"):
            SlidingWindowSmoother(window_size=10.5)

    def test_process_filling_window(self):
        smoother = SlidingWindowSmoother(window_size=3)
        
        r1 = smoother.process(1.0)
        self.assertEqual(len(smoother.reward_history), 1)
        self.assertAlmostEqual(r1, 1.0)
        self.assertAlmostEqual(smoother.current_smoothed_reward, 1.0)

        r2 = smoother.process(2.0)
        self.assertEqual(len(smoother.reward_history), 2)
        self.assertAlmostEqual(r2, (1.0 + 2.0) / 2)
        self.assertAlmostEqual(smoother.current_smoothed_reward, 1.5)

        r3 = smoother.process(3.0)
        self.assertEqual(len(smoother.reward_history), 3)
        self.assertAlmostEqual(r3, (1.0 + 2.0 + 3.0) / 3)
        self.assertAlmostEqual(smoother.current_smoothed_reward, 2.0)

    def test_process_full_window_slides(self):
        smoother = SlidingWindowSmoother(window_size=3)
        smoother.process(1.0)
        smoother.process(2.0)
        smoother.process(3.0) # Window: [1,2,3], Mean: 2.0
        self.assertAlmostEqual(smoother.current_smoothed_reward, 2.0)

        r4 = smoother.process(4.0) # Window: [2,3,4]
        self.assertEqual(len(smoother.reward_history), 3)
        self.assertTrue(1.0 not in smoother.reward_history)
        self.assertAlmostEqual(r4, (2.0 + 3.0 + 4.0) / 3)
        self.assertAlmostEqual(smoother.current_smoothed_reward, 3.0)

        r5 = smoother.process(0.0) # Window: [3,4,0]
        self.assertAlmostEqual(r5, (3.0 + 4.0 + 0.0) / 3)
        self.assertAlmostEqual(smoother.current_smoothed_reward, 7.0/3.0)

    def test_reset(self):
        smoother = SlidingWindowSmoother(window_size=3)
        smoother.process(1.0)
        smoother.process(2.0)
        self.assertNotEqual(len(smoother.reward_history), 0)
        self.assertNotEqual(smoother.current_smoothed_reward, 0.0)

        smoother.reset()
        self.assertEqual(len(smoother.reward_history), 0)
        self.assertEqual(smoother.current_smoothed_reward, 0.0)
        
        # Ensure it works correctly after reset
        r1_after_reset = smoother.process(10.0)
        self.assertEqual(len(smoother.reward_history), 1)
        self.assertAlmostEqual(r1_after_reset, 10.0)
        self.assertAlmostEqual(smoother.current_smoothed_reward, 10.0)

    def test_repr_sliding(self):
        smoother = SlidingWindowSmoother(window_size=123)
        self.assertEqual(repr(smoother), "SlidingWindowSmoother(window_size=123)")


class TestMountainCarCustomReward(unittest.TestCase):
    def test_initialization_defaults(self):
        processor = MountainCarCustomReward()
        self.assertAlmostEqual(processor.goal_position, 0.5)
        self.assertAlmostEqual(processor.position_weight, 1.0)
        self.assertAlmostEqual(processor.velocity_weight, 0.1)
        self.assertIsNone(processor.previous_position)
        self.assertIsNone(processor.previous_velocity)

    def test_initialization_custom(self):
        processor = MountainCarCustomReward(
            goal_position=0.6, position_weight=2.0, velocity_weight=0.5
        )
        self.assertAlmostEqual(processor.goal_position, 0.6)
        self.assertAlmostEqual(processor.position_weight, 2.0)
        self.assertAlmostEqual(processor.velocity_weight, 0.5)

    def test_process_first_step(self):
        processor = MountainCarCustomReward(position_weight=1.0, velocity_weight=0.1)
        # MountainCar observation: [position, velocity]
        obs = [-0.5, 0.01] # pos = -0.5, vel = 0.01
        # First step, no previous_velocity, so no acceleration component
        expected_reward = 1.0 * (-0.5) # Only position component
        reward = processor.process(obs, dt=0.02)
        self.assertAlmostEqual(reward, expected_reward)
        self.assertAlmostEqual(processor.previous_position, -0.5)
        self.assertAlmostEqual(processor.previous_velocity, 0.01)

    def test_process_subsequent_step_with_acceleration(self):
        processor = MountainCarCustomReward(position_weight=1.0, velocity_weight=0.1)
        dt = 0.02
        # First step
        obs1 = [-0.5, 0.01]
        processor.process(obs1, dt=dt)
        
        # Second step
        obs2 = [-0.4, 0.02] # pos = -0.4, vel = 0.02
        # position component: 1.0 * (-0.4) = -0.4
        # acceleration component: (0.02 - 0.01) / 0.02 = 0.01 / 0.02 = 0.5
        # velocity_weight * acceleration = 0.1 * 0.5 = 0.05
        expected_reward = (1.0 * (-0.4)) + (0.1 * 0.5)
        reward = processor.process(obs2, dt=dt)
        self.assertAlmostEqual(reward, expected_reward)
        self.assertAlmostEqual(processor.previous_position, -0.4)
        self.assertAlmostEqual(processor.previous_velocity, 0.02)

    def test_process_no_dt_no_acceleration_component(self):
        processor = MountainCarCustomReward(position_weight=1.0, velocity_weight=0.1)
        # First step
        obs1 = [-0.5, 0.01]
        processor.process(obs1, dt=0.02) # dt provided here to set previous_velocity
        
        # Second step, but dt is None
        obs2 = [-0.4, 0.02]
        expected_reward = 1.0 * (-0.4) # Only position component
        reward = processor.process(obs2, dt=None)
        self.assertAlmostEqual(reward, expected_reward)

        # Second step, but dt is 0
        reward_dt_zero = processor.process(obs2, dt=0)
        self.assertAlmostEqual(reward_dt_zero, expected_reward)

    def test_reset(self):
        processor = MountainCarCustomReward()
        processor.process([-0.5, 0.01], dt=0.02)
        self.assertIsNotNone(processor.previous_position)
        self.assertIsNotNone(processor.previous_velocity)

        processor.reset()
        self.assertIsNone(processor.previous_position)
        self.assertIsNone(processor.previous_velocity)

    def test_repr_mountain_car_reward(self):
        processor = MountainCarCustomReward(goal_position=0.6, position_weight=1.5, velocity_weight=0.2)
        expected_repr = "MountainCarCustomReward(goal=0.6, pos_w=1.5, vel_w=0.2)"
        self.assertEqual(repr(processor), expected_repr)


if __name__ == "__main__":
    unittest.main() 