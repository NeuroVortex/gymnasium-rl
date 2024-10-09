import unittest
from gymnasium_rl.environment.env import Environment


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = Environment('FrozenLake-v1', render=False)

    def test_env_initialization(self):
        self.assertEqual(self.env.states, 16)  # Assuming FrozenLake has 16 states
        self.assertEqual(self.env.action_space.n, 4)  # Assuming 4 actions in FrozenLake

    def test_env_reset(self):
        state, _ = self.env.reset()
        self.assertIsNotNone(state)

    def test_env_step(self):
        state, _, _, _, _ = self.env.step(0)
        self.assertIn(state, range(self.env.states))


if __name__ == '__main__':
    unittest.main()
