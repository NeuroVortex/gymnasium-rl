import unittest
from gymnasium_rl.environment.env import Environment
from gymnasium_rl.algorithms.monte_carlo.monte_carlo import MonteCarlo
from gymnasium_rl.algorithms.q_learning.q_learning import QLearning
from gymnasium_rl.algorithms.sarsa.sarsa import SARSA


class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        self.env = Environment('FrozenLake-v1', render=False)

    def test_monte_carlo_training(self):
        algo = MonteCarlo(self.env)
        algo.train(episode_num=10)
        self.assertGreater(len(algo.q_table), 0)

    def test_q_learning_training(self):
        algo = QLearning(env=self.env, learning_rate=0.1, discount_factor=0.9)
        algo.train(episode_num=10)
        self.assertGreater(len(algo.q_table), 0)

    def test_sarsa_training(self):
        algo = SARSA(env=self.env, learning_rate=0.1, discount_factor=0.9)
        algo.train(episode_num=10)
        self.assertGreater(len(algo.q_table), 0)


if __name__ == '__main__':
    unittest.main()
