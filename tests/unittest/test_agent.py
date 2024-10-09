
import unittest
from gymnasium_rl.agent.agent import Agent
from gymnasium_rl.environment.env import Environment
from gymnasium_rl.contracts.rl_type import RLType


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.env = Environment('FrozenLake-v1', render=False)
        self.agent = Agent(self.env, rl_type=RLType.QLearning)

    def test_agent_initialization(self):
        self.assertIsNotNone(self.agent.policy)

    def test_agent_training(self):
        self.agent.train()
        self.assertGreater(len(self.agent.policy), 0)


if __name__ == '__main__':
    unittest.main()

