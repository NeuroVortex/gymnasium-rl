import pytest
from gymnasium_rl.environment.env import Environment
from gymnasium_rl.algorithms.monte_carlo.monte_carlo import MonteCarlo
from gymnasium_rl.algorithms.q_learning.q_learning import QLearning
from gymnasium_rl.algorithms.sarsa.sarsa import SARSA


@pytest.fixture
def env():
    return Environment('FrozenLake-v1', render=False)


def test_monte_carlo_training(env):
    algo = MonteCarlo(env)
    algo.train(episode_num=10)
    assert len(algo.q_table) > 0


def test_q_learning_training(env):
    algo = QLearning(env=env, learning_rate=0.1, discount_factor=0.9)
    algo.train(episode_num=10)
    assert len(algo.q_table) > 0


def test_sarsa_training(env):
    algo = SARSA(env=env, learning_rate=0.1, discount_factor=0.9)
    algo.train(episode_num=10)
    assert len(algo.q_table) > 0
