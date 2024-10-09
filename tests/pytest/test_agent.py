import pytest
from gymnasium_rl.agent.agent import Agent
from gymnasium_rl.environment.env import Environment
from gymnasium_rl.contracts.rl_type import RLType


@pytest.fixture
def agent():
    env = Environment('FrozenLake-v1', render=False)
    return Agent(env, rl_type=RLType.QLearning)


def test_agent_initialization(agent):
    assert agent.policy is not None


def test_agent_training(agent):
    agent.train()
    assert len(agent.policy) > 0