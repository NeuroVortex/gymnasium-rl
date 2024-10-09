import pytest
from gymnasium_rl.environment.env import Environment


@pytest.fixture
def env():
    return Environment('FrozenLake-v1', render=False)


def test_env_initialization(env):
    assert env.states == 16  # Assuming FrozenLake has 16 states
    assert env.action_space.n == 4  # Assuming 4 actions in FrozenLake


def test_env_reset(env):
    state, _ = env.reset()
    assert state is not None


def test_env_step(env):
    state, _, _, _, _ = env.step(0)
    assert state in range(env.states)
