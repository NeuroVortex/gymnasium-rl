import gymnasium as gym
import logging

logger = logging.getLogger(__name__)


def create_env(env_id: str, render_mode='rgb_array'):
    env = gym.make(env_id, render_mode=render_mode)
    init_state, _ = env.reset(seed=42)  # we set seed to enable reproduction of result capability

    print(f"The number of actions is {env.action_space.n}")
    print(f"The number of states is {env.observation_space.shape}")
    return env, init_state
