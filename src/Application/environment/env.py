import gymnasium as gym
import matplotlib as plt
import logging

logger = logging.getLogger(__name__)


class Environment:
    def __init__(self, env_id: str, render_mode: str = 'rgb_array') -> None:
        self.__create_env(env_id, render_mode)

    def __create_env(self, env_id: str, render_mode):
        self.__env = gym.make(env_id, render_mode=render_mode)
        self.__init_state, _ = self.__env.reset(seed=42)  # we set seed to enable reproduction of result capability

        print(f"The number of actions is {self.__env.action_space.n}")
        print(f"The number of states is {self.__env.observation_space.shape}")

    @property
    def observation_space(self):
        return self.__env.observation_space

    @property
    def action_space(self):
        return self.__env.action_space

    def render(self):
        snapshot = self.__env.render()
        plt.imshow(snapshot)
        plt.show()
