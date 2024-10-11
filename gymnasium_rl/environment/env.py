import gymnasium as gym
from matplotlib import pyplot as plt
import logging

logger = logging.getLogger(__name__)


class Environment:
    def __init__(self, env_id: str, render: bool = True, render_mode: str = 'human') -> None:
        self.__states = 0
        self.__episode = []
        self.__render = render
        self.__create_env(env_id, render_mode)

    def __create_env(self, env_id: str, render_mode):
        self.__env = gym.make(env_id, render_mode=render_mode)
        self.__env.metadata['render_fps'] = 15
        self.__init_state, _ = self.reset()  # we set seed to enable reproduction of result capability
        self.__states = (self.__env.observation_space.shape
                         if bool(self.__env.observation_space.shape) else self.__env.observation_space.n)
        print(f"The number of actions is {self.__env.action_space.n}")
        print(f"The number of states is {self.__states}")

    def reset(self, seed=42):
        return self.__env.reset(seed=seed)

    def step(self, action):
        return self.__env.step(action)

    def perform_action(self, action):
        return self.__env.step(action)

    def observe(self, current_state, action):
        _, next_state, reward, terminal = self.__env.unwrapped.P[current_state][action][0]
        return next_state, reward, terminal

    @property
    def observation_space(self):
        return self.__env.observation_space

    @property
    def action_space(self):
        return self.__env.action_space

    @property
    def states(self):
        return self.__states

    @property
    def is_render_active(self):
        return self.__render

    def render(self):
        # snapshot = self.__env.render()
        # plt.imshow(snapshot)
        # plt.show()
        self.__env.render()

    def close(self):
        self.__env.close()
