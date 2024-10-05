import numpy as np
from src.Application.environment.env import Environment
from src.Domain.IReinforcementLearning import IReinforcementLearning


class QLearning(IReinforcementLearning):

    def __init__(self, learning_rate: float, discount_factor: float, env: Environment):
        self.__env = env
        self.__learning_rate = learning_rate
        self.__discount_factor = discount_factor
        self.__random_action_reward = []
        self.__q_table = np.zeros((self.__env.states, self.__env.action_space.n))

    def train(self, episode_num: int, seed=42):
        self.__random_action_reward = []
        for _ in range(episode_num):
            self.__generate_episode(seed)

    def __generate_episode(self, seed=42):
        state, _ = self.__env.reset(seed)
        terminated = False
        episode_reward = 0

        while not terminated:
            action = self.__env.action_space.sample()
            next_state, reward, terminated, _, _ = self.__env.step(action)
            self.__update_q_table(state, action, reward, next_state)
            episode_reward += reward
            state = next_state

            if self.__env.is_render_active:
                self.__env.render()

            self.__random_action_reward.append(episode_reward)

    def __update_q_table(self, state, action, reward, next_state):
        self.__q_table[state, action] = (
                (1 - self.__learning_rate) * self.__q_table[state, action] +
                self.__learning_rate * (reward + self.__discount_factor * np.max(self.__q_table[next_state])))

    @property
    def q_table(self):
        return self.__q_table
