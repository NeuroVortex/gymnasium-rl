import numpy as np
from gymnasium_rl.environment import Environment
from gymnasium_rl.interfaces import AbstractReinforcementLearningAlgo


class DoubleQLearning(AbstractReinforcementLearningAlgo):

    def __init__(self, learning_rate: float, discount_factor: float, env: Environment):
        self.__env = env
        self.__learning_rate = learning_rate
        self.__discount_factor = discount_factor
        self.__random_generator = np.random.default_rng(seed=42)
        self.__random_action_reward = []
        self.__q_table = np.zeros((2, self.__env.states, self.__env.action_space.n))

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
        table_index = self.__random_generator.integers(2)
        max_q_table_index = np.argmax(self.__q_table[table_index][state])
        self.__q_table[table_index][state, action] = (
                (1 - self.__learning_rate) * self.__q_table[state, action] +
                self.__learning_rate * (
                        reward + self.__discount_factor * self.__q_table[1 - table_index][next_state][
                    max_q_table_index]))

    @property
    def q_table(self):
        return (self.__q_table[0] + self.__q_table[1]) / 2
