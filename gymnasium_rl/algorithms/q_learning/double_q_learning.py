import numpy as np

from gymnasium_rl.agent.policy import Policy
from gymnasium_rl.environment import Environment
from gymnasium_rl.interfaces import AbstractReinforcementLearningAlgo


class DoubleQLearning(AbstractReinforcementLearningAlgo):

    def __init__(self, env: Environment, policy: Policy, learning_rate: float, discount_factor: float):
        self.__policy = policy
        self.__env = env
        self.__learning_rate = learning_rate
        self.__discount_factor = discount_factor
        self.__random_generator = np.random.default_rng(seed=42)
        self.__q_table = np.zeros((2, self.__env.states, self.__env.action_space.n))

    def generate_episode(self, seed=42):
        state, _ = self.__env.reset(seed)
        terminated = False
        episode_reward = 0

        while not terminated:
            action = self.__policy.get_action(self.__q_table, state)
            next_state, reward, terminated, _, _ = self.__env.step(action)
            self.__update_q_table(state, action, reward, next_state)
            episode_reward += reward
            state = next_state

        return episode_reward

    def __update_q_table(self, state, action, reward, next_state):
        table_index = self.__random_generator.integers(2)
        max_q_table_index = np.argmax(self.__q_table[table_index][state])
        self.__q_table[table_index][state, action] = (
                (1 - self.__learning_rate) * self.__q_table[table_index][state, action] +
                self.__learning_rate * (
                        reward + self.__discount_factor * self.__q_table[1 - table_index][next_state][max_q_table_index]
                )
        )

    @property
    def q_table(self):
        return (self.__q_table[0] + self.__q_table[1]) / 2
