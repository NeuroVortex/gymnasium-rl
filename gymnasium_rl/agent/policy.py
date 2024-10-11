import numpy as np

from gymnasium_rl.environment import Environment


class Policy:
    def __init__(self, env: Environment, epsilon: float = 1, epsilon_decay: float = 0.999):
        self.__eps = epsilon
        self.__eps_decay = epsilon_decay
        self.__min_eps = 0.001
        self.__env = env
        self.__optimal_policy = np.zeros((env.states, env.action_space.n))
        self.__q_table = {}

    def get_action(self, q_table, state):
        self.__q_table = q_table

        if np.random.rand() < self.__eps:
            action = self.__env.action_space.sample()

        else:
            action = np.argmax(q_table[state, :])

        return action

    def __update_epsilon(self):
        self.__eps = max(self.__min_eps, self.__eps * self.__eps_decay)

    @property
    def optimal_policy(self):
        self.__optimal_policy = {state: np.argmax(self.__q_table[state]) for state in range(self.__env.states)}
        return self.__optimal_policy
