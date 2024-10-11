import numpy as np

from gymnasium_rl.agent.policy import Policy
from gymnasium_rl.environment import Environment
from gymnasium_rl.interfaces.rl_algo import AbstractReinforcementLearningAlgo


class SARSA(AbstractReinforcementLearningAlgo):

    def __init__(self, env: Environment, policy: Policy, learning_rate: float, discount_factor: float):
        self.__learning_rate = learning_rate
        self.__discount_factor = discount_factor
        self.__policy = policy
        self.__env = env
        self.__q_table = np.zeros((self.__env.states, self.__env.action_space.n))

    def generate_episode(self, seed=42):
        state, _ = self.__env.reset(seed)
        action = self.__policy.get_action(self.__q_table, state)
        terminated = False
        episode_reward = 0

        while not terminated:
            next_state, reward, terminated, _, _ = self.__env.step(action)
            next_action = self.__policy.get_action(self.__q_table, next_state)
            self.__update_q_table(state, action, reward, next_state, next_action)
            episode_reward += reward
            state, action = next_state, next_action

            if self.__env.is_render_active:
                self.__env.render()

        return episode_reward

    def __update_q_table(self, state, action, reward, next_state, next_action):
        self.__q_table[state, action] = (
                (1 - self.__learning_rate) * self.__q_table[state, action] +
                self.__learning_rate * (reward + self.__discount_factor * self.__q_table[next_state, next_action]))

    @property
    def q_table(self):
        return self.__q_table
