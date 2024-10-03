import numpy as np

from src.Application.environment.env import create_env
from src.Application.environment.visualization import render


class CartPole:
    def __init__(self, discount_factor: float, learning_rate: float):
        self.__discount_factor = discount_factor
        self.__learning_rate = learning_rate
        self.__env, self.__init_state = create_env('CartPole-v1')
        self.__reward = - np.inf
        self.__actions = list(range(self.__env.action_space.n))

    def run(self):

        terminate = False

        while not terminate:
            state, info = self.__env.reset(seed=42)
            action = self.__choose_action(state)
            state, reward, done, truncate, info = self.__execute()
            render(self.__env)

            if done:
                print('terminate')

    def __execute(self, action):
        return self.__env.step(action)

    def __choose_action(self, state):
        pass

    def __update_knowledge(self, state, action, reward):
        pass

    def __calculate_reward(self, expected_rewards):
        return np.sum(np.array([self.__discount_factor ** i for i in range(len(expected_rewards))]))