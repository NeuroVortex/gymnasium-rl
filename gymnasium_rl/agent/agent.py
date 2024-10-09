import numpy as np

from gymnasium_rl.environment.env import Environment
from gymnasium_rl.interfaces import AbstractReinforcementLearningAlgo
from gymnasium_rl.algorithms.monte_carlo import MonteCarlo
from gymnasium_rl.algorithms.q_learning import QLearning, DoubleQLearning
from gymnasium_rl.algorithms.sarsa import ExpectedSARSA, SARSA
from gymnasium_rl.contracts.rl_type import RLType


class Agent:
    def __init__(self, env: Environment,
                 episode_num: int = 1000,
                 rl_type: RLType = RLType.QLearning,
                 discount_factor: float = 0.9,
                 learning_rate: float = 0.9):
        self.__env = env
        self.__episode_num = episode_num
        self.__rl_func: AbstractReinforcementLearningAlgo | None = None
        self.__optimal_policy = np.zeros((env.states, env.action_space.n))
        self.__initialize(rl_type, learning_rate, discount_factor)

    def __initialize(self, rl_type, learning_rate, discount_factor):
        match rl_type:
            case RLType.MonteCarlo:
                self.__rl_func = MonteCarlo(self.__env)

            case RLType.QLearning:
                self.__rl_func = QLearning(env=self.__env, learning_rate=learning_rate,
                                           discount_factor=discount_factor)

            case RLType.SARSA:
                self.__rl_func = SARSA(env=self.__env, learning_rate=learning_rate,
                                       discount_factor=discount_factor)

            case RLType.ExpectedSARSA:
                self.__rl_func = ExpectedSARSA(env=self.__env, learning_rate=learning_rate,
                                               discount_factor=discount_factor)

            case RLType.DoubleQLearning:
                self.__rl_func = DoubleQLearning(env=self.__env, learning_rate=learning_rate,
                                                 discount_factor=discount_factor)

    def train(self):
        self.__rl_func.train(self.__episode_num)
        self.__optimal_policy = {state: np.argmax(self.__rl_func.q_table[state]) for state in range(self.__env.states)}

    @property
    def policy(self):
        return self.__optimal_policy
