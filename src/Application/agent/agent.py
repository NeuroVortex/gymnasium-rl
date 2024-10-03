from src.Application.environment.env import Environment
from src.Domain.model_based.model_based_toolbox import ModelBasedToolbox


class Agent:
    def __init__(self, env: Environment, discount_factor: float, learning_rate: float):
        self.__env = env
        self.__initialize_policy()
        self.__learning_rate = learning_rate
        self.__rl_toolbox = ModelBasedToolbox(env=self.__env, policy=self.__policy, discount_factor=discount_factor)

    def __initialize_policy(self):
        self.__policy = {state: 0 for state in range(self.__env.observation_space.shape[0])}

    def act(self):
        self.__rl_toolbox.run()
