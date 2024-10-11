from gymnasium_rl.agent.policy import Policy
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
                 discount_factor: float = 0.99,
                 learning_rate: float = 0.1,
                 epsilon: float = 1,
                 epsilon_decay: float = 0.999):
        self.__env = env
        self.__episode_num = episode_num
        self.__rl_func: AbstractReinforcementLearningAlgo | None = None
        self.__policy = Policy(env=self.__env, epsilon=epsilon, epsilon_decay=epsilon_decay)
        self.__initialize(rl_type, learning_rate, discount_factor)
        self.__episode_rewards = []
        self.__cumulative_reward = 0

    def __initialize(self, rl_type, learning_rate, discount_factor):
        match rl_type:
            case RLType.MonteCarlo:
                self.__rl_func = MonteCarlo(self.__env)

            case RLType.QLearning:
                self.__rl_func = QLearning(env=self.__env, policy=self.__policy, learning_rate=learning_rate,
                                           discount_factor=discount_factor)

            case RLType.SARSA:
                self.__rl_func = SARSA(env=self.__env, policy=self.__policy, learning_rate=learning_rate,
                                       discount_factor=discount_factor)

            case RLType.ExpectedSARSA:
                self.__rl_func = ExpectedSARSA(env=self.__env, policy=self.__policy, learning_rate=learning_rate,
                                               discount_factor=discount_factor)

            case RLType.DoubleQLearning:
                self.__rl_func = DoubleQLearning(env=self.__env, policy=self.__policy, learning_rate=learning_rate,
                                                 discount_factor=discount_factor)

    def train(self, seed=42):
        self.__episode_rewards = []
        self.__cumulative_reward = 0

        for _ in range(self.__episode_num):
            episode_reward = self.__rl_func.generate_episode(seed)

            self.__episode_rewards.append(episode_reward)
            self.__cumulative_reward += episode_reward

            print("Cumulative Reward", self.__cumulative_reward)

            if self.__env.is_render_active:
                self.__env.render()

    @property
    def policy(self):
        return self.__policy.optimal_policy
