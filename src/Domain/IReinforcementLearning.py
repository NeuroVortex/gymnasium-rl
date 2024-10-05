from abc import ABC, abstractmethod


class IReinforcementLearning(ABC):

    @property
    @abstractmethod
    def q_table(self):
        raise NotImplementedError

    def get_optimal_policy(self):
        raise NotImplementedError

    def train(self, episode_num: int, seed):
        raise NotImplementedError
