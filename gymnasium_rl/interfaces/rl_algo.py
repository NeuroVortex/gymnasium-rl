from abc import ABC, abstractmethod


class AbstractReinforcementLearningAlgo(ABC):

    @property
    @abstractmethod
    def q_table(self):
        raise NotImplementedError

    @abstractmethod
    def train(self, episode_num: int, seed=42):
        raise NotImplementedError
