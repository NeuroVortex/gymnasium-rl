from abc import ABC, abstractmethod


class AbstractReinforcementLearningAlgo(ABC):

    @property
    @abstractmethod
    def q_table(self):
        raise NotImplementedError

    @abstractmethod
    def generate_episode(self, seed=42):
        raise NotImplementedError
