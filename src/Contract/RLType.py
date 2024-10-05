from enum import IntEnum


class RLType(IntEnum):
    MonteCarlo = 0
    QLearning = 1
    DoubleQLearning = 2
    SARSA = 3
    ExpectedSARSA = 4
