import numpy as np

from environment.env import create_env
from environment.visualization import render


class FrozenLake:

    def __init__(self, discount_factor: float, learning_rate: float):
        self.__discount_factor = discount_factor
        self.__learning_rate = learning_rate
        self.__env, self.__init_state = create_env('FrozenLake-v1')
        self.__rewards = np.array([])
        self.__actions = list(range(self.__env.action_space.n))

    def run(self):

        is_terminal = False

        while not is_terminal:
            state, _ = self.__env.reset(seed=42)

            action = self.__choose_action(state)

            # Agent in cell state 6 deciding to move left with action 0
            # env.unwrapped.P[6][0] --> list of tuple of potential outcomes
            # [(0.333, 2, 0.0, False), (0.333, 5, 0.0, True), (0.333, 10, 0.0, False)]
            # ex: the agent has a 33% chance of ending up in each of the cell 2, 5, or
            # 10 after executing the left action (and 5 is the termination of game) because
            # it is True (maybe lost or win bot game done)
            # (0.333, 2, 0.0, False)
            # print(f"Probability: {probability}, Next State: {next_state}, Reward: {reward}, Done: {done}")

            print("Transitional Probability")
            print(f"Probable action {action},"
                  f"Probability: {self.__env.unwrapped.P[state][action][0]}, "
                  f"next state: {self.__env.unwrapped.P[state][action][1]},"
                  f"reward: {self.__env.unwrapped.P[state][action][2]},"
                  f"terminated:{self.__env.unwrapped.P[state][action][3]}")
            state, reward, is_terminal, _, _ = self.__execute(action)
            np.append(self.__rewards, reward)
            print(f"current action: {action}, current state {state}")
            print(f'Cumulative Reward: {self.__calculate_reward(reward)}')
            render(self.__env)

            if is_terminal:
                print('terminate')
                break

    def __execute(self, action):
        return self.__env.step(action)

    def __choose_action(self, state):
        pass

    def __update_knowledge(self, state, action, reward):
        pass

    def __calculate_reward(self, expected_rewards):
        return np.sum(np.array([self.__discount_factor ** i for i in range(len(expected_rewards))]))


if __name__ == '__main__':
    rl_prob = FrozenLake(0.9, 0.5)
    rl_prob.run()
