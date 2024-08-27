from environment.env import create_env


class FrozenLake:
    def __init__(self):
        self.__env, self.__init_state = create_env('CliffWalking-v0')
        self.__actions = [0, 1, 2, 3]  # 0: left; 1: down; 2: right; 3: up

    def run(self):
        for action in self.__actions:
            state, reward, done, truncate, info = self.__env.step(action)
            # print(f"Probability of action {action} in state {state} is "
            #       f"{self.__env.unwrapped.P[state][action][0]}")
            # print(f"next state of action {action} in state {state} is "
            #       f"{self.__env.unwrapped.P[state][action][1]}")
            # print(f"reward of action {action} in state {state} is "
            #       f"{self.__env.unwrapped.P[state][action][2]}")
            # print(f"is done {self.__env.unwrapped.P[state][action][3]}")

            # Agent in cell state 6 deciding to move left with action 0
            # env.unwrapped.P[6][0] --> list of tuple of potential outcomes
            # [(0.333, 2, 0.0, False), (0.333, 5, 0.0, True), (0.333, 10, 0.0, False)]
            # ex: the agent has a 33% chance of ending up in each of the cell 2, 5, or
            # 10 after executing the left action (and 5 is the termination of game) because
            # it is True (maybe lost or win bot game done)

            if done:
                print('terminate')


if __name__ == '__main__':
    rl_prob = FrozenLake()
    rl_prob.run()
