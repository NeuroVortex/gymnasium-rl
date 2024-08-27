from environment.env import create_env


class CartPole:
    def __init__(self):
        self.__env, self.__init_state = create_env('CartPole-v0')
        self.__actions = [0, 1]  # 0: left; 1: right

    def run(self):
        for action in self.__actions:
            state, reward, done, truncate, info = self.__env.step(action)

            if done:
                print('terminate')


