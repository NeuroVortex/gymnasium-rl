class Agent:
    def __init__(self, action_space):
        self.__action_space = action_space

    def observe(self, state, action, reward, next_state, done):
        pass

    def act(self):
        pass
