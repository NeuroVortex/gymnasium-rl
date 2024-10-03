from src.Application.environment.env import Environment


class ModelBasedToolbox:
    def __init__(self, env: Environment, discount_factor, policy):
        self.__env = env
        self.__discount_factor = discount_factor
        self.__policy = policy
        self.__q_value = {}
        self.__state_value = {}
        self.__threshold = 0.001

    def __initialize_state_value(self):
        self.__state_value = {state: self.calculate_state_value(state)
                              for state in range(self.__env.observation_space.shape[0])}

    def calculate_state_value(self, current_state):
        action = self.__policy.get(current_state)
        next_state, reward, terminal = self.__env.observe(current_state, action)

        if current_state == terminal:
            return 0

        return reward + self.__discount_factor * self.calculate_state_value(next_state)

    def calculate_action_value(self, current_state, action):
        next_state, reward, terminal = self.__env.observe(current_state, action)

        if terminal:
            return None

        return reward + self.__discount_factor * self.__state_value[next_state]

    def calculate_q_value(self):
        self.__q_value = {}
        self.__q_value = {(state, action): self.calculate_action_value(state, action)
                          for state in range(self.__env.observation_space.shape[0])
                          for action in range(self.__env.action_space.n)}

    def improve_policy(self):
        improved_policy = {}
        self.calculate_q_value()

        for state in range(self.__env.observation_space.shape[0]-1):
            max_q_value_action = max(range(self.__env.action_space.n), key=lambda action: self.__q_value[(state, action)])
            improved_policy.update({state: max_q_value_action})

        self.__policy = improved_policy

    def __update_state_value(self):
        improved_state_value = {state: self.__q_value[(state, self.__policy[state])]
                                for state in range(self.__env.observation_space.shape[0])}
        return improved_state_value

    def run(self):

        while True:
            self.improve_policy()
            improved_state_value = self.__update_state_value()

            if all(abs(improved_state_value[state] - self.__state_value[state]) < self.__threshold
                   for state in self.__state_value):
                break

            self.__state_value = improved_state_value
