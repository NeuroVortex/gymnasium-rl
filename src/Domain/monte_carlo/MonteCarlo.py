import numpy as np

from src.Application.environment.env import Environment


class MonteCarlo:
    def __init__(self, env: Environment, every_visit: bool = False):
        self.__env = env
        self.__every_visit = every_visit
        self.__q_table = np.zeros((env.states, env.action_space))

    def train(self, episode_num, seed=42):
        self.every_visit_mc(episode_num, seed) if self.__every_visit else self.first_visit_mc(episode_num, seed)

    def __generate_episode(self, seed=42):
        episode = []
        state, _ = self.__env.reset(seed)
        terminated = False

        while not terminated:
            action = self.__env.action_space.sample()
            next_state, reward, terminated, _, _ = self.__env.step(action)
            episode.append((state, action, reward))
            state = next_state

            if self.__env.is_render_active:
                self.__env.render()

        return episode

    def every_visit_mc(self, episode_num, seed=42):
        self.__q_table = np.zeros((self.__env.states, self.__env.action_space))
        returns_sum = np.zeros((self.__env.states, self.__env.action_space))
        returns_count = np.zeros((self.__env.states, self.__env.action_space))

        for _ in range(episode_num):
            episode = self.__generate_episode(seed)

            for episode_index, (state, action, reward) in enumerate(episode):
                returns_sum[state, action] += sum(visited[2] for visited in episode[episode_index:])
                returns_count[state, action] += 1

        non_zero_counts = returns_count != 0
        self.__q_table[non_zero_counts] = returns_sum[non_zero_counts] / returns_count[non_zero_counts]

    def first_visit_mc(self, episode_num, seed=42):

        self.__q_table = np.zeros((self.__env.states, self.__env.action_space))
        returns_sum = np.zeros((self.__env.states, self.__env.action_space))
        returns_count = np.zeros((self.__env.states, self.__env.action_space))

        for _ in range(episode_num):
            episode = self.__generate_episode(seed)
            visited_states_action = set()

            for episode_index, (state, action, reward) in enumerate(episode):
                if (state, action) not in visited_states_action:
                    returns_sum[state, action] += sum(visited[2] for visited in episode[episode_index:])
                    returns_count[state, action] += 1
                    visited_states_action.add((state, action))

        non_zero_counts = returns_count != 0
        self.__q_table[non_zero_counts] = returns_sum[non_zero_counts] / returns_count[non_zero_counts]

    @property
    def q_table(self):
        return self.__q_table