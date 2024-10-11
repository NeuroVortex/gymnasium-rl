from gymnasium_rl.agent import Agent
from gymnasium_rl.contracts import RLType
from gymnasium_rl.environment import Environment

if __name__ == '__main__':
    env = Environment('FrozenLake-v1')
    agent = Agent(env, rl_type=RLType.DoubleQLearning)
    agent.train()
    env.close()
