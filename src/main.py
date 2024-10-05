from src.Application.agent.agent import Agent
from src.Application.environment.env import Environment


if __name__ == '__main__':
    env = Environment('FrozenLake-v1')
    agent = Agent(env)
    agent.train()
