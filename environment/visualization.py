import matplotlib.pyplot as plt
import gymnasium as gym


def render(env: gym.Env):
    snapshot = env.render()
    plt.imshow(snapshot)
    plt.show()
