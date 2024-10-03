from src.Application.environment.env import create_env
from src.Application.environment.visualization import render


if __name__ == '__main__':
    env, init_state = create_env('MountainCar-v0')
    print('Environment is initialized')
    print('car positioned at {} (m) on the x-axis'.format(init_state[0]))
    print('car initial {} (m/sec)'.format(init_state[1]))
    render(env)