
# GymnasiumRL

GymnasiumRL is a Python-based reinforcement learning project designed to demonstrate the implementation and comparison of various RL algorithms using the `Gymnasium` environment. The project includes algorithms such as Monte Carlo, Q-Learning, SARSA, and Double Q-Learning. The agent interacts with different environments, like the FrozenLake, to learn optimal policies for decision-making tasks.

## Project Components

### 1. `gymnasium_rl/` - Primary Package
- **Agent (`gymnasium_rl/agent/agent.py`)**: Contains the agent responsible for training and interacting with the environment.
- **Environment (`gymnasium_rl/environment/env.py`)**: A wrapper around the Gymnasium environment, providing state transitions, rendering, and other environment-related functionalities.
- **Contracts (`gymnasium_rl/contracts/rl_type.py`)**: Defines the types of reinforcement learning algorithms used within the agent.
- **Algorithms (`gymnasium_rl/algorithms/`)**: Includes implementations of various reinforcement learning algorithms:
    - **Monte Carlo (`monte_carlo/monte_carlo.py`)**
    - **Q-Learning (`q_learning/q_learning.py`)**
    - **Double Q-Learning (`q_learning/double_q_learning.py`)**
    - **SARSA (`sarsa/sarsa.py`)**
    - **Expected SARSA (`sarsa/expected_sarsa.py`)**
    - **Model-based RL (`model_based/model_based_toolbox.py`)**
- **Interfaces (`gymnasium_rl/interfaces/rl_algo.py`)**: Shared interfaces for reinforcement learning algorithms.

### 2. `tests/` - Testing
- **Pytest (`tests/pytest/`)**: Test suite for testing the core functionalities of the RL algorithms and agent using pytest.
- **Unittest (`tests/unittest/`)**: Provides an alternative test suite using unittest framework for those preferring unittest.

### 3. `scripts/` - Main Application
- **Main (`scripts/main.py`)**: Entry point for running the agent and environment for training and testing purposes.

### 4. Other Files
- **`requirements.txt`**: Contains the Python dependencies needed for the project.
- **`setup.py`**: Used for installing the package if it is to be distributed.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/neurovortex/GymnasiumRL.git
   cd GymnasiumRL
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:
   ```bash
   python scripts/main.py
   ```

## Usage

You can modify the RL algorithm or environment by adjusting the parameters in `main.py`. For example, to switch to the Double Q-Learning algorithm:
```python
from gymnasium_rl.contracts import RLType

agent = Agent(env, rl_type=RLType.DoubleQLearning)
```

## Supported Algorithms
- Monte Carlo (First-visit and Every-visit)
- Q-Learning
- Double Q-Learning
- SARSA
- Expected SARSA
- Model-based Reinforcement Learning

## Example: FrozenLake Environment

This project includes a demonstration using the `FrozenLake-v1` environment from Gymnasium. The agent learns to navigate a frozen lake by avoiding holes and reaching a goal.

## Future Work
- Expand to more environments.
- Implement more advanced algorithms like DDPG and PPO.
- Add hyperparameter optimization.

## License
This project is licensed under the MIT License.
