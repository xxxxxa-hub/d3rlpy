import numpy as np
import pdb
import time
from ..envs import GymEnv
from ..interface import QLearningAlgoProtocol, StatefulTransformerAlgoProtocol
from collections import defaultdict

__all__ = [
    "evaluate_qlearning_with_environment",
    "evaluate_transformer_with_environment",
]


def evaluate_qlearning_with_environment(
    algo: QLearningAlgoProtocol,
    env: GymEnv,
    n_trials: int = 10,
    epsilon: float = 0.0,
    gamma: float = 1.0
) -> float:
    """Returns average environment score.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.utility import evaluate_with_environment

        env = gym.make('CartPole-v0')

        cql = CQL()

        mean_episode_return = evaluate_with_environment(cql, env)


    Args:
        alg: algorithm object.
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.

    Returns:
        average score.
    """
    episode_rewards_1 = []
    episode_rewards = []
    transitions = defaultdict(list)

    for _ in range(n_trials):
        observation, _ = env.reset()
        episode_reward_1 = 0.0
        episode_reward = 0.0
        t = 0
        transition = defaultdict(list)

        while True:
            # take action
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = algo.predict(np.expand_dims(observation, axis=0))[0]

            # store observation, action
            transition["actions"].append(action)
            transition["observations"].append(observation)
            observation, reward, done, truncated, _ = env.step(action)
            episode_reward_1 += float(reward)
            episode_reward += gamma**t * float(reward)

            transition["rewards"].append(reward)
            transition["terminals"].append(1.0 if done else 0.0)
            transition["timeouts"].append(truncated)
            t += 1

            if done or truncated:
                break
        transition["actions"] = np.array(transition["actions"],dtype=np.float32)
        transition["observations"] = np.array(transition["observations"],dtype=np.float32)
        transition["rewards"] = np.array(transition["rewards"],dtype=np.float32)
        transition["terminals"] = np.array(transition["terminals"],dtype=np.float32)
        transition["timeouts"] = np.array(transition["timeouts"])

        transitions["actions"].append(transition["actions"])
        transitions["observations"].append(transition["observations"])
        transitions["rewards"].append(transition["rewards"])
        transitions["terminals"].append(transition["terminals"])
        transitions["timeouts"].append(transition["timeouts"])
        episode_rewards_1.append(episode_reward_1)
        episode_rewards.append(episode_reward)

    transitions["actions"] = np.concatenate(transitions["actions"],dtype=np.float32)
    transitions["observations"] = np.concatenate(transitions["observations"],dtype=np.float32)
    transitions["rewards"] = np.concatenate(transitions["rewards"],dtype=np.float32)
    transitions["terminals"] = np.concatenate(transitions["terminals"],dtype=np.float32)
    transitions["timeouts"] = np.concatenate(transitions["timeouts"])
    
    return float(np.mean(episode_rewards_1)), float(np.std(episode_rewards_1)), float(np.mean(episode_rewards)), float(np.std(episode_rewards)), transitions


def evaluate_transformer_with_environment(
    algo: StatefulTransformerAlgoProtocol,
    env: GymEnv,
    n_trials: int = 10,
) -> float:
    """Returns average environment score.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.utility import evaluate_with_environment

        env = gym.make('CartPole-v0')

        cql = CQL()

        mean_episode_return = evaluate_with_environment(cql, env)


    Args:
        alg: algorithm object.
        env: gym-styled environment.
        n_trials: the number of trials.

    Returns:
        average score.
    """
    episode_rewards = []
    for _ in range(n_trials):
        algo.reset()
        observation, reward = env.reset()[0], 0.0
        episode_reward = 0.0

        while True:
            # take action
            action = algo.predict(observation, reward)

            '''
            observation, _reward, done, truncated, _ = env.step(action)
            reward = float(_reward)
            episode_reward += reward
            '''

            observation, _reward, done, _ = env.step(action)
            reward = float(_reward)
            episode_reward += reward

            if done:
                break
        episode_rewards.append(episode_reward)
    return float(np.mean(episode_rewards))
