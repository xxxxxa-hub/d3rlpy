# pylint: disable=unused-import,too-many-return-statements

import os
import random
import re
from typing import Any, Dict, Optional, Tuple
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader
import gym
import numpy as np
from gym.wrappers.time_limit import TimeLimit
import h5py
import pdb

from . import (
    BasicTrajectorySlicer,
    BasicTransitionPicker,
    Episode,
    EpisodeGenerator,
    FrameStackTrajectorySlicer,
    FrameStackTransitionPicker,
    InfiniteBuffer,
    MDPDataset,
    ReplayBuffer,
    TrajectorySlicerProtocol,
    TransitionPickerProtocol,
    create_infinite_replay_buffer,
)
from ..envs import ChannelFirst, FrameStack
from ..logging import LOG
from ..types import NDArray, UInt8NDArray

__all__ = [
    "DATA_DIRECTORY",
    "DROPBOX_URL",
    "CARTPOLE_URL",
    "CARTPOLE_RANDOM_URL",
    "PENDULUM_URL",
    "PENDULUM_RANDOM_URL",
    "D4rlDataset",
    "infinite_loader",
    "get_cartpole",
    "get_pendulum",
    "get_atari",
    "get_atari_transitions",
    "get_d4rl",
    "get_dataset",
]

DATA_DIRECTORY = "d3rlpy_data"
DROPBOX_URL = "https://www.dropbox.com/s"
CARTPOLE_URL = f"{DROPBOX_URL}/uep0lzlhxpi79pd/cartpole_v1.1.0.h5?dl=1"
CARTPOLE_RANDOM_URL = f"{DROPBOX_URL}/4lgai7tgj84cbov/cartpole_random_v1.1.0.h5?dl=1"  # pylint: disable=line-too-long
PENDULUM_URL = f"{DROPBOX_URL}/ukkucouzys0jkfs/pendulum_v1.1.0.h5?dl=1"
PENDULUM_RANDOM_URL = f"{DROPBOX_URL}/hhbq9i6ako24kzz/pendulum_random_v1.1.0.h5?dl=1"  # pylint: disable=line-too-long



def load_v1(f):
    r"""Loads v1 dataset data.

    Args:
        f: Binary file-like object.

    Returns:
        Sequence of episodes.
    """
    with h5py.File(f, "r") as h5:
        observations = h5["observations"][()]
        actions = h5["actions"][()]
        rewards = h5["rewards"][()]
        terminals = h5["terminals"][()]

        if "episode_terminals" in h5:
            episode_terminals = h5["episode_terminals"][()]
        else:
            episode_terminals = None

    if episode_terminals is None:
        timeouts = None
    else:
        timeouts = np.logical_and(np.logical_not(terminals), episode_terminals)

    return {
      "actions": actions,
      "observations": observations,
      "rewards": rewards,
      "terminals": terminals,
      "timeouts": timeouts
      }


def get_cartpole(
    dataset_type: str = "replay"
):
    """Returns cartpole dataset and environment.

    The dataset is automatically downloaded to ``d3rlpy_data/cartpole.h5`` if
    it does not exist.

    Args:
        dataset_type: dataset type. Available options are
            ``['replay', 'random']``.
        transition_picker: TransitionPickerProtocol object.
        trajectory_slicer: TrajectorySlicerProtocol object.
        render_mode: Mode of rendering (``human``, ``rgb_array``).

    Returns:
        tuple of :class:`d3rlpy.dataset.ReplayBuffer` and gym environment.
    """
    if dataset_type == "replay":
        url = CARTPOLE_URL
        file_name = "cartpole_replay_v1.1.0.h5"
    elif dataset_type == "random":
        url = CARTPOLE_RANDOM_URL
        file_name = "cartpole_random_v1.1.0.h5"
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}.")

    data_path = os.path.join(DATA_DIRECTORY, file_name)

    # download dataset
    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Downloading cartpole.pkl into {data_path}...")
        request.urlretrieve(url, data_path)

    # load dataset
    with open(data_path, "rb") as f:
        episodes = load_v1(f)

    return episodes


def get_pendulum(
    dataset_type: str = "replay"
):
    """Returns pendulum dataset and environment.

    The dataset is automatically downloaded to ``d3rlpy_data/pendulum.h5`` if
    it does not exist.

    Args:
        dataset_type: dataset type. Available options are
            ``['replay', 'random']``.
        transition_picker: TransitionPickerProtocol object.
        trajectory_slicer: TrajectorySlicerProtocol object.
        render_mode: Mode of rendering (``human``, ``rgb_array``).

    Returns:
        tuple of :class:`d3rlpy.dataset.ReplayBuffer` and gym environment.
    """
    if dataset_type == "replay":
        url = PENDULUM_URL
        file_name = "pendulum_replay_v1.1.0.h5"
    elif dataset_type == "random":
        url = PENDULUM_RANDOM_URL
        file_name = "pendulum_random_v1.1.0.h5"
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}.")

    data_path = os.path.join(DATA_DIRECTORY, file_name)

    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Donwloading pendulum.pkl into {data_path}...")
        urllib.request.urlretrieve(url, data_path)

    # load dataset
    with open(data_path, "rb") as f:
        episodes = load_v1(f)
    return episodes

def augment_data(dataset,
                 noise_scale):
  """Augments the data.

  Args:
    dataset: Dictionary with data.
    noise_scale: Scale of noise to apply.

  Returns:
    Augmented data.
  """
  noise_std = np.std(np.concatenate(dataset['rewards'], 0))
  for k, v in dataset.items():
    dataset[k] = np.repeat(v, 3, 0)

  dataset['rewards'][1::3] += noise_std * noise_scale
  dataset['rewards'][2::3] -= noise_std * noise_scale

  return dataset


def weighted_moments(x, weights):
  mean = np.sum(x * weights, 0) / np.sum(weights)
  sqr_diff = np.sum((x - mean)**2 * weights, 0)
  std = np.sqrt(sqr_diff / (weights.sum() - 1))
  return mean, std


class D4rlDataset(Dataset):
  """Dataset class for policy evaluation."""

  # pylint: disable=super-init-not-called
  def __init__(self,
               d4rl_dataset,
               normalize_states = False,
               normalize_rewards = False,
               eps = 1e-5,
               noise_scale = 0.0,
               bootstrap = True):
    """Processes data from D4RL environment.

    Args:
      d4rl_env: gym.Env corresponding to D4RL environment.
      normalize_states: whether to normalize the states.
      normalize_rewards: whether to normalize the rewards.
      eps: Epsilon used for normalization.
      noise_scale: Data augmentation noise scale.
      bootstrap: Whether to generated bootstrapped weights.
    """
    dataset = dict(
        trajectories=dict(
            states=[],
            actions=[],
            next_states=[],
            rewards=[],
            masks=[]))
    # d4rl_dataset = d4rl_env.get_dataset()
    dataset_length = len(d4rl_dataset['actions'])
    new_trajectory = True
    for idx in range(dataset_length):
      if new_trajectory:
        trajectory = dict(
            states=[], actions=[], next_states=[], rewards=[], masks=[])

      trajectory['states'].append(d4rl_dataset['observations'][idx])
      trajectory['actions'].append(d4rl_dataset['actions'][idx])
      trajectory['rewards'].append(d4rl_dataset['rewards'][idx])
      trajectory['masks'].append(1.0 - d4rl_dataset['terminals'][idx])
      if not new_trajectory:
        trajectory['next_states'].append(d4rl_dataset['observations'][idx])

      end_trajectory = (d4rl_dataset['terminals'][idx] or
                        d4rl_dataset['timeouts'][idx])
      if end_trajectory:
        trajectory['next_states'].append(d4rl_dataset['observations'][idx])
        if d4rl_dataset['timeouts'][idx] and not d4rl_dataset['terminals'][idx]:
          for key in trajectory:
            del trajectory[key][-1]
        if trajectory['actions']:
          for k, v in trajectory.items():
            assert len(v) == len(trajectory['actions'])
            dataset['trajectories'][k].append(np.array(v, dtype=np.float32))
          print('Added trajectory %d with length %d.' % (
              len(dataset['trajectories']['actions']),
              len(trajectory['actions'])))

      new_trajectory = end_trajectory

    if noise_scale > 0.0:
      dataset['trajectories'] = augment_data(dataset['trajectories'],  # pytype: disable=wrong-arg-types  # dict-kwargs
                                             noise_scale)

    dataset['trajectories']['steps'] = [
        np.arange(len(state_trajectory))
        for state_trajectory in dataset['trajectories']['states']
    ]

    dataset['initial_states'] = np.stack([
        state_trajectory[0]
        for state_trajectory in dataset['trajectories']['states']
    ])

    num_trajectories = len(dataset['trajectories']['states'])
    if bootstrap:
      dataset['initial_weights'] = np.random.multinomial(
          num_trajectories, [1.0 / num_trajectories] * num_trajectories,
          1).astype(np.float32)[0]
    else:
      dataset['initial_weights'] = np.ones(num_trajectories, dtype=np.float32)

    dataset['trajectories']['weights'] = []
    for i in range(len(dataset['trajectories']['masks'])):
      dataset['trajectories']['weights'].append(
          np.ones_like(dataset['trajectories']['masks'][i]) *
          dataset['initial_weights'][i])

    dataset['initial_weights'] = torch.tensor(
        dataset['initial_weights'],dtype=torch.float32)
    dataset['initial_states'] = torch.tensor(dataset['initial_states'],dtype=torch.float32)
    for k, v in dataset['trajectories'].items():
      if 'initial' not in k:
        dataset[k] = torch.tensor(
            np.concatenate(dataset['trajectories'][k], axis=0),dtype=torch.float32)

    self.states = dataset['states']
    self.actions = dataset['actions']
    self.next_states = dataset['next_states']
    self.masks = dataset['masks']
    self.weights = dataset['weights']
    self.rewards = dataset['rewards']
    self.steps = dataset['steps']

    self.initial_states = dataset['initial_states']
    self.initial_weights = dataset['initial_weights']

    self.eps = torch.tensor(eps,dtype=torch.float32)
    self.model_filename = None

    if normalize_states:
      self.state_mean = torch.mean(self.states, dim=0)
      self.state_std = torch.std(self.states, dim=0, unbiased=False)

      self.initial_states = self.normalize_states(self.initial_states)
      self.states = self.normalize_states(self.states)
      self.next_states = self.normalize_states(self.next_states)
    else:
      self.state_mean = 0.0
      self.state_std = 1.0

    if normalize_rewards:
      self.reward_mean = torch.mean(self.rewards)
      if torch.min(self.masks) == 0.0:
        self.reward_mean = torch.zeros_like(self.reward_mean)
      self.reward_std = torch.std(self.rewards, unbiased=False)

      self.rewards = self.normalize_rewards(self.rewards)
    else:
      self.reward_mean = 0.0
      self.reward_std = 1.0
  # pylint: enable=super-init-not-called

  def normalize_states(self, states):
    dtype = torch.float32
    return ((states - self.state_mean) /
            torch.maximum(torch.tensor(self.eps,dtype=dtype), self.state_std))

  def unnormalize_states(self, states):
    dtype = torch.float32
    return (states * torch.maximum(torch.tensor(self.eps,dtype=dtype), self.state_std)
            + self.state_mean)

  def normalize_rewards(self, rewards):
    return (rewards - self.reward_mean) / torch.maximum(self.reward_std, self.eps)

  def unnormalize_rewards(self, rewards):
    return rewards * torch.maximum(self.reward_std, self.eps) + self.reward_mean

  def __len__(self):
    return len(self.states)

  def __getitem__(self, idx):
        return (self.states[idx], self.actions[idx], self.next_states[idx],
                self.rewards[idx], self.masks[idx], self.weights[idx], self.steps[idx])

def infinite_loader(dataloader, behavior_dataset):
    while True:
        for data in dataloader:
            yield data
        # 当 DataLoader 的数据遍历完毕，重新创建 DataLoader
        dataloader = DataLoader(behavior_dataset, batch_size=256, shuffle=True, drop_last=True, num_workers=4)


def _stack_frames(episode: Episode, num_stack: int) -> Episode:
    assert isinstance(episode.observations, np.ndarray)
    episode_length = episode.observations.shape[0]
    observations: UInt8NDArray = np.zeros(
        (episode_length, num_stack, 84, 84),
        dtype=np.uint8,
    )
    for i in range(num_stack):
        pad_size = num_stack - i - 1
        if pad_size > 0:
            observations[pad_size:, i] = np.reshape(
                episode.observations[:-pad_size], [-1, 84, 84]
            )
        else:
            observations[:, i] = np.reshape(episode.observations, [-1, 84, 84])
    return Episode(
        observations=observations,
        actions=episode.actions.copy(),
        rewards=episode.rewards.copy(),
        terminated=episode.terminated,
    )


def get_atari(
    env_name: str,
    num_stack: Optional[int] = None,
    sticky_action: bool = True,
    pre_stack: bool = False,
    render_mode: Optional[str] = None,
) -> Tuple[ReplayBuffer, gym.Env[NDArray, int]]:
    """Returns atari dataset and envrironment.

    The dataset is provided through d4rl-atari. See more details including
    available dataset from its GitHub page.

    .. code-block:: python

        from d3rlpy.datasets import get_atari

        dataset, env = get_atari('breakout-mixed-v0')

    References:
        * https://github.com/takuseno/d4rl-atari

    Args:
        env_name: environment id of d4rl-atari dataset.
        num_stack: the number of frames to stack (only applied to env).
        sticky_action: Flag to enable sticky action.
        pre_stack: Flag to pre-stack observations. If this is ``False``,
            ``FrameStackTransitionPicker`` and ``FrameStackTrajectorySlicer``
            will be used to stack observations at sampling-time.
        render_mode: Mode of rendering (``human``, ``rgb_array``).

    Returns:
        tuple of :class:`d3rlpy.dataset.ReplayBuffer` and gym environment.
    """
    try:
        import d4rl_atari  # type: ignore

        env = gym.make(
            env_name,
            render_mode=render_mode,
            sticky_action=sticky_action,
        )
        raw_dataset = env.get_dataset()  # type: ignore
        episode_generator = EpisodeGenerator(**raw_dataset)
        episodes = episode_generator()

        if pre_stack:
            stacked_episodes = []
            for episode in episodes:
                assert num_stack is not None
                stacked_episode = _stack_frames(episode, num_stack)
                stacked_episodes.append(stacked_episode)
            episodes = stacked_episodes

        picker: TransitionPickerProtocol
        slicer: TrajectorySlicerProtocol
        if num_stack is None or pre_stack:
            picker = BasicTransitionPicker()
            slicer = BasicTrajectorySlicer()
        else:
            picker = FrameStackTransitionPicker(num_stack or 1)
            slicer = FrameStackTrajectorySlicer(num_stack or 1)

        dataset = create_infinite_replay_buffer(
            episodes=episodes,
            transition_picker=picker,
            trajectory_slicer=slicer,
        )
        if num_stack:
            env = FrameStack(env, num_stack=num_stack)
        else:
            env = ChannelFirst(env)
        return dataset, env
    except ImportError as e:
        raise ImportError(
            "d4rl-atari is not installed.\n" "$ d3rlpy install d4rl_atari"
        ) from e


def get_atari_transitions(
    game_name: str,
    fraction: float = 0.01,
    index: int = 0,
    num_stack: Optional[int] = None,
    sticky_action: bool = True,
    pre_stack: bool = False,
    render_mode: Optional[str] = None,
) -> Tuple[ReplayBuffer, gym.Env[NDArray, int]]:
    """Returns atari dataset as a list of Transition objects and envrironment.

    The dataset is provided through d4rl-atari.
    The difference from ``get_atari`` function is that this function will
    sample transitions from all epochs.
    This function is necessary for reproducing Atari experiments.

    .. code-block:: python

        from d3rlpy.datasets import get_atari_transitions

        # get 1% of transitions from all epochs (1M x 50 epoch x 1% = 0.5M)
        dataset, env = get_atari_transitions('breakout', fraction=0.01)

    References:
        * https://github.com/takuseno/d4rl-atari

    Args:
        game_name: Atari 2600 game name in lower_snake_case.
        fraction: fraction of sampled transitions.
        index: index to specify which trial to load.
        num_stack: the number of frames to stack (only applied to env).
        sticky_action: Flag to enable sticky action.
        pre_stack: Flag to pre-stack observations. If this is ``False``,
            ``FrameStackTransitionPicker`` and ``FrameStackTrajectorySlicer``
            will be used to stack observations at sampling-time.
        render_mode: Mode of rendering (``human``, ``rgb_array``).

    Returns:
        tuple of a list of :class:`d3rlpy.dataset.Transition` and gym
        environment.
    """
    try:
        import d4rl_atari

        # each epoch consists of 1M steps
        num_transitions_per_epoch = int(1000000 * fraction)

        copied_episodes = []
        for i in range(50):
            env_name = f"{game_name}-epoch-{i + 1}-v{index}"
            LOG.info(f"Collecting {env_name}...")
            env = gym.make(
                env_name,
                sticky_action=sticky_action,
                render_mode=render_mode,
            )
            raw_dataset = env.get_dataset()  # type: ignore
            episode_generator = EpisodeGenerator(**raw_dataset)
            episodes = list(episode_generator())

            # copy episode data to release memory of unused data
            random.shuffle(episodes)
            num_data = 0
            for episode in episodes:
                if num_data >= num_transitions_per_epoch:
                    break

                assert isinstance(episode.observations, np.ndarray)
                copied_episode = Episode(
                    observations=episode.observations.copy(),
                    actions=episode.actions.copy(),
                    rewards=episode.rewards.copy(),
                    terminated=episode.terminated,
                )
                if pre_stack:
                    assert num_stack is not None
                    copied_episode = _stack_frames(copied_episode, num_stack)

                # trim episode
                if num_data + copied_episode.size() > num_transitions_per_epoch:
                    end = num_transitions_per_epoch - num_data
                    copied_episode = Episode(
                        observations=copied_episode.observations[:end],
                        actions=copied_episode.actions[:end],
                        rewards=copied_episode.rewards[:end],
                        terminated=False,
                    )

                copied_episodes.append(copied_episode)
                num_data += copied_episode.size()

        picker: TransitionPickerProtocol
        slicer: TrajectorySlicerProtocol
        if num_stack is None or pre_stack:
            picker = BasicTransitionPicker()
            slicer = BasicTrajectorySlicer()
        else:
            picker = FrameStackTransitionPicker(num_stack or 1)
            slicer = FrameStackTrajectorySlicer(num_stack or 1)

        dataset = ReplayBuffer(
            InfiniteBuffer(),
            episodes=copied_episodes,
            transition_picker=picker,
            trajectory_slicer=slicer,
        )

        if num_stack:
            env = FrameStack(env, num_stack=num_stack)
        else:
            env = ChannelFirst(env)

        return dataset, env
    except ImportError as e:
        raise ImportError(
            "d4rl-atari is not installed.\n" "$ d3rlpy install d4rl_atari"
        ) from e


def get_d4rl(
    env_name: str,
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
    render_mode: Optional[str] = None,
) -> Tuple[ReplayBuffer, gym.Env[NDArray, NDArray]]:
    """Returns d4rl dataset and envrironment.

    The dataset is provided through d4rl.

    .. code-block:: python

        from d3rlpy.datasets import get_d4rl

        dataset, env = get_d4rl('hopper-medium-v0')

    References:
        * `Fu et al., D4RL: Datasets for Deep Data-Driven Reinforcement
          Learning. <https://arxiv.org/abs/2004.07219>`_
        * https://github.com/rail-berkeley/d4rl

    Args:
        env_name: environment id of d4rl dataset.
        transition_picker: TransitionPickerProtocol object.
        trajectory_slicer: TrajectorySlicerProtocol object.
        render_mode: Mode of rendering (``human``, ``rgb_array``).

    Returns:
        tuple of :class:`d3rlpy.dataset.ReplayBuffer` and gym environment.
    """
    try:
        import d4rl  # type: ignore

        env = gym.make(env_name)
        raw_dataset: Dict[str, NDArray] = env.get_dataset()  # type: ignore

        observations = raw_dataset["observations"]
        actions = raw_dataset["actions"]
        rewards = raw_dataset["rewards"]
        terminals = raw_dataset["terminals"]
        timeouts = raw_dataset["timeouts"]

        dataset = MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            timeouts=timeouts,
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
        )

        # wrapped by NormalizedBoxEnv that is incompatible with newer Gym
        unwrapped_env: gym.Env[Any, Any] = env.env.env.env.wrapped_env  # type: ignore
        unwrapped_env.render_mode = render_mode  # overwrite

        return dataset, TimeLimit(unwrapped_env, max_episode_steps=1000)
    except ImportError as e:
        raise ImportError(
            "d4rl is not installed.\n" "$ d3rlpy install d4rl"
        ) from e


ATARI_GAMES = [
    "adventure",
    "air-raid",
    "alien",
    "amidar",
    "assault",
    "asterix",
    "asteroids",
    "atlantis",
    "bank-heist",
    "battle-zone",
    "beam-rider",
    "berzerk",
    "bowling",
    "boxing",
    "breakout",
    "carnival",
    "centipede",
    "chopper-command",
    "crazy-climber",
    "defender",
    "demon-attack",
    "double-dunk",
    "elevator-action",
    "enduro",
    "fishing-derby",
    "freeway",
    "frostbite",
    "gopher",
    "gravitar",
    "hero",
    "ice-hockey",
    "jamesbond",
    "journey-escape",
    "kangaroo",
    "krull",
    "kung-fu-master",
    "montezuma-revenge",
    "ms-pacman",
    "name-this-game",
    "phoenix",
    "pitfall",
    "pong",
    "pooyan",
    "private-eye",
    "qbert",
    "riverraid",
    "road-runner",
    "robotank",
    "seaquest",
    "skiing",
    "solaris",
    "space-invaders",
    "star-gunner",
    "tennis",
    "time-pilot",
    "tutankham",
    "up-n-down",
    "venture",
    "video-pinball",
    "wizard-of-wor",
    "yars-revenge",
    "zaxxon",
]


def get_dataset(
    env_name: str,
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
    render_mode: Optional[str] = None,
) -> Tuple[ReplayBuffer, gym.Env[Any, Any]]:
    """Returns dataset and envrironment by guessing from name.

    This function returns dataset by matching name with the following datasets.

    - cartpole-replay
    - cartpole-random
    - pendulum-replay
    - pendulum-random
    - d4rl-pybullet
    - d4rl-atari
    - d4rl

    .. code-block:: python

       import d3rlpy

       # cartpole dataset
       dataset, env = d3rlpy.datasets.get_dataset('cartpole')

       # pendulum dataset
       dataset, env = d3rlpy.datasets.get_dataset('pendulum')

       # d4rl-atari dataset
       dataset, env = d3rlpy.datasets.get_dataset('breakout-mixed-v0')

       # d4rl dataset
       dataset, env = d3rlpy.datasets.get_dataset('hopper-medium-v0')

    Args:
        env_name: environment id of the dataset.
        transition_picker: TransitionPickerProtocol object.
        trajectory_slicer: TrajectorySlicerProtocol object.
        render_mode: Mode of rendering (``human``, ``rgb_array``).

    Returns:
        tuple of :class:`d3rlpy.dataset.ReplayBuffer` and gym environment.
    """
    if env_name == "cartpole-replay":
        return get_cartpole(
            dataset_type="replay",
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            render_mode=render_mode,
        )
    elif env_name == "cartpole-random":
        return get_cartpole(
            dataset_type="random",
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            render_mode=render_mode,
        )
    elif env_name == "pendulum-replay":
        return get_pendulum(
            dataset_type="replay",
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            render_mode=render_mode,
        )
    elif env_name == "pendulum-random":
        return get_pendulum(
            dataset_type="random",
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            render_mode=render_mode,
        )
    elif re.match(r"^bullet-.+$", env_name):
        return get_d4rl(
            env_name,
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            render_mode=render_mode,
        )
    elif re.match(r"hopper|halfcheetah|walker|ant", env_name):
        return get_d4rl(
            env_name,
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            render_mode=render_mode,
        )
    raise ValueError(f"Unrecognized env_name: {env_name}.")
