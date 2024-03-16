from collections import deque
from typing import Any, Deque, Dict, Optional, Sequence, Tuple, TypeVar

import gym
import numpy as np

try:
    import cv2  # this is used in AtariPreprocessing
except ImportError:
    cv2 = None

from gym.spaces import Box
from gym.wrappers.transform_reward import TransformReward

from ..types import NDArray

__all__ = [
    "ChannelFirst",
    "FrameStack",
    "AtariPreprocessing",
    "Atari",
]

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class ChannelFirst(gym.Wrapper[_ObsType, _ActType]):
    """Channel-first wrapper for image observation environments.

    d3rlpy expects channel-first images since it's built with PyTorch.
    You can transform the observation shape with ``ChannelFirst`` wrapper.

    Args:
        env (gym.Env): gym environment.
    """

    observation_space: Box

    def __init__(self, env: gym.Env[_ObsType, _ActType]):
        super().__init__(env)
        shape = self.observation_space.shape
        low = self.observation_space.low
        high = self.observation_space.high
        dtype = self.observation_space.dtype
        assert dtype is not None

        if len(shape) == 3:
            self.observation_space = Box(
                low=np.transpose(low, [2, 0, 1]),
                high=np.transpose(high, [2, 0, 1]),
                shape=(shape[2], shape[0], shape[1]),
                dtype=dtype,  # type: ignore
            )
        elif len(shape) == 2:
            self.observation_space = Box(
                low=np.reshape(low, (1, *shape)),
                high=np.reshape(high, (1, *shape)),
                shape=(1, *shape),
                dtype=dtype,  # type: ignore
            )
        else:
            raise ValueError("image observation is only allowed.")

    def step(
        self, action: _ActType
    ) -> Tuple[_ObsType, float, bool, bool, Dict[str, Any]]:
        observation, reward, terminal, truncated, info = self.env.step(action)
        # make channel first observation
        if observation.ndim == 3:
            observation_T = np.transpose(observation, [2, 0, 1])
        else:
            observation_T = np.reshape(observation, (1, *observation.shape))
        assert observation_T.shape == self.observation_space.shape
        return observation_T, reward, terminal, truncated, info  # type: ignore

    def reset(self, **kwargs: Any) -> Tuple[_ObsType, Dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        # make channel first observation
        if observation.ndim == 3:
            observation_T = np.transpose(observation, [2, 0, 1])
        else:
            observation_T = np.reshape(observation, (1, *observation.shape))
        assert observation_T.shape == self.observation_space.shape
        return observation_T, info  # type: ignore


class FrameStack(gym.Wrapper[NDArray, _ActType]):
    """Observation wrapper that stacks the observations in a rolling manner.

    This wrapper is implemented based on gym.wrappers.FrameStack. The
    difference is that this wrapper returns stacked frames as numpy array.

    Args:
        env (gym.Env): gym environment.
        num_stack (int): the number of frames to stack.
    """

    _num_stack: int
    _frames: Deque[NDArray]

    def __init__(self, env: gym.Env[NDArray, _ActType], num_stack: int):
        super().__init__(env)
        self._num_stack = num_stack
        self._frames = deque(maxlen=num_stack)

        low = np.repeat(
            self.observation_space.low[np.newaxis, ...],  # type: ignore
            num_stack,
            axis=0,
        )
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...],  # type: ignore
            num_stack,
            axis=0,
        )
        self.observation_space = Box(
            low=low,
            high=high,
            dtype=self.observation_space.dtype,  # type: ignore
        )

    def observation(self, observation: Any) -> NDArray:
        return np.array(self._frames, dtype=self._frames[-1].dtype)

    def step(
        self, action: _ActType
    ) -> Tuple[NDArray, float, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(observation)
        return self.observation(None), reward, terminated, truncated, info

    def reset(self, **kwargs: Any) -> Tuple[NDArray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        for _ in range(self._num_stack - 1):
            self._frames.append(np.zeros_like(obs))
        self._frames.append(obs)
        return self.observation(None), info


# https://github.com/openai/gym/blob/0.17.3/gym/wrappers/atari_preprocessing.py
class AtariPreprocessing(gym.Wrapper[NDArray, int]):
    r"""Atari 2600 preprocessings.
    This class follows the guidelines in
    Machado et al. (2018), "Revisiting the Arcade Learning Environment:
    Evaluation Protocols and Open Problems for General Agents".
    Specifically:

    * NoopReset: obtain initial state by taking random number of no-ops on
        reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost: turned off by default. Not
        recommended by Machado et al. (2018).
    * Resize to a square image: 84x84 by default
    * Grayscale observation: optional
    * Scale observation: optional

    Args:
        env (Env): environment
        noop_max (int): max number of no-ops
        frame_skip (int): the frequency at which the agent experiences the game.
        screen_size (int): resize Atari frame
        terminal_on_life_loss (bool): if True, then step() returns done=True
            whenever a life is lost.
        grayscale_obs (bool): if True, then gray scale observation is returned,
            otherwise, RGB observation is returned.
        grayscale_newaxis (bool): if True and grayscale_obs=True, then a
            channel axis is added to grayscale observations to make them
            3-dimensional.
        scale_obs (bool): if True, then observation normalized in range [0,1]
            is returned. It also limits memory optimization benefits of
            FrameStack Wrapper.

    """
    _obs_buffer: Sequence[NDArray]

    def __init__(
        self,
        env: gym.Env[NDArray, int],
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int = 84,
        terminal_on_life_loss: bool = False,
        grayscale_obs: bool = True,
        grayscale_newaxis: bool = False,
        scale_obs: bool = False,
    ):
        super().__init__(env)
        assert cv2 is not None, (
            "opencv-python package not installed! Try"
            " running pip install gym[atari] to get dependencies for atari"
        )
        assert frame_skip > 0
        assert screen_size > 0
        assert noop_max >= 0
        if frame_skip > 1:
            assert "NoFrameskip" in env.spec.id, (
                "disable frame-skipping in"
                " the original env. for more than one frame-skip as it will"
                " be done by the wrapper"
            )
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # type: ignore

        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.terminal_on_life_loss = terminal_on_life_loss
        self.grayscale_obs = grayscale_obs
        self.grayscale_newaxis = grayscale_newaxis
        self.scale_obs = scale_obs

        # buffer of most recent two observations for max pooling
        shape = env.observation_space.shape
        assert shape is not None
        if grayscale_obs:
            self._obs_buffer = [
                np.empty(shape[:2], dtype=np.uint8),
                np.empty(shape[:2], dtype=np.uint8),
            ]
        else:
            self._obs_buffer = [
                np.empty(shape, dtype=np.uint8),
                np.empty(shape, dtype=np.uint8),
            ]

        self.ale = env.unwrapped.ale  # type: ignore
        self.lives = 0
        self.game_over = True

        _low, _high, _obs_dtype = (
            (0, 255, np.uint8) if not scale_obs else (0, 1, np.float32)
        )
        _shape = (screen_size, screen_size, 1 if grayscale_obs else 3)
        if grayscale_obs and not grayscale_newaxis:
            _shape = _shape[:-1]  # type: ignore
        self.observation_space = Box(
            low=_low, high=_high, shape=_shape, dtype=_obs_dtype
        )

    def step(
        self, action: int
    ) -> Tuple[NDArray, float, bool, bool, Dict[str, Any]]:
        R = 0.0

        for t in range(self.frame_skip):
            _, reward, done, truncated, info = self.env.step(action)
            R += reward
            self.game_over = done

            if self.terminal_on_life_loss:
                new_lives = self.ale.lives()
                done = done or new_lives < self.lives
                self.lives = new_lives

            if done or truncated:
                break
            if t == self.frame_skip - 2:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self._obs_buffer[1])
                else:
                    self.ale.getScreenRGB2(self._obs_buffer[1])
            elif t == self.frame_skip - 1:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self._obs_buffer[0])
                else:
                    self.ale.getScreenRGB2(self._obs_buffer[0])

        return self._get_obs(), R, done, truncated, info

    def reset(self, **kwargs: Any) -> Tuple[NDArray, Dict[str, Any]]:
        # this condition is not included in the original code
        if self.game_over:
            _, info = self.env.reset(**kwargs)
        else:
            # NoopReset
            _, _, _, _, info = self.env.step(0)

        noops = (
            self.env.unwrapped.np_random.integers(1, self.noop_max + 1)
            if self.noop_max > 0
            else 0
        )
        for _ in range(noops):
            _, _, done, truncated, info = self.env.step(0)
            if done or truncated:
                _, info = self.env.reset(**kwargs)

        self.lives = self.ale.lives()
        if self.grayscale_obs:
            self.ale.getScreenGrayscale(self._obs_buffer[0])
        else:
            self.ale.getScreenRGB2(self._obs_buffer[0])
        self._obs_buffer[1].fill(0)
        return self._get_obs(), info

    def _get_obs(self) -> NDArray:
        if self.frame_skip > 1:  # more efficient in-place pooling
            np.maximum(
                self._obs_buffer[0],
                self._obs_buffer[1],
                out=self._obs_buffer[0],
            )
        obs = cv2.resize(
            self._obs_buffer[0],
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA,
        )

        if self.scale_obs:
            obs = np.asarray(obs, dtype=np.float32) / 255.0
        else:
            obs = np.asarray(obs, dtype=np.uint8)

        if self.grayscale_obs and self.grayscale_newaxis:
            obs = np.expand_dims(obs, axis=-1)  # Add a channel axis
        return obs  # type: ignore


class Atari(gym.Wrapper[NDArray, int]):
    """Atari 2600 wrapper for experiments.

    Args:
        env (gym.Env): gym environment.
        num_stack (int): the number of frames to stack.
        is_eval (bool): flag to enter evaluation mode.
    """

    def __init__(
        self,
        env: gym.Env[NDArray, int],
        num_stack: Optional[int] = None,
        is_eval: bool = False,
    ):
        env = AtariPreprocessing(env, terminal_on_life_loss=not is_eval)
        if not is_eval:
            env = TransformReward(env, lambda r: float(np.clip(r, -1.0, 1.0)))
        if num_stack:
            env = FrameStack(env, num_stack)
        else:
            env = ChannelFirst(env)
        super().__init__(env)
