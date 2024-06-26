from abc import abstractmethod
import pdb
import time
import pickle
from collections import defaultdict
from typing import (
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    cast,
)
import os
import csv
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm, trange
from typing_extensions import Self
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from ...base import ImplBase, LearnableBase, LearnableConfig, save_config
from ...constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ...dataset import (
    ReplayBuffer,
    D4rlDataset,
    TransitionMiniBatch,
    check_non_1d_array,
    create_fifo_replay_buffer,
    is_tuple_shape,
    infinite_loader
)
from ...envs import GymEnv
from ...logging import (
    LOG,
    D3RLPyLogger,
    FileAdapterFactory,
    LoggerAdapterFactory,
)
from ...metrics import EvaluatorProtocol, evaluate_qlearning_with_environment
from ...models.torch import Policy
from ...torch_utility import (
    TorchMiniBatch,
    convert_to_torch,
    convert_to_torch_recursively,
    eval_api,
    hard_sync,
    sync_optimizer_state,
    train_api,
)
from ...types import NDArray, Observation
from ..utility import (
    assert_action_space_with_dataset,
    assert_action_space_with_env,
    build_scalers_with_env,
    build_scalers_with_transition_picker,
)
from .explorers import Explorer
from ...utils import save_policy,run
import pandas as pd


__all__ = [
    "QLearningAlgoImplBase",
    "QLearningAlgoBase",
    "TQLearningImpl",
    "TQLearningConfig",
]


class QLearningAlgoImplBase(ImplBase):
    @train_api
    def update(self, batch: TorchMiniBatch, grad_step: int) -> Dict[str, float]:
        return self.inner_update(batch, grad_step)

    @abstractmethod
    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        pass

    @eval_api
    def predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    @abstractmethod
    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @eval_api
    def sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_sample_action(x)

    @abstractmethod
    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @eval_api
    def predict_value(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self.inner_predict_value(x, action)

    @abstractmethod
    def inner_predict_value(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        pass

    @property
    def policy(self) -> Policy:
        raise NotImplementedError

    def copy_policy_from(self, impl: "QLearningAlgoImplBase") -> None:
        if not isinstance(impl.policy, type(self.policy)):
            raise ValueError(
                f"Invalid policy type: expected={type(self.policy)},"
                f"actual={type(impl.policy)}"
            )
        hard_sync(self.policy, impl.policy)

    @property
    def policy_optim(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    def copy_policy_optim_from(self, impl: "QLearningAlgoImplBase") -> None:
        if not isinstance(impl.policy_optim, type(self.policy_optim)):
            raise ValueError(
                "Invalid policy optimizer type: "
                f"expected={type(self.policy_optim)},"
                f"actual={type(impl.policy_optim)}"
            )
        sync_optimizer_state(self.policy_optim, impl.policy_optim)

    @property
    def q_function(self) -> nn.ModuleList:
        raise NotImplementedError

    def copy_q_function_from(self, impl: "QLearningAlgoImplBase") -> None:
        q_func = self.q_function[0]
        if not isinstance(impl.q_function[0], type(q_func)):
            raise ValueError(
                f"Invalid Q-function type: expected={type(q_func)},"
                f"actual={type(impl.q_function[0])}"
            )
        hard_sync(self.q_function, impl.q_function)

    @property
    def q_function_optim(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    def copy_q_function_optim_from(self, impl: "QLearningAlgoImplBase") -> None:
        if not isinstance(impl.q_function_optim, type(self.q_function_optim)):
            raise ValueError(
                "Invalid Q-function optimizer type: "
                f"expected={type(self.q_function_optim)}",
                f"actual={type(impl.q_function_optim)}",
            )
        sync_optimizer_state(self.q_function_optim, impl.q_function_optim)

    def reset_optimizer_states(self) -> None:
        self.modules.reset_optimizer_states()


TQLearningImpl = TypeVar("TQLearningImpl", bound=QLearningAlgoImplBase)
TQLearningConfig = TypeVar("TQLearningConfig", bound=LearnableConfig)



class QLearningAlgoBase(
    Generic[TQLearningImpl, TQLearningConfig],
    LearnableBase[TQLearningImpl, TQLearningConfig],
):
    def save_policy(self, fname: str) -> None:
        """Save the greedy-policy computational graph as TorchScript or ONNX.

        The format will be automatically detected by the file name.

        .. code-block:: python

            # save as TorchScript
            algo.save_policy('policy.pt')

            # save as ONNX
            algo.save_policy('policy.onnx')

        The artifacts saved with this method will work without d3rlpy.
        This method is especially useful to deploy the learned policy to
        production environments or embedding systems.

        See also

            * https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html (for Python).
            * https://pytorch.org/tutorials/advanced/cpp_export.html (for C++).
            * https://onnx.ai (for ONNX)

        Args:
            fname: Destination file path.
        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        if is_tuple_shape(self._impl.observation_shape):
            dummy_x = [
                torch.rand(1, *shape, device=self._device)
                for shape in self._impl.observation_shape
            ]
        else:
            dummy_x = torch.rand(
                1, *self._impl.observation_shape, device=self._device
            )

        # workaround until version 1.6
        self._impl.modules.freeze()

        # local function to select best actions
        def _func(x: torch.Tensor) -> torch.Tensor:
            assert self._impl

            if self._config.observation_scaler:
                x = self._config.observation_scaler.transform(x)

            action = self._impl.predict_best_action(x)

            if self._config.action_scaler:
                action = self._config.action_scaler.reverse_transform(action)

            return action

        traced_script = torch.jit.trace(_func, dummy_x, check_trace=False)

        if fname.endswith(".onnx"):
            # currently, PyTorch cannot directly export function as ONNX.
            torch.onnx.export(
                traced_script,
                dummy_x,
                fname,
                export_params=True,
                opset_version=11,
                input_names=["input_0"],
                output_names=["output_0"],
            )
        elif fname.endswith(".pt"):
            traced_script.save(fname)
        else:
            raise ValueError(
                f"invalid format type: {fname}."
                " .pt and .onnx extensions are currently supported."
            )

        # workaround until version 1.6
        self._impl.modules.unfreeze()

    def predict(self, x: Observation) -> NDArray:
        """Returns greedy actions.

        .. code-block:: python

            # 100 observations with shape of (10,)
            x = np.random.random((100, 10))

            actions = algo.predict(x)
            # actions.shape == (100, action size) for continuous control
            # actions.shape == (100,) for discrete control

        Args:
            x: Observations

        Returns:
            Greedy actions
        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        assert check_non_1d_array(x), "Input must have batch dimension."

        # TODO: support tuple inputs
        torch_x = cast(
            torch.Tensor, convert_to_torch_recursively(x, self._device)
        )

        with torch.no_grad():
            if self._config.observation_scaler:
                torch_x = self._config.observation_scaler.transform(torch_x)

            action = self._impl.predict_best_action(torch_x)

            if self._config.action_scaler:
                action = self._config.action_scaler.reverse_transform(action)

        return action.cpu().detach().numpy()  # type: ignore

    def predict_value(self, x: Observation, action: NDArray) -> NDArray:
        """Returns predicted action-values.

        .. code-block:: python

            # 100 observations with shape of (10,)
            x = np.random.random((100, 10))

            # for continuous control
            # 100 actions with shape of (2,)
            actions = np.random.random((100, 2))

            # for discrete control
            # 100 actions in integer values
            actions = np.random.randint(2, size=100)

            values = algo.predict_value(x, actions)
            # values.shape == (100,)

        Args:
            x: Observations
            action: Actions

        Returns:
            Predicted action-values
        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        assert check_non_1d_array(x), "Input must have batch dimension."

        # TODO: support tuple inputs
        torch_x = cast(
            torch.Tensor, convert_to_torch_recursively(x, self._device)
        )

        torch_action = convert_to_torch(action, self._device)

        with torch.no_grad():
            if self._config.observation_scaler:
                torch_x = self._config.observation_scaler.transform(torch_x)

            if self.get_action_type() == ActionSpace.CONTINUOUS:
                if self._config.action_scaler:
                    torch_action = self._config.action_scaler.transform(
                        torch_action
                    )
            elif self.get_action_type() == ActionSpace.DISCRETE:
                torch_action = torch_action.long()
            else:
                raise ValueError("invalid action type")

            value = self._impl.predict_value(torch_x, torch_action)

        return value.cpu().detach().numpy()  # type: ignore

    def sample_action(self, x: Observation) -> NDArray:
        """Returns sampled actions.

        The sampled actions are identical to the output of `predict` method if
        the policy is deterministic.

        Args:
            x: Observations.

        Returns:
            Sampled actions.
        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        assert check_non_1d_array(x), "Input must have batch dimension."

        # TODO: support tuple inputs
        torch_x = cast(
            torch.Tensor, convert_to_torch_recursively(x, self._device)
        )

        with torch.no_grad():
            if self._config.observation_scaler:
                torch_x = self._config.observation_scaler.transform(torch_x)

            action = self._impl.sample_action(torch_x)

            # transform action back to the original range
            if self._config.action_scaler:
                action = self._config.action_scaler.reverse_transform(action)

        return action.cpu().detach().numpy()  # type: ignore

    def fit(
        self,
        method: str,
        dataset: Optional[Dict[str, float]],
        n_epoch: int = 100,
        show_progress: bool = True,
        save_interval: int = 1,
        evaluators: Optional[Dict[str, EvaluatorProtocol]] = None,
        callback: Optional[Callable[[Self, int, int], None]] = None,
        epoch_callback: Optional[Callable[[Self, int, int], None]] = None,
        dir_path: str = None,
        seed: int = 0,
        env_name: str = None,
        collect_epoch: int = 50,
        estimator_lr: float = 0.003,
        estimator_lr_decay: float = 0.86,
        temp: float = 1.0,
        algo: str = None,
        upload: bool = False,
        collect: bool = False
    ) -> List[Tuple[int, Dict[str, float]]]:
        """Trains with given dataset.

        .. code-block:: python

            algo.fit(episodes, n_steps=1000000)

        Args:
            dataset: ReplayBuffer object.
            n_steps: Number of steps to train.
            n_steps_per_epoch: Number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            experiment_name: Experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: Flag to add timestamp string to the last of
                directory name.
            logger_adapter: LoggerAdapterFactory object.
            show_progress: Flag to show progress bar for iterations.
            save_interval: Interval to save parameters.
            evaluators: List of evaluators.
            callback: Callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
            epoch_callback: Callable function that takes
                ``(algo, epoch, total_step)``, which is called at the end of
                every epoch.

        Returns:
            List of result tuples (epoch, metrics) per epoch.
        """
        if method == "baseline1":
            self.fitter_1(
                method=method,
                dataset=dataset,
                n_epoch=n_epoch,
                show_progress=show_progress,
                save_interval=save_interval,
                evaluators=evaluators,
                callback=callback,
                epoch_callback=epoch_callback,
                dir_path=dir_path,
                seed=seed,
                env_name=env_name,
                collect_epoch=collect_epoch,
                algo=algo,
                upload=upload,
                collect=collect
                )
        elif method == "baseline2":
            self.fitter_2(
                method=method,
                dataset=dataset,
                n_epoch=n_epoch,
                show_progress=show_progress,
                save_interval=save_interval,
                evaluators=evaluators,
                callback=callback,
                epoch_callback=epoch_callback,
                dir_path=dir_path,
                seed=seed,
                env_name=env_name,
                collect_epoch=collect_epoch,
                estimator_lr=estimator_lr,
                estimator_lr_decay=estimator_lr_decay,
                temp=temp,
                algo=algo,
                upload=upload,
                collect=collect
                )

    def fitter_1(
        self,
        method: str,
        dataset: Optional[Dict[str, float]],
        n_epoch: int = 100,
        show_progress: bool = True,
        save_interval: int = 1,
        evaluators: Optional[Dict[str, EvaluatorProtocol]] = None,
        callback: Optional[Callable[[Self, int, int], None]] = None,
        epoch_callback: Optional[Callable[[Self, int, int], None]] = None,
        dir_path: str = None,
        seed: int = 0,
        env_name: str = None,
        collect_epoch: int = 50,
        algo: str = None,
        upload: bool = False,
        collect: bool = False
    ) -> Generator[Tuple[int, Dict[str, float]], None, None]:
        """Iterate over epochs steps to train with the given dataset. At each
        iteration algo methods and properties can be changed or queried.

        .. code-block:: python

            for epoch, metrics in algo.fitter(episodes):
                my_plot(metrics)
                algo.save_model(my_path)

        Args:
            dataset: Offline dataset to train.
            n_steps: Number of steps to train.
            n_steps_per_epoch: Number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            experiment_name: Experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: Flag to add timestamp string to the last of
                directory name.
            logger_adapter: LoggerAdapterFactory object.
            show_progress: Flag to show progress bar for iterations.
            save_interval: Interval to save parameters.
            evaluators: List of evaluators.
            callback: Callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
            epoch_callback: Callable function that takes
                ``(algo, epoch, total_step)``, which is called at the end of
                every epoch.

        Returns:
            Iterator yielding current epoch and metrics dict.
        """
        if self._impl is None:
            LOG.debug("Building models...")
            action_size = evaluators["environment"]._env.unwrapped.action_space.shape[0]
            observation_shape = evaluators["environment"]._env.unwrapped.observation_space.shape
            self.create_impl(observation_shape, action_size)
            LOG.debug("Models have been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open("{}/loss.csv".format(dir_path), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "actor_loss", "critic_loss", "temp_loss", "temp", "Oracle_1.0", "Oracle"])  # 写入标题

        # training loop
        total_step = 0

        # scheduler_actor = optim.lr_scheduler.StepLR(self._impl.modules.actor_optim, step_size=decay_epoch, gamma=lr_decay)
        # scheduler_critic = optim.lr_scheduler.StepLR(self._impl.modules.critic_optim, step_size=decay_epoch, gamma=lr_decay)

        if upload:
            new_run = wandb.init(
                project="{}-{}".format(env_name, method),
                name="Baseline1-{}-{}-{}-{}".format(self.config.actor_learning_rate,
                                        self.config.batch_size, algo, seed),
                config={"learning_rate": self.config.actor_learning_rate,
                        "batch_size": self.config.batch_size,
                        "algo": algo,
                        "seed": seed}
            )


        for epoch in range(1, n_epoch + 1):
            # dict to add incremental mean losses to epoch
            epoch_loss = defaultdict(list)
            
            behavior_dataset = D4rlDataset(
                dataset,
                normalize_states=False,
                normalize_rewards=False,
                noise_scale=0.0,
                bootstrap=False)
            
            dataloader = DataLoader(behavior_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True, num_workers=4)
            dataloader = iter(dataloader)
            
            range_gen = tqdm(
                range(len(behavior_dataset) // self._config.batch_size),
                disable=not show_progress,
                desc=f"Epoch {int(epoch)}/{n_epoch}",
            )

            for itr in range_gen:
                # pick transitions
                states, actions, next_states, rewards, masks, _, _ = next(dataloader)

                states = states.to(self._device)
                actions = actions.to(self._device)
                rewards = rewards.to(self._device)
                next_states = next_states.to(self._device)
                masks = masks.to(self._device)

                loss = self.update(states, actions, next_states, rewards, masks)
                # record metrics
                for name, val in loss.items():
                    epoch_loss[name].append(val)

                # update progress postfix with losses
                if itr % 10 == 0:
                    mean_loss = {
                        k: np.mean(v) for k, v in epoch_loss.items()
                    }
                    range_gen.set_postfix(mean_loss)

                total_step += 1

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            # call epoch_callback if given
            if epoch_callback:
                epoch_callback(self, epoch, total_step)

            if epoch <= collect_epoch: # and epoch % 5 == 0
                if evaluators:
                    for name, evaluator in evaluators.items():
                        test_score_1, _, test_score, _, transitions = evaluator(self) # rollout 10条trajectory时长3.75s
                
                if collect:
                    for k,v in transitions.items():
                        dataset[k] = np.append(dataset[k], v, axis=0)

                   
                with open("{}/loss.csv".format(dir_path), 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch, np.mean(epoch_loss["actor_loss"]), 
                                        np.mean(epoch_loss["critic_loss"]),
                                        np.mean(epoch_loss["temp_loss"]),
                                        np.mean(epoch_loss["temp"]),
                                        test_score_1,
                                        test_score])
                
                if upload:
                    wandb.log({"epoch":epoch,
                            "outer_actor_loss": np.mean(epoch_loss["actor_loss"]),
                            "outer_critic_loss": np.mean(epoch_loss["critic_loss"]),
                            "temp_loss": np.mean(epoch_loss["temp_loss"]),
                            "temp": np.mean(epoch_loss["temp"]),
                            "Oracle_1.0": test_score_1,
                            "Oracle_0.995": test_score})
            
            elif epoch > collect_epoch: # and epoch % 5 == 0
                if evaluators:
                    for name, evaluator in evaluators.items():
                        test_score_1, test_score, transitions = evaluator(self) # rollout 10条trajectory时长3.75s
                
                
                with open("{}/loss.csv".format(dir_path), 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch, np.mean(epoch_loss["actor_loss"]), 
                                        np.mean(epoch_loss["critic_loss"]),
                                        np.mean(epoch_loss["temp_loss"]),
                                        np.mean(epoch_loss["temp"]),
                                        test_score_1,
                                        test_score])
                
                if upload:
                    wandb.log({"epoch":epoch,
                            "outer_actor_loss": np.mean(epoch_loss["actor_loss"]),
                            "outer_critic_loss": np.mean(epoch_loss["critic_loss"]),
                            "temp_loss": np.mean(epoch_loss["temp_loss"]),
                            "temp": np.mean(epoch_loss["temp"]),
                            "Oracle_1.0": test_score_1,
                            "Oracle_0.995": test_score})
            
                    
            # save model parameters
            if epoch % save_interval == 0:
                torch.save(self, "{}/model_{}.pt".format(dir_path,epoch))
            
            # scheduler_actor.step()
            # scheduler_critic.step()
                

    def fitter_2(
            self,
            method: str,
            dataset: Optional[Dict[str, float]],
            n_epoch: int,
            show_progress: bool = True,
            save_interval: int = 1,
            evaluators: Optional[Dict[str, EvaluatorProtocol]] = None,
            callback: Optional[Callable[[Self, int, int], None]] = None,
            epoch_callback: Optional[Callable[[Self, int, int], None]] = None,
            dir_path: str = None,
            seed: int = 0,
            env_name: str = None,
            collect_epoch: int = 50,
            estimator_lr: float = 0.003,
            estimator_lr_decay: float = 0.86,
            temp: float = 1.0,
            algo: str = None,
            upload: bool = False,
            collect: bool = False
        ) -> Generator[Tuple[int, Dict[str, float]], None, None]:
            """Iterate over epochs steps to train with the given dataset. At each
            iteration algo methods and properties can be changed or queried.

            .. code-block:: python

                for epoch, metrics in algo.fitter(episodes):
                    my_plot(metrics)
                    algo.save_model(my_path)

            Args:
                dataset: Offline dataset to train.
                n_steps: Number of steps to train.
                n_steps_per_epoch: Number of steps per epoch. This value will
                    be ignored when ``n_steps`` is ``None``.
                experiment_name: Experiment name for logging. If not passed,
                    the directory name will be `{class name}_{timestamp}`.
                with_timestamp: Flag to add timestamp string to the last of
                    directory name.
                logger_adapter: LoggerAdapterFactory object.
                show_progress: Flag to show progress bar for iterations.
                save_interval: Interval to save parameters.
                evaluators: List of evaluators.
                callback: Callable function that takes ``(algo, epoch, total_step)``
                    , which is called every step.
                epoch_callback: Callable function that takes
                    ``(algo, epoch, total_step)``, which is called at the end of
                    every epoch.

            Returns:
                Iterator yielding current epoch and metrics dict.
            """
            if self._impl is None:
                LOG.debug("Building models...")
                action_size = evaluators["environment"]._env.unwrapped.action_space.shape[0]
                observation_shape = evaluators["environment"]._env.unwrapped.observation_space.shape
                self.create_impl(observation_shape, action_size)
                LOG.debug("Models have been built.")
            else:
                LOG.warning("Skip building models since they're already built.")

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            with open("{}/loss.csv".format(dir_path), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Epoch", "actor_loss", "critic_loss", "temp_loss", "temp",
                                 "normalized_reward_incentive", "Oracle_1.0", "Oracle"])  # 写入标题

            # training loop
            total_step = 0
            reward_incentive_list = []
            normalized_reward_incentive = torch.tensor(0.0,dtype=torch.float32,device=self._device)

            # scheduler_actor = optim.lr_scheduler.StepLR(self._impl.modules.actor_optim, step_size=decay_epoch, gamma=lr_decay)
            # scheduler_critic = optim.lr_scheduler.StepLR(self._impl.modules.critic_optim, step_size=decay_epoch, gamma=lr_decay)

            if upload:
                new_run = wandb.init(
                    project="{}-{}".format(env_name, method),
                    name="Baseline2-{}-{}-{}-{}-{}".format(self.config.actor_learning_rate,
                                            self.config.batch_size, temp, algo, seed),
                    config={"actor_learning_rate": self.config.actor_learning_rate,
                            "batch_size": self.config.batch_size,
                            "temp": temp,
                            "algo": algo,
                            "seed": seed}
                )


            for epoch in range(1, n_epoch + 1):
                # dict to add incremental mean losses to epoch
                epoch_loss = defaultdict(list)

                behavior_dataset = D4rlDataset(
                dataset,
                normalize_states=False,
                normalize_rewards=False,
                noise_scale=0.0,
                bootstrap=False)
            
                dataloader = DataLoader(behavior_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True, num_workers=4)
                dataloader = iter(dataloader)
                
                range_gen = tqdm(
                    range(len(behavior_dataset) // self._config.batch_size),
                    disable=not show_progress,
                    desc=f"Epoch {int(epoch)}/{n_epoch}",
                )


                if len(reward_incentive_list) >= 10:
                    reward_incentive_reduce_mean = reward_incentive - torch.tensor(np.mean(reward_incentive_list))
                    normalized_reward_incentive = reward_incentive_reduce_mean / torch.std(torch.tensor(reward_incentive_list), unbiased=False)
                
                for itr in range_gen:
                    # pick transitions
                    states, actions, next_states, rewards, masks, _, _ = next(dataloader)

                    states = states.to(self._device)
                    actions = actions.to(self._device)
                    rewards = rewards.to(self._device)
                    next_states = next_states.to(self._device)
                    masks = masks.to(self._device)


                    loss = self.update(states, actions, next_states, rewards + temp * normalized_reward_incentive, masks)
                    # record metrics
                    for name, val in loss.items():
                        epoch_loss[name].append(val)

                    # update progress postfix with losses
                    if itr % 10 == 0:
                        mean_loss = {
                            k: np.mean(v) for k, v in epoch_loss.items()
                        }
                        range_gen.set_postfix(mean_loss)

                    total_step += 1

                    # call callback if given
                    if callback:
                        callback(self, epoch, total_step)

                # call epoch_callback if given
                if epoch_callback:
                    epoch_callback(self, epoch, total_step)

                if epoch <= collect_epoch: # and epoch % 5 == 0
                    if evaluators:
                        for name, evaluator in evaluators.items():
                            test_score_1, _, test_score, _, transitions = evaluator(self) # rollout 10条trajectory时长3.75s
                    
                    # Only collect transitions when epoch <= 100
                    # For epoch > 100, just evaluate with no collection
                    if collect:
                        for k,v in transitions.items():
                            dataset[k] = np.append(dataset[k], v, axis=0)


                    # generate reward incentive
                    save_policy(self,dir_path)
                    run(device=self._device.split(":")[-1],
                        env_name=env_name,
                        lr=estimator_lr,
                        policy_path="{}/policy.pkl".format(dir_path),
                        lr_decay=estimator_lr_decay,
                        seed=seed,
                        algo=algo)

                    estimate = pd.read_csv("{}/ope.csv".format(dir_path)).iloc[0,0]
                    reward_incentive = np.float32((estimate - test_score * (1-evaluators["environment"]._gamma)) ** 2)
                    reward_incentive = -reward_incentive

                    # store new reward_incentive into reward_list
                    if reward_incentive not in reward_incentive_list:
                        reward_incentive_list.append(reward_incentive)


                    with open("{}/loss.csv".format(dir_path), 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([epoch, np.mean(epoch_loss["actor_loss"]), 
                                            np.mean(epoch_loss["critic_loss"]),
                                            np.mean(epoch_loss["temp_loss"]),
                                            np.mean(epoch_loss["temp"]),
                                            normalized_reward_incentive,
                                            test_score_1,
                                            test_score])
                    
                    if upload:
                        wandb.log({"epoch":epoch,
                                "outer_actor_loss": np.mean(epoch_loss["actor_loss"]),
                                "outer_critic_loss": np.mean(epoch_loss["critic_loss"]),
                                "temp_loss": np.mean(epoch_loss["temp_loss"]),
                                "temp": np.mean(epoch_loss["temp"]),
                                "normalized_reward_incentive": normalized_reward_incentive,
                                "Oracle_1.0": test_score_1,
                                "Oracle_0.995": test_score})
                
                # difference is we do not have real value, so we use real value at 100 epoch
                elif epoch > collect_epoch: # and epoch % 5 == 0
                    if evaluators:
                        for name, evaluator in evaluators.items():
                            test_score_1_after_100, _, test_score_after_100, _, transitions = evaluator(self) # rollout 10条trajectory时长3.75s

                    # generate reward incentive
                    save_policy(self,dir_path)
                    run(device=self._device.split(":")[-1],
                        env_name=env_name,
                        lr=estimator_lr,
                        policy_path="{}/policy.pkl".format(dir_path),
                        lr_decay=estimator_lr_decay,
                        seed=seed,
                        algo=algo)

                    estimate = pd.read_csv("{}/ope.csv".format(dir_path)).iloc[0,0]
                    reward_incentive = np.float32((estimate - test_score * (1-evaluators["environment"]._gamma)) ** 2)
                    reward_incentive = -reward_incentive

                    # store new reward_incentive into reward_list
                    if reward_incentive not in reward_incentive_list:
                        reward_incentive_list.append(reward_incentive)


                    with open("{}/loss.csv".format(dir_path), 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([epoch, np.mean(epoch_loss["actor_loss"]), 
                                            np.mean(epoch_loss["critic_loss"]),
                                            np.mean(epoch_loss["temp_loss"]),
                                            np.mean(epoch_loss["temp"]),
                                            normalized_reward_incentive,
                                            test_score_1_after_100,
                                            test_score_after_100])
                    
                    if upload:
                        wandb.log({"epoch":epoch,
                                "outer_actor_loss": np.mean(epoch_loss["actor_loss"]),
                                "outer_critic_loss": np.mean(epoch_loss["critic_loss"]),
                                "temp_loss": np.mean(epoch_loss["temp_loss"]),
                                "temp": np.mean(epoch_loss["temp"]),
                                "normalized_reward_incentive": normalized_reward_incentive,
                                "Oracle_1.0": test_score_1_after_100,
                                "Oracle_0.995": test_score_after_100})

                        
                # save model parameters
                if epoch % save_interval == 0:
                    torch.save(self, "{}/model_{}.pt".format(dir_path,epoch))
                
                # scheduler_actor.step()
                # scheduler_critic.step() 


    def fit_online(
        self,
        env: GymEnv,
        buffer: Optional[ReplayBuffer] = None,
        explorer: Optional[Explorer] = None,
        n_steps: int = 1000000,
        n_steps_per_epoch: int = 10000,
        update_interval: int = 1,
        update_start_step: int = 0,
        random_steps: int = 0,
        eval_env: Optional[GymEnv] = None,
        eval_epsilon: float = 0.0,
        save_interval: int = 1,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logger_adapter: LoggerAdapterFactory = FileAdapterFactory(),
        show_progress: bool = True,
        callback: Optional[Callable[[Self, int, int], None]] = None,
    ) -> None:
        """Start training loop of online deep reinforcement learning.

        Args:
            env: Gym-like environment.
            buffer : Replay buffer.
            explorer: Action explorer.
            n_steps: Number of total steps to train.
            n_steps_per_epoch: Number of steps per epoch.
            update_interval: Number of steps per update.
            update_start_step: Steps before starting updates.
            random_steps: Steps for the initial random explortion.
            eval_env: Gym-like environment. If None, evaluation is skipped.
            eval_epsilon: :math:`\\epsilon`-greedy factor during evaluation.
            save_interval: Number of epochs before saving models.
            experiment_name: Experiment name for logging. If not passed,
                the directory name will be ``{class name}_online_{timestamp}``.
            with_timestamp: Flag to add timestamp string to the last of
                directory name.
            logger_adapter: LoggerAdapterFactory object.
            show_progress: Flag to show progress bar for iterations.
            callback: Callable function that takes ``(algo, epoch, total_step)``
                , which is called at the end of epochs.
        """

        # create default replay buffer
        if buffer is None:
            buffer = create_fifo_replay_buffer(1000000)

        # check action-space
        assert_action_space_with_env(self, env)


        # initialize algorithm parameters
        build_scalers_with_env(self, env)

        # setup algorithm
        if self.impl is None:
            LOG.debug("Building model...")
            self.build_with_env(env)
            LOG.debug("Model has been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        frames = []

        # switch based on show_progress flag
        xrange = trange if show_progress else range

        # start training loop
        observation, _ = env.reset()
        rollout_return = 0.0
        for total_step in xrange(1, n_steps + 1):
            frames.append(env.render())
            # sample exploration action
            if total_step < random_steps:
                action = env.action_space.sample()
            elif explorer:
                x = observation.reshape((1,) + observation.shape)
                action = explorer.sample(self, x, total_step)[0]
            else:
                action = self.sample_action(
                    np.expand_dims(observation, axis=0)
                )[0]

            # step environment
            (
                next_observation,
                reward,
                terminal,
                truncated, # 
                _,
            ) = env.step(action)
            rollout_return += float(reward)

            clip_episode = terminal or truncated

            # store observation
            buffer.append(observation, action, float(reward))

            # reset if terminated
            if clip_episode:
                buffer.clip_episode(terminal)
                observation, _ = env.reset()
                print("Rollout return:", rollout_return)
                rollout_return = 0.0
            else:
                observation = next_observation

            # psuedo epoch count
            epoch = total_step // n_steps_per_epoch

            if (
                total_step > update_start_step
                and buffer.transition_count > self.batch_size
            ):
                if total_step % update_interval == 0:
                    # sample mini-batch
                    batch = buffer.sample_transition_batch(
                        self.batch_size
                    )
                    observations = batch.observations
                    actions = batch.actions
                    rewards = batch.rewards
                    next_observations = batch.next_observations
                    terminals = batch.terminals

                    states = torch.tensor(observations, dtype=torch.float32,device=self._device)
                    actions = torch.tensor(actions, dtype=torch.float32,device=self._device)
                    next_states = torch.tensor(next_observations, dtype=torch.float32,device=self._device)
                    rewards = torch.tensor(rewards, dtype=torch.float32,device=self._device)
                    masks = torch.tensor(terminals, dtype=torch.float32,device=self._device)

                    # update parameters
                    loss = self.update(states=states,
                                       actions=actions,
                                       next_states=next_states,
                                       rewards=rewards,
                                       masks=masks)

                    # record metrics

            # call callback if given
            if callback:
                callback(self, epoch, total_step)

            if epoch > 0 and total_step % n_steps_per_epoch == 0:
                # evaluation
                if eval_env:
                    eval_score = evaluate_qlearning_with_environment(
                        self, eval_env, epsilon=eval_epsilon, gamma=0.995, n_trials=10
                    )

                    print("Eval score:", eval_score[0])
                # if epoch % save_interval == 0:
                #     logger.save_model(total_step, self, 0)

        # clip the last episode
        buffer.clip_episode(False)


    def collect(
        self,
        env: GymEnv,
        buffer: Optional[ReplayBuffer] = None,
        explorer: Optional[Explorer] = None,
        deterministic: bool = False,
        n_steps: int = 1000000,
        show_progress: bool = True,
    ) -> ReplayBuffer:
        """Collects data via interaction with environment.

        If ``buffer`` is not given, ``ReplayBuffer`` will be internally created.

        Args:
            env: Fym-like environment.
            buffer: Replay buffer.
            explorer: Action explorer.
            deterministic: Flag to collect data with the greedy policy.
            n_steps: Number of total steps to train.
            show_progress: Flag to show progress bar for iterations.

        Returns:
            Replay buffer with the collected data.
        """
        # create default replay buffer
        if buffer is None:
            buffer = create_fifo_replay_buffer(1000000, env=env)

        # check action-space
        assert_action_space_with_env(self, env)

        # initialize algorithm parameters
        build_scalers_with_env(self, env)

        # setup algorithm
        if self.impl is None:
            LOG.debug("Building model...")
            self.build_with_env(env)
            LOG.debug("Model has been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        # switch based on show_progress flag
        xrange = trange if show_progress else range

        # start training loop
        observation, _ = env.reset()
        for total_step in xrange(1, n_steps + 1):
            # sample exploration action
            if deterministic:
                action = self.predict(np.expand_dims(observation, axis=0))[0]
            else:
                if explorer:
                    x = observation.reshape((1,) + observation.shape)
                    action = explorer.sample(self, x, total_step)[0]
                else:
                    action = self.sample_action(
                        np.expand_dims(observation, axis=0)
                    )[0]

            # step environment
            next_observation, reward, terminal, truncated, _ = env.step(action)

            clip_episode = terminal or truncated

            # store observation
            buffer.append(observation, action, float(reward))

            # reset if terminated
            if clip_episode:
                buffer.clip_episode(terminal)
                observation, _ = env.reset()
            else:
                observation = next_observation

        # clip the last episode
        buffer.clip_episode(False)

        return buffer

    # def update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
    def update(self, states, actions, next_states, rewards, masks) -> Dict[str, float]:
        """Update parameters with mini-batch of data.

        Args:
            batch: Mini-batch data.

        Returns:
            Dictionary of metrics.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        # torch_batch = TorchMiniBatch.from_batch(
        #     batch=batch,
        #     device=self._device,
        #     observation_scaler=self._config.observation_scaler,
        #     action_scaler=self._config.action_scaler,
        #     reward_scaler=self._config.reward_scaler,
        # )
        loss = self._impl.inner_update(states, actions, next_states, rewards, masks, self._grad_step)
        self._grad_step += 1
        return loss

    def copy_policy_from(
        self, algo: "QLearningAlgoBase[QLearningAlgoImplBase, LearnableConfig]"
    ) -> None:
        """Copies policy parameters from the given algorithm.

        .. code-block:: python

            # pretrain with static dataset
            cql = d3rlpy.algos.CQL()
            cql.fit(dataset, n_steps=100000)

            # transfer to online algorithm
            sac = d3rlpy.algos.SAC()
            sac.create_impl(cql.observation_shape, cql.action_size)
            sac.copy_policy_from(cql)

        Args:
            algo: Algorithm object.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert isinstance(algo.impl, QLearningAlgoImplBase)
        self._impl.copy_policy_from(algo.impl)

    def copy_policy_optim_from(
        self, algo: "QLearningAlgoBase[QLearningAlgoImplBase, LearnableConfig]"
    ) -> None:
        """Copies policy optimizer states from the given algorithm.

        .. code-block:: python

            # pretrain with static dataset
            cql = d3rlpy.algos.CQL()
            cql.fit(dataset, n_steps=100000)

            # transfer to online algorithm
            sac = d3rlpy.algos.SAC()
            sac.create_impl(cql.observation_shape, cql.action_size)
            sac.copy_policy_optim_from(cql)

        Args:
            algo: Algorithm object.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert isinstance(algo.impl, QLearningAlgoImplBase)
        self._impl.copy_policy_optim_from(algo.impl)

    def copy_q_function_from(
        self, algo: "QLearningAlgoBase[QLearningAlgoImplBase, LearnableConfig]"
    ) -> None:
        """Copies Q-function parameters from the given algorithm.

        .. code-block:: python

            # pretrain with static dataset
            cql = d3rlpy.algos.CQL()
            cql.fit(dataset, n_steps=100000)

            # transfer to online algorithmn
            sac = d3rlpy.algos.SAC()
            sac.create_impl(cql.observation_shape, cql.action_size)
            sac.copy_q_function_from(cql)

        Args:
            algo: Algorithm object.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert isinstance(algo.impl, QLearningAlgoImplBase)
        self._impl.copy_q_function_from(algo.impl)

    def copy_q_function_optim_from(
        self, algo: "QLearningAlgoBase[QLearningAlgoImplBase, LearnableConfig]"
    ) -> None:
        """Copies Q-function optimizer states from the given algorithm.

        .. code-block:: python

            # pretrain with static dataset
            cql = d3rlpy.algos.CQL()
            cql.fit(dataset, n_steps=100000)

            # transfer to online algorithm
            sac = d3rlpy.algos.SAC()
            sac.create_impl(cql.observation_shape, cql.action_size)
            sac.copy_policy_optim_from(cql)

        Args:
            algo: Algorithm object.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert isinstance(algo.impl, QLearningAlgoImplBase)
        self._impl.copy_q_function_optim_from(algo.impl)

    def reset_optimizer_states(self) -> None:
        """Resets optimizer states.

        This is especially useful when fine-tuning policies with setting inital
        optimizer states.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        self._impl.reset_optimizer_states()
