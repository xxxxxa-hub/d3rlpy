from abc import abstractmethod
import pdb
import pickle
from d3rlpy import *
import time
import csv
import os
import sys
from glob import glob
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
from d3rlpy.base import DeviceArg
import random
import numpy as np
import torch
from torch import nn
# from tqdm.auto import tqdm, trange
from typing_extensions import Self
from ...base import ImplBase, LearnableBase, LearnableConfig, save_config
from .base import QLearningAlgoBase, QLearningAlgoImplBase
from ...constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ...dataset import (
    ReplayBuffer,
    ReplayBuffer_,
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
from ...models.torch import Policy, build_squashed_gaussian_distribution
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
import math
from ...models.builders import (
    create_categorical_policy,
    create_continuous_q_function,
    create_discrete_q_function,
    create_normal_policy,
    create_parameter,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.optimizers import OptimizerFactory, make_optimizer_field
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...types import Shape
from .base import QLearningAlgoBase
from .sac import SAC,SACConfig
from .torch.sac_impl import (
    DiscreteSACImpl,
    DiscreteSACModules,
    SACImpl,
    SACModules,
)
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
from tqdm import tqdm
import pandas as pd
from ...utils import save_policy,run
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader

class Model():
    def __init__(self, sac1: SAC, sac2: SAC):
        self.sac1 = sac1
        self.sac2 = sac2

    def fit(
        self,
        dataset: Optional[Dict[str, float]],
        buffer: Optional[ReplayBuffer_] = None,
        n_epoch: int = 100,
        show_progress: bool = True,
        save_interval: int = 1,
        evaluators: Optional[Dict[str, EvaluatorProtocol]] = None,
        callback: Optional[Callable[[Self, int, int], None]] = None,
        epoch_callback: Optional[Callable[[Self, int, int], None]] = None,
        dir_path: str = None,
        seed: int = 0,
        env_name: Optional[str] = None,
        decay_epoch: int = 5,
        lr_decay: float = 0.96,
        estimator_lr: float = 0.003,
        estimator_lr_decay: float = 0.86,
        algo: Optional[str] = None,
        ratio: int = 1,
        temp: float = 1.0,
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
        self.fitter(
            dataset,
            buffer,
            n_epoch,
            show_progress,
            save_interval,
            evaluators,
            callback,
            epoch_callback,
            dir_path,
            seed,
            env_name,
            decay_epoch,
            lr_decay,
            estimator_lr,
            estimator_lr_decay,
            algo,
            ratio,
            temp,
            upload,
            collect
        )


    def fitter(
        self,
        dataset: Optional[Dict[str, float]],
        buffer: Optional[ReplayBuffer_] = None,
        n_epoch: int = 100,
        show_progress: bool = True,
        save_interval: int = 1,
        evaluators: Optional[Dict[str, EvaluatorProtocol]] = None,
        callback: Optional[Callable[[Self, int, int], None]] = None,
        epoch_callback: Optional[Callable[[Self, int, int], None]] = None,
        dir_path: str = None,
        seed: int = 0,
        env_name: Optional[str] = None,
        decay_epoch: int = 5,
        lr_decay: float = 0.96,
        collect_epoch: int = 30,
        estimator_lr: float = 0.003,
        estimator_lr_decay: float = 0.86,
        algo: Optional[str] = None,
        ratio: int = 1,
        temp: float = 1.0,
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
        # instantiate implementation
        if self.sac1._impl is None:
            LOG.debug("Building model_1 ...")
            action_size = evaluators["environment"]._env.unwrapped.action_space.shape[0]
            observation_shape = evaluators["environment"]._env.unwrapped.observation_space.shape
            self.sac1.create_impl(observation_shape, action_size)
            LOG.debug("Model_1 have been built.")
        else:
            LOG.warning("Skip building models since they're already built.")


        if self.sac2._impl is None:
            LOG.debug("Building model_2 ...")
            action_size = evaluators["environment"]._env.unwrapped.action_space.shape[0]
            observation_shape = evaluators["environment"]._env.unwrapped.observation_space.shape
            self.sac2.create_impl((observation_shape[0] + action_size,), 1)
            LOG.debug("Model_2 have been built.")
        else:
            LOG.warning("Skip building models since they're already built.")


        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        

        with open("{}/loss.csv".format(dir_path), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "actor_loss", "critic_loss", "temp_loss", "temp", "inner_actor_loss", "inner_critic_loss", "inner_temp_loss", "inner_temp", "Oracle_1.0", "Oracle"])  # 写入标题
        with open("{}/estimate.csv".format(dir_path), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Estimate"]) 


        # training loop
        total_step = 0
        A = torch.zeros((self.sac1._config.batch_size,1),device=self.sac1._device)

        scheduler_actor_1 = optim.lr_scheduler.StepLR(self.sac1._impl.modules.actor_optim, step_size=decay_epoch, gamma=lr_decay)
        scheduler_critic_1 = optim.lr_scheduler.StepLR(self.sac1._impl.modules.critic_optim, step_size=decay_epoch, gamma=lr_decay)

        scheduler_actor_2 = optim.lr_scheduler.StepLR(self.sac2._impl.modules.actor_optim, step_size=decay_epoch, gamma=lr_decay)
        scheduler_critic_2 = optim.lr_scheduler.StepLR(self.sac2._impl.modules.critic_optim, step_size=decay_epoch, gamma=lr_decay)

        if upload:
            new_run = wandb.init(
                project="{}-{}".format(env_name, 2),
                name="New-{}-{}-{}-{}-{}-{}".format(self.sac1.config.actor_learning_rate,
                                        self.sac1.config.critic_learning_rate,
                                        decay_epoch, lr_decay,
                                        ratio, algo),
                config={"actor_learning_rate": self.sac1.config.actor_learning_rate,
                        "critic_learning_rate": self.sac1.config.critic_learning_rate,
                        "decay_epoch": decay_epoch,
                        "lr_decay": lr_decay,
                        "ratio": ratio,
                        "algo": algo}
            )

        for epoch in range(1, n_epoch + 1):
            print("Epoch" + str(epoch) + "...")
            '''
            store the batch.observations in the list. After updating for one epoch, calculate 
            the S_ by inputing elements in S_ to the new policy.
            '''
            # if epoch % 5 == 1:
            S_ = [] # stores s'_i in xxx epochs

            # dict to add incremental mean losses to epoch
            epoch_loss = defaultdict(list)
            inner_epoch_loss = defaultdict(list)

            behavior_dataset = D4rlDataset(
                dataset,
                normalize_states=False,
                normalize_rewards=False,
                noise_scale=0.0,
                bootstrap=False)
            
            dataloader = DataLoader(behavior_dataset, batch_size=256, shuffle=True, drop_last=True, num_workers=4)
            dataloader = iter(dataloader)

            n_steps_per_epoch = len(behavior_dataset) // self._config.batch_size

            range_gen = tqdm(
                range(n_steps_per_epoch),
                disable=not show_progress,
                desc=f"Epoch {int(epoch)}/{n_epoch}",
            )

            if buffer.is_full():
                buffer.extended_count -= self.sac1._config.batch_size * n_steps_per_epoch

            for itr in range_gen:
                states, actions, next_states, rewards, masks, _, _ = next(dataloader)

                # Generate Transitions
                states = states.to(self.sac1._device)
                actions = actions.to(self.sac1._device)
                rewards = rewards.to(self.sac1._device)
                next_states = next_states.to(self.sac1._device)
                masks = masks.to(self.sac1._device)

                with torch.no_grad():
                    # Generate S
                    dist = build_squashed_gaussian_distribution(
                        self.sac1._impl._modules.policy(next_states)
                    )
                    next_action, _ = dist.sample_with_log_prob()
                    S = torch.cat((next_states, next_action), dim=1)

                    # generate A
                    dist = build_squashed_gaussian_distribution(
                        self.sac2._impl._modules.policy(S)
                    )
                    A, _ = dist.sample_with_log_prob()
                    
                    # Transfer tensor to ndarray
                    S_detach_cpu = S.detach().cpu()# .numpy()
                    A_detach_cpu = A.detach().cpu()# .numpy()
                    S_.append(states)
                    Masks = torch.ones((self.sac1._config.batch_size,),dtype=torch.float32)
                    '''
                    terminal means that trajectory ends or not,
                    There is no end in our case because policy can be updated forever unlike in the real env where there is an end.
                    '''
                    # Store into buffer
                    buffer.store(S_detach_cpu,A_detach_cpu,Masks)

                rewards_with_incentive = rewards + A.squeeze()
                loss = self.sac1.update(states, actions, next_states, rewards_with_incentive, masks)

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

            # save the returns when gamma is 1.0 and 0.995
            if epoch <= collect_epoch: # epoch % 5 ==0:
                if evaluators:
                    for name, evaluator in evaluators.items():
                        test_score_1, _, test_score, _, transitions = evaluator(self.sac1)
                
                if collect:
                    for k,v in transitions.items():
                        dataset[k] = np.append(dataset[k], v, axis=0)

                # Generate R
                save_policy(self.sac1,dir_path)
                run(device=self.sac1._device.split(":")[-1],
                    env_name=env_name,
                    lr=estimator_lr,
                    policy_path="{}/policy.pkl".format(dir_path),
                    lr_decay=estimator_lr_decay,
                    seed=seed,
                    algo=algo)

                # reward - mean(rewards) should be calculated when we start to sample rather than when we collect transitions.
                # store the rewards in the buffer, and we normalize rewards based on every seen rewards.
                estimate = pd.read_csv("{}/ope.csv".format(dir_path)).iloc[0,0]
                inner_reward = np.float32((estimate - test_score * (1-evaluators["environment"]._gamma)) ** 2)
                inner_reward = -inner_reward

                # store new inner_reward into reward_list
                if inner_reward not in buffer.reward_list:
                    buffer.reward_list.append(inner_reward)
            
                with open("{}/estimate.csv".format(dir_path), 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch, estimate])
                
                # generate S_
                # index = min(buffer.extended_count,buffer.capacity - self.sac1._config.batch_size * n_steps_per_epoch * 5)
                # index = buffer.extended_count
                for batch_state in tqdm(S_):
                    dist = build_squashed_gaussian_distribution(
                                self.sac1._impl._modules.policy(batch_state)
                            )
                    batch_action, _ = dist.sample_with_log_prob()
                    batch_action_detach = batch_action.detach()
                    batch_state_detach = batch_state.detach()
                    next_states_tensor = torch.cat([batch_state_detach,batch_action_detach],axis=1).cpu()
                    # inner_reward = torch.tensor(inner_reward,dtype=torch.float32,device=self.sac1._device)
                    inner_reward_tensor = torch.full((self.sac1._config.batch_size,), 
                                              inner_reward,dtype=torch.float32)
                    # for i in range(self.sac1._config.batch_size):
                    #     buffer.buffer[index].extend([temp[i],inner_reward])
                    #     if buffer.extended_count < buffer.capacity:
                    #         buffer.extended_count += 1
                    #     index += 1
                    
                    buffer.synchronize(next_states_tensor,inner_reward_tensor)
                inner_dataloader = DataLoader(buffer, batch_size=256, shuffle=True, drop_last=True, num_workers=1)
                inner_data_iterator = iter(inner_dataloader)
            
            elif epoch > collect_epoch:
                if evaluators:
                    for name, evaluator in evaluators.items():
                        test_score_1_after_100, test_score_after_100, transitions = evaluator(self.sac1)

                # Generate R
                save_policy(self.sac1,dir_path)
                run(device=self.sac1._device.split(":")[-1],
                    env_name=env_name,
                    lr=estimator_lr,
                    policy_path="{}/policy.pkl".format(dir_path),
                    lr_decay=estimator_lr_decay,
                    seed=seed,
                    algo=algo)

                # reward - mean(rewards) should be calculated when we start to sample rather than when we collect transitions.
                # store the rewards in the buffer, and we normalize rewards based on every seen rewards.
                estimate = pd.read_csv("{}/ope.csv".format(dir_path)).iloc[0,0]
                inner_reward = np.float32((estimate - test_score * (1-evaluators["environment"]._gamma)) ** 2)
                inner_reward = -inner_reward

                if inner_reward not in buffer.reward_list:
                    buffer.reward_list.append(inner_reward)

                with open("{}/estimate.csv".format(dir_path), 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch, estimate])
                
                # generate S_
                # index = min(buffer.extended_count,buffer.capacity - self.sac1._config.batch_size * n_steps_per_epoch * 5)
                # index = buffer.extended_count
                for batch_state in tqdm(S_):
                    dist = build_squashed_gaussian_distribution(
                                self.sac1._impl._modules.policy(batch_state)
                            )
                    batch_action, _ = dist.sample_with_log_prob()
                    batch_action_detach = batch_action.detach()
                    batch_state_detach = batch_state.detach()
                    next_states_tensor = torch.cat([batch_state_detach,batch_action_detach],axis=1).cpu()
                    inner_reward_tensor = torch.full((self.sac1._config.batch_size,), 
                                              inner_reward,dtype=torch.float32)
                    
                    buffer.synchronize(next_states_tensor,inner_reward_tensor)
                inner_dataloader = DataLoader(buffer, batch_size=256, shuffle=True, drop_last=True, num_workers=1)
                inner_data_iterator = iter(inner_dataloader)


            # update inner loop
            if len(buffer.reward_list) >= 10:
                for itr in tqdm(range(n_steps_per_epoch * ratio)):
                    states_,actions_,rewards_,next_states_,masks_ = next(inner_data_iterator)

                    states_ = states_.to(self.sac1._device)
                    actions_ = actions_.to(self.sac1._device)
                    rewards_ = rewards_.to(self.sac1._device)
                    next_states_ = next_states_.to(self.sac1._device)
                    masks_ = masks_.to(self.sac1._device)
                    # normalize rewards based on reward_list
                    rewards_ = rewards_ - torch.tensor(np.mean(buffer.reward_list))
                    rewards_ = rewards_ / torch.std(torch.tensor(buffer.reward_list), unbiased=False)
                    
                    loss = self.sac2.update(states_,actions_,next_states_,rewards_,masks_)
                    
                    for name, val in loss.items():
                        inner_epoch_loss["inner_" + name].append(val)

            if epoch <= 100:
                with open("{}/loss.csv".format(dir_path), 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch, np.mean(epoch_loss["actor_loss"]), 
                                        np.mean(epoch_loss["critic_loss"]),
                                        np.mean(epoch_loss["temp_loss"]),
                                        np.mean(epoch_loss["temp"]),
                                        np.mean(inner_epoch_loss["inner_actor_loss"]), 
                                        np.mean(inner_epoch_loss["inner_critic_loss"]), 
                                        np.mean(inner_epoch_loss["inner_temp_loss"]),
                                        np.mean(inner_epoch_loss["inner_temp"]),
                                        test_score_1,
                                        test_score])
                    
                if upload:
                    wandb.log({"epoch":epoch,
                            "outer_actor_loss": np.mean(epoch_loss["actor_loss"]),
                            "outer_critic_loss": np.mean(epoch_loss["critic_loss"]),
                            "temp_loss": np.mean(epoch_loss["temp_loss"]),
                            "temp": np.mean(epoch_loss["temp"]),
                            "inner_actor_loss": np.mean(inner_epoch_loss["inner_actor_loss"]), 
                            "inner_critic_loss": np.mean(inner_epoch_loss["inner_critic_loss"]),
                            "inner_temp_loss": np.mean(inner_epoch_loss["inner_temp_loss"]),
                            "inner_temp": np.mean(inner_epoch_loss["inner_temp"]),
                            "Oracle_1.0": test_score_1,
                            "Oracle_0.995": test_score})
            elif epoch > 100:
                with open("{}/loss.csv".format(dir_path), 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch, np.mean(epoch_loss["actor_loss"]), 
                                        np.mean(epoch_loss["critic_loss"]),
                                        np.mean(epoch_loss["temp_loss"]),
                                        np.mean(epoch_loss["temp"]),
                                        np.mean(inner_epoch_loss["inner_actor_loss"]), 
                                        np.mean(inner_epoch_loss["inner_critic_loss"]), 
                                        np.mean(inner_epoch_loss["inner_temp_loss"]),
                                        np.mean(inner_epoch_loss["inner_temp"]),
                                        test_score_1_after_100,
                                        test_score_after_100])
                    
                if upload:
                    wandb.log({"epoch":epoch,
                            "outer_actor_loss": np.mean(epoch_loss["actor_loss"]),
                            "outer_critic_loss": np.mean(epoch_loss["critic_loss"]),
                            "temp_loss": np.mean(epoch_loss["temp_loss"]),
                            "temp": np.mean(epoch_loss["temp"]),
                            "inner_actor_loss": np.mean(inner_epoch_loss["inner_actor_loss"]), 
                            "inner_critic_loss": np.mean(inner_epoch_loss["inner_critic_loss"]),
                            "inner_temp_loss": np.mean(inner_epoch_loss["inner_temp_loss"]),
                            "inner_temp": np.mean(inner_epoch_loss["inner_temp"]),
                            "Oracle_1.0": test_score_1_after_100,
                            "Oracle_0.995": test_score_after_100})
            

            # Save model parameters
            if epoch % save_interval == 0:
                torch.save(self.sac1, "{}/model_{}_1.pt".format(dir_path,epoch))
                torch.save(self.sac2, "{}/model_{}_2.pt".format(dir_path,epoch))


            scheduler_actor_1.step()
            scheduler_critic_1.step()
            scheduler_actor_2.step()
            scheduler_critic_2.step()