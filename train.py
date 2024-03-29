import argparse
import pdb
import numpy as np
import d3rlpy
import pickle
import gym
import torch
from d3rlpy.dataset import ReplayBuffer_, D4rlDataset, get_cartpole, get_pendulum, infinite_loader
from d3rlpy.algos.qlearning.model_sac import Model
from torch.utils.data import DataLoader

# 0, 1, 1234
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Pendulum-random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=str, default="cuda:3")
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--decay_epoch", type=int, default=10)
    parser.add_argument("--lr_decay", type=float, default=0.85)
    

    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--ratio", type=int, default=1)
    # We select one pre-trained estimator
    parser.add_argument("--estimator_lr", type=float, default=0.003)
    parser.add_argument("--estimator_lr_decay", type=float, default=0.8)

    parser.add_argument("--collect_epoch", type=int, default=100)
    parser.add_argument("--n_epoch", type=int, default=200)
    parser.add_argument("--n_episodes", type=int, default=2)
    parser.add_argument("--algo", type=str, default="iw") # "iw" or "mb"
    parser.add_argument("--method", type=str, default="baseline2") # "new" or "baseline"
    parser.add_argument("--save_dir", type=str, default="/home/xiaoan/checkpoints")
    parser.add_argument('--upload', dest='upload', action='store_true', help='Enable upload')
    parser.add_argument('--no-upload', dest='upload', action='store_false', help='Disable upload')
    parser.add_argument('--collect', dest='collect', action='store_true', help='Enable collect')
    parser.add_argument('--no-collect', dest='collect', action='store_false', help='Disable collect')
    parser.set_defaults(upload=True)
    parser.set_defaults(collect=True)


    args = parser.parse_args()

    if "Pendulum" in args.dataset:
        env = gym.make("Pendulum-v1")
        d3rlpy.seed(args.seed)
        d3rlpy.envs.seed_env(env, args.seed)
        d4rl_dataset = get_pendulum(dataset_type=args.dataset.split("-")[1])
    elif "CartPole" in args.dataset:
        env = gym.make("CartPole-v1")
        d3rlpy.seed(args.seed)
        d3rlpy.envs.seed_env(env, args.seed)
        d4rl_dataset = get_cartpole(dataset_type=args.dataset.split("-")[1])
    else:
        env = gym.make(args.dataset)
        d3rlpy.seed(args.seed)
        d3rlpy.envs.seed_env(env, args.seed)
        d4rl_dataset = env.get_dataset()


    sac1 = d3rlpy.algos.SACConfig(
        actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr,
        temp_learning_rate=3e-4,
        batch_size=256,
    ).create(device=args.gpu)

    sac2 = d3rlpy.algos.SACConfig(
        actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr,
        temp_learning_rate=3e-4,
        batch_size=256,
    ).create(device=args.gpu)

    if args.method == "new":
        buffer = ReplayBuffer_(capacity=1280000)
        model = Model(sac1=sac1,sac2=sac2)
        model.fit(
            dataset=d4rl_dataset,
            buffer=buffer,
            n_epoch=args.n_epoch,
            save_interval=10,
            evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env,gamma=0.995,n_trials=args.n_episodes)},
            dir_path="{}/{}/{}/{}-{}-{}-{}-{}-{}/seed{}/{}-{}-{}".format(args.save_dir, args.method, args.dataset, 
                                                                    args.actor_lr,args.critic_lr,
                                                                    args.decay_epoch, args.lr_decay, args.ratio, args.temp,
                                                                    args.seed, args.algo, args.estimator_lr, 
                                                                    args.estimator_lr_decay),
            seed = args.seed,
            env_name = args.dataset,
            decay_epoch = args.decay_epoch,
            lr_decay = args.lr_decay,
            collect_epoch = args.collect_epoch,
            estimator_lr = args.estimator_lr,
            estimator_lr_decay = args.estimator_lr_decay,
            algo = args.algo,
            ratio = args.ratio,
            temp = args.temp,
            upload = args.upload,
            collect = args.collect
        )
    elif args.method == "baseline1":
        sac1.fit(
            method=args.method,
            dataset=d4rl_dataset,
            n_epoch=args.n_epoch,
            save_interval=10,
            evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env,gamma=0.995,n_trials=args.n_episodes)},
            dir_path="{}/{}/{}/{}-{}-{}-{}/seed{}".format(args.save_dir, args.method, args.dataset, 
                                                          args.actor_lr, args.critic_lr, args.decay_epoch,
                                                          args.lr_decay, args.seed),
            seed = args.seed,
            env_name = args.dataset,
            decay_epoch=args.decay_epoch,
            lr_decay=args.lr_decay,
            collect_epoch = args.collect_epoch,
            estimator_lr = args.estimator_lr,
            estimator_lr_decay = args.estimator_lr_decay,
            temp=args.temp,
            temp_decay=args.temp_decay,
            algo = args.algo,
            upload = args.upload,
            collect = args.collect
        )
    elif args.method == "baseline2":
        sac1.fit(
            method=args.method,
            dataset=d4rl_dataset,
            n_epoch=args.n_epoch,
            save_interval=10,
            evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env,gamma=0.995,n_trials=args.n_episodes)},
            dir_path="{}/{}/{}/{}-{}-{}-{}-{}/seed{}".format(args.save_dir, args.method, args.dataset, 
                                                          args.actor_lr, args.critic_lr, args.decay_epoch,
                                                          args.lr_decay, args.temp, args.seed),
            seed = args.seed,
            env_name = args.dataset,
            decay_epoch=args.decay_epoch,
            lr_decay=args.lr_decay,
            collect_epoch = args.collect_epoch,
            estimator_lr = args.estimator_lr,
            estimator_lr_decay = args.estimator_lr_decay,
            temp=args.temp,
            algo = args.algo,
            upload = args.upload,
            collect = args.collect
        )
    


if __name__ == "__main__":
    main()