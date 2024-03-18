import argparse
import pdb
import numpy as np
import d3rlpy
import pickle
import gym
import torch
from d3rlpy.dataset import ReplayBuffer_, D4rlDataset, get_pendulum, infinite_loader
from d3rlpy.algos.qlearning.model_sac import Model
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Pendulum-replay")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=str, default="cuda:0")
    parser.add_argument("--actor_lr", type=float, default=0.001)
    parser.add_argument("--critic_lr", type=float, default=0.001)
    parser.add_argument("--decay_epoch", type=int, default=1)
    parser.add_argument("--lr_decay", type=float, default=1.0)
    parser.add_argument("--ratio", type=int, default=1)
    # We select one pre-trained estimator
    parser.add_argument("--estimator_lr", type=float, default=0.003)
    parser.add_argument("--estimator_lr_decay", type=float, default=0.8)

    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--n_steps_per_epoch", type=int, default=500)
    parser.add_argument("--n_episodes", type=int, default=1)
    parser.add_argument("--algo", type=str, default="iw") # "iw" or "mb"
    parser.add_argument("--method", type=str, default="new") # "new" or "baseline"
    parser.add_argument('--upload', dest='upload', action='store_true', help='Enable upload')
    parser.add_argument('--no-upload', dest='upload', action='store_false', help='Disable upload')
    parser.add_argument('--collect', dest='collect', action='store_true', help='Enable collect')
    parser.add_argument('--no-collect', dest='collect', action='store_false', help='Disable collect')
    parser.set_defaults(upload=False)
    parser.set_defaults(collect=True)


    args = parser.parse_args()
    # pdb.set_trace()
    if "Pendulum" in args.dataset:
        env = gym.make("Pendulum-v1")
        d3rlpy.seed(args.seed)
        d3rlpy.envs.seed_env(env, args.seed)
        d4rl_dataset = get_pendulum(dataset_type=args.dataset.split("-")[1])
    else:
        env = gym.make(args.dataset)
        d3rlpy.seed(args.seed)
        d3rlpy.envs.seed_env(env, args.seed)
        d4rl_dataset = env.get_dataset()

    behavior_dataset = D4rlDataset(
        d4rl_dataset,
        normalize_states=False,
        normalize_rewards=False,
        noise_scale=0.0,
        bootstrap=False)

    dataloader = DataLoader(behavior_dataset, batch_size=256, shuffle=True, drop_last=True, num_workers=4)
    inf_loader = infinite_loader(dataloader, behavior_dataset)

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
        buffer = ReplayBuffer_(capacity=2560000)
        model = Model(sac1=sac1,sac2=sac2)
        model.fit(
            inf_loader,
            d4rl_dataset,
            buffer,
            n_steps=args.n_epoch * args.n_steps_per_epoch,
            n_steps_per_epoch=args.n_steps_per_epoch,
            save_interval=10,
            evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env,gamma=0.995,n_trials=args.n_episodes)},
            experiment_name=f"SAC_{args.dataset}_{args.seed}",
            dir_path="/home/xiaoan/checkpoints/new/{}/{}-{}-{}-{}-{}-{}/seed{}/{}-{}-{}".format(args.dataset,
                                                                    sac1._config.actor_learning_rate,
                                                                    sac1._config.critic_learning_rate, 
                                                                    sac2._config.actor_learning_rate,
                                                                    sac2._config.critic_learning_rate,
                                                                    args.decay_epoch, args.lr_decay,
                                                                    args.seed, args.algo, args.estimator_lr, args.estimator_lr_decay),
            seed = args.seed,
            env_name = args.dataset,
            decay_epoch = args.decay_epoch,
            lr_decay = args.lr_decay,
            estimator_lr = args.estimator_lr,
            estimator_lr_decay = args.estimator_lr_decay,
            algo = args.algo,
            ratio = args.ratio,
            upload = args.upload,
            collect = args.collect
        )
    elif args.method == "baseline":
        sac1.fit(
            inf_loader,
            d4rl_dataset,
            n_steps=args.n_epoch * args.n_steps_per_epoch,
            n_steps_per_epoch=args.n_steps_per_epoch,
            save_interval=10,
            evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env,gamma=0.995,n_trials=args.n_episodes)},
            experiment_name=f"SAC_{args.dataset}_{args.seed}",
            dir_path="/home/xiaoan/checkpoints/origin/{}/{}-{}-{}-{}/seed{}".format(args.dataset,
                                                                    sac1._config.actor_learning_rate,
                                                                    sac1._config.critic_learning_rate,
                                                                    args.decay_epoch,args.lr_decay, 
                                                                    args.seed),
            env_name = args.dataset,
            decay_epoch=args.decay_epoch,
            lr_decay=args.lr_decay,
            algo = args.algo,
            upload = args.upload,
            collect = args.collect
        )
    


if __name__ == "__main__":
    main()