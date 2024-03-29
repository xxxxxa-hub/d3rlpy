
import argparse
from gym import wrappers
import gym
import pdb
import d3rlpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=str, default="cuda:0")
    args = parser.parse_args()

    env = gym.make(args.env, render_mode="rgb_array")
    eval_env = gym.make(args.env)

    # env=wrappers.Monitor(env,'/tmp/train_env')
    # eval_env=wrappers.Monitor(eval_env,'/tmp/test_env')

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)
    d3rlpy.envs.seed_env(eval_env, args.seed)

    # setup algorithm
    sac = d3rlpy.algos.SACConfig(
        batch_size=512,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=0,
    ).create(device=args.gpu)

    # replay buffer for experience replay
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=50000, env=env)

    # start training
    sac.fit_online(
        env,
        buffer,
        eval_env=eval_env,
        n_steps=50000,
        n_steps_per_epoch=1000,
        update_interval=1,
        update_start_step=1000,
    )


if __name__ == "__main__":
    main()
