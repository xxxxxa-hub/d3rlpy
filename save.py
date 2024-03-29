from d3rlpy.utils import save_policy, run
import torch
import os
import pdb

def process_baseline1():
    dir_path = "/home/xiaoan/checkpoints/baseline1/Pendulum-random"
    x = os.listdir(dir_path)

    for hp in x:
        y = os.path.join(dir_path,hp,"seed0")

        # save model to pkl
        model = torch.load(os.path.join(y,"model_200.pt"))
        save_policy(model,y)

        # estimate
        run(device=0,
            env_name="Pendulum-random",
            lr=0.003,
            policy_path="{}/policy.pkl".format(y),
            lr_decay=0.8,
            seed=0,
            algo="iw")
        
        # on-policy
        # rollout 100 episodes and save in oracle.csv
