import subprocess

def run(device,env_name,lr,policy_path,lr_decay,seed,algo):
    commands = ["CUDA_VISIBLE_DEVICES={} /environment/miniconda3/envs/ope/bin/python /home/featurize/policy_eval/eval.py --logtostderr --d4rl \
                --env_name={} --d4rl_policy_filename={} \
                --target_policy_std=0.0 --seed={} --nobootstrap --algo={} \
                --noise_scale=0.0 --lr={} --lr_decay={}".format(device,env_name,policy_path,seed,algo,lr,lr_decay)]

    for command in commands:
        subprocess.run(command, shell=True)

# if __name__ == "__main__":
#     device = 1
#     env_name = "halfcheetah-random-v0"
#     lr = 0.01
#     lr_decay = 0.86
#     seed = 0
#     algo = "fqe"
#     run(device=device,env_name=env_name,lr=lr,lr_decay=lr_decay,seed=seed,algo=algo)