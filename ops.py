import pandas as pd
import numpy as np
import os
import pdb
import torch
import d3rlpy
import gym
from d3rlpy.utils import run
from save import process_baseline1
# def ops(oracle_list, estimate_list):
#     indices = sorted(range(len(estimate_list)), key=lambda i: estimate_list[i], reverse=True)[:1]

#     # 使用这些索引在oracle_list中找到对应的元素
#     selected_oracle_values = [oracle_list[i] for i in indices]

#     # 计算这些元素在oracle_list中的平均值
#     average_of_selected = sum(selected_oracle_values) / len(selected_oracle_values)

#     return indices, selected_oracle_values, average_of_selected


env = gym.make("Pendulum-v1")
d3rlpy.seed(0)
d3rlpy.envs.seed_env(env, 0)


dir_path = "/home/xiaoan/checkpoints/baseline2/Pendulum-random"
x = os.listdir(dir_path)

if "baseline1" in dir_path:
    process_baseline1()

estimate_list = []

for _ in x:
    y = os.path.join(dir_path,_,"seed0") # ,"iw-0.003-0.8"
    estiamte = pd.read_csv(os.path.join(y,"ope.csv")).iloc[0,0]
    estimate_list.append(estiamte)

print(estimate_list)
print(len(estimate_list))
index = sorted(range(len(estimate_list)), key=lambda i: estimate_list[i], reverse=True)[0]
print(index)
print(max(estimate_list))
print(x[index])

best_hp_model = os.path.join(dir_path,x[index],"seed0","model_200.pt")
model = torch.load(best_hp_model)
evaluator = d3rlpy.metrics.EnvironmentEvaluator(env,gamma=0.995,n_trials=1000)
test_score_1_mean, test_score_1_std, test_score_mean, test_score_std, _ = evaluator(model)

print("Return mean when gamma = 1.0:", test_score_1_mean)
print("Return std when gamma = 1.0:", test_score_1_std)
print("Return mean when gamma = 0.995:", test_score_mean)
print("Return std when gamma = 0.995:", test_score_std)