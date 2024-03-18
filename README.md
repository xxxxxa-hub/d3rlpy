# d3rlpy

d3rlpy can be installed by cloning the repository as follows:
```
git clone https://github.com/xxxxxa-hub/d3rlpy.git
cd d3rlpy
pip install -e .
```

# Offline Training
```
python train.py --dataset Pendulum-replay --seed 0 --gpu cuda:0 --actor_lr 0.001 --critic_lr 0.001 --decay_epoch 1 --lr_decay 1.0 --ratio 1 --estimator_lr 0.003 --estimator_lr_decay 0.8 --n_epoch 100 --n_steps_per_epoch 500 --n_episodes 1 --algo iw --method baseline --upload
```

`dataset`: The offline dataset we use for training.

`seed`: Random seed.

`gpu`: The device on which we run the experiment.

`actor_lr`: Initial learning rate of actor network.

`critic_lr`: Initial learning rate of critic network.

`decay_epoch`: The number of epochs between two consecutive learning rate decays.

`lr_decay`: Decay rate of learning rate of actor and critic network.

`ratio`: Ratio between the number of update steps of inner policy and that of outer policy.

`estimator_lr`: Initial learning rate of policy value estimator.

`estimator_lr_decay`: Decay rate of learning rate of estimator.

`n_epoch`: Number of training epochs.

`n_steps_per_epoch`: Number of update steps in each epoch.

`n_episodes`: Number of episodes to evaluate on-policy value.

`algo`: Algorithm of estimator training.

`method`: Paradigm of training. The default method is "baseline".

`upload`: Whether to upload the result to wandb for visualization.

`collect`: Whether to collect transitions during online evaluation.