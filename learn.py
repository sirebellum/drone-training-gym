"""Script demonstrating the use of `gym_pybullet_drones`' Gym interface.

Class HoverAviary is used as a learning env for the A2C and PPO algorithms.

Example
-------
In a terminal, run as:

    $ python learn.py

Notes
-----
The boolean argument --rllib switches between `stable-baselines3` and `ray[rllib]`.
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning libraries `stable-baselines3` and `ray[rllib]`.
It is not meant as a good/effective learning example.

"""
import time
import os
import argparse
import gym
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.td3 import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from torch.nn.modules.activation import Tanh
from torch.optim import RMSprop

import path_finder

def run():
    env = gym.make("PathFinder-v0",
               sim_freq=120,
               init_xyzs=np.array([0,0,0]))
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.array([0.5]*n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(MlpPolicy,
                env,
                action_noise=action_noise,
                learning_rate=1e-3,
                verbose=1,
                tensorboard_log="tensorboard/hover",
                policy_kwargs={"activation_fn": Tanh,
                               "net_arch": [256,256],
                               "optimizer_class": RMSprop},
                batch_size=128,
                )

    if os.path.exists("fc.zip"):
        model.set_parameters("fc")

    # Train with random environments
    try:
        sessions = 1000
        for i in range(sessions):
            init_x = (np.random.random()*2-1)/(sessions-i)
            init_y = (np.random.random()*2-1)/(sessions-i)
            init_z = np.random.random()*2-1
            init_yaw = np.random.random()*2-1
            env = gym.make("PathFinder-v0",
                       init_xyzs=np.array([init_x,init_y,0.5]),
                       final_xyzs=np.array([0,0,0.5]),
                       init_RPYs=np.array([0,0,0]),
                       final_yaw=np.array([0]),
                       sim_freq=120)
            model.set_env(env)

            model.learn(total_timesteps=10000)
    except KeyboardInterrupt:
        pass
    
    model.save("fc")

if __name__ == "__main__":
    run()
