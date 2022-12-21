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
from math import pi

from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from torch.nn.modules.activation import Tanh

import path_finder

def run():
    env = gym.make("PathFinder-v0",
               sim_freq=200,
               init_xyzs=[0,0,0])
    
    model = PPO(MlpPolicy,
                env,
                learning_rate=1e-4,
                verbose=1,
                tensorboard_log="tensorboard/hover",
                policy_kwargs={"activation_fn": Tanh,
                               "net_arch": [256,256]},
                batch_size=2048,
                )

    if os.path.exists("fc.zip"):
        model.set_parameters("fc")

    # Train with random environments
    try:
        sessions = 10000
        for i in range(sessions):
            init_x = 0
            init_y = 0
            init_z = 0
            init_yaw = 0
            env = gym.make("PathFinder-v0",
                       init_xyzs=[init_x,init_y,init_z],
                       final_xyzs=[0,0,0],
                       init_RPYs=[0,0,init_yaw],
                       final_yaw=0,
                       sim_freq=200)
            env._max_episode_steps = 500
            model.set_env(env)

            model.learn(total_timesteps=100000)
            env.close()
            model.save("./model_archive/fc"+str(i+1))
    except KeyboardInterrupt:
        pass
    
    model.save("fc")

if __name__ == "__main__":
    run()
