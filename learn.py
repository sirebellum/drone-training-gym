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
from torch.nn import Identity

import path_finder

def run():
    env = gym.make("PathFinder-v0",
               sim_freq=200,
               init_xyzs=[0,0,0])
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.array([0.0]*n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(MlpPolicy,
                env,
                action_noise=action_noise,
                learning_rate=1e-4,
                verbose=1,
                tensorboard_log="tensorboard/hover",
                batch_size=2048,
                )

    if os.path.exists("fc.zip"):
        model.set_parameters("fc")

    # Train with random environments
    try:
        sessions = 10000
        for i in range(sessions):
            init_x = np.random.random()*2-1
            init_y = np.random.random()*2-1
            init_z = np.random.random()*2-1
            init_yaw = np.random.random()*2-1
            env = gym.make("PathFinder-v0",
                       init_xyzs=[init_x,init_y,init_z],
                       final_xyzs=[0,0,0],
                       init_RPYs=[0,0,0],
                       final_yaw=0,
                       sim_freq=200)
            env._max_episode_steps = 100
            model.set_env(env)

            model.learn(total_timesteps=10000)
            env.close()
            model.save("./model_archive/fc"+str(i+1))
    except KeyboardInterrupt:
        pass
    
    model.save("fc")

if __name__ == "__main__":
    run()
