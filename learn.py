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
from torch.nn.modules.activation import ReLU

import path_finder

def run():
    env = gym.make("PathFinder-v0",
               sim_freq=120,
               init_xyzs=[0,0,0])

    model = PPO(MlpPolicy,
                env,
                learning_rate=1e-5,
                verbose=1,
                tensorboard_log="tensorboard/hover",
                policy_kwargs={"activation_fn": ReLU,
                               "net_arch": [64,64],
                               "squash_output": True},
                batch_size=2048,
                device="cuda"
                )

    if os.path.exists("fc.zip"):
        model.set_parameters("fc")

    # Train with random environments
    try:
        sessions = 1000
        for i in range(sessions):
            init_x = 0
            init_y = 0
            init_z = 0
            init_roll = (np.random.random()*2-1)*pi/2/(sessions-i)
            init_pitch = (np.random.random()*2-1)*pi/2/(sessions-i)
            init_yaw = (np.random.random()*2-1)*pi/2/(sessions-i)
            env = gym.make("PathFinder-v0",
                       init_xyzs=[init_x,init_y,init_z],
                       final_xyzs=[init_x,init_y,init_z],
                       init_RPYs=[init_roll,init_pitch,init_yaw],
                       final_yaw=init_yaw,
                       sim_freq=120)
            env._max_episode_steps = 200
            model.set_env(env)

            model.learn(total_timesteps=100000)
            model.save("./model_archive/fc"+str(i+1))
            env.close()
    except KeyboardInterrupt:
        pass
    model.save("fc")
    
if __name__ == "__main__":
    run()
