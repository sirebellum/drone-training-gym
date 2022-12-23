import time
import numpy as np

import path_finder
import gym

import onnx
import onnxruntime as ort
from stable_baselines3 import TD3

def main():

    env = gym.make("PathFinder-v0",
               sim_freq=120,
               init_xyzs=np.array([0,0.0,0.0]),
               init_RPYs=[0.2,0.2,0.00],
               gui=True,)
    env._max_episode_steps = 200

    onnx_path = "fc.onnx"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    model = ort.InferenceSession(onnx_path)

    obs = env.reset()
    start = time.time()
    done = False
    tot_reward = 0
    i = 0
    while not done:
        action = model.run(None, {"input": np.expand_dims(obs.astype("float32"), 0)})[0]
        action = (action+1)/2
        input(f"{i} {action}, {tot_reward}\r")
        obs, reward, done, info = env.step(action)
        tot_reward += reward
        input(f"{i} {action}, {tot_reward}\r")
        env.render()
        i += 1
    env.close()

if __name__ == "__main__":
    main()