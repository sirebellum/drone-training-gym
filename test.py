import time
import numpy as np

import path_finder
import gym

import onnx
import onnxruntime as ort

def main():

    env = gym.make("PathFinder-v0",
               sim_freq=200,
               init_xyzs=np.array([0,0.0,0.0]),
               gui=True,)
    env._max_episode_steps = 100
    onnx_path = "fc.onnx"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    model = ort.InferenceSession(onnx_path)

    obs = env.reset()
    start = time.time()
    done = False
    i = 0
    while not done:
        action = model.run(None, {"input": np.expand_dims(obs.astype("float32"), 0)})[0][0]
        obs, reward, done, info = env.step(action)
        input(f"{action}, {reward}\r")
        env.render()
        i += 1
    env.close()

if __name__ == "__main__":
    main()