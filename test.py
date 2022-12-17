import time
import numpy as np

import path_finder
import gym

import onnx
import onnxruntime as ort

def sync(i, start_time, timestep):
    """Syncs the stepped simulation with the wall-clock.

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    i : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    timestep : float
        Desired, wall-clock step of the simulation's rendering.

    """
    if timestep > .04 or i%(int(1/(24*timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i*timestep):
            time.sleep(timestep*i - elapsed)

def main():

    env = gym.make("PathFinder-v0",
               sim_freq=120,
               init_xyzs=np.array([0,0,0.0]),
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
        sync(i, start, env.TIMESTEP)
        i += 1
    env.close()

if __name__ == "__main__":
    main()