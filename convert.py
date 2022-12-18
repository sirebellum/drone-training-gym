import torch as th
from stable_baselines3 import TD3
import gym
import path_finder

class OnnxablePolicy(th.nn.Module):
    def __init__(self, extractor, action_net):
        super().__init__()
        self.extractor = extractor
        self.action_net = action_net

    def forward(self, observation):
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this. See `common.preprocessing.preprocess_obs`
        action_hidden = self.extractor(observation)
        return self.action_net(action_hidden)

def run():

    env = gym.make("PathFinder-v0")
    model = TD3.load("fc", custom_objects={"action_space": env.action_space}, device="cpu")
    onnxable_model = OnnxablePolicy(
        model.actor.features_extractor, model.actor.mu
    )
    observation_size = model.observation_space.shape
    th.onnx.export(onnxable_model, 
                   th.randn([1, *observation_size]),
                   "fc.onnx",
                   input_names=["input"])

if __name__ == "__main__":
    run()
