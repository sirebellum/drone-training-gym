import torch as th
from stable_baselines3 import PPO
import gym
import path_finder
from torch.nn.modules.activation import Sigmoid

class OnnxablePolicy(th.nn.Module):
    def __init__(self, extractor, mlp_extractor, action_net):
        super().__init__()
        self.extractor = extractor
        self.mlp_extractor = mlp_extractor
        self.action_net = action_net

    def forward(self, observation):
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this. See `common.preprocessing.preprocess_obs`
        action_hidden = self.extractor(observation)
        action_hidden = self.mlp_extractor(action_hidden)
        action_hidden = self.action_net(action_hidden)
        return Sigmoid()(action_hidden)

def run():

    env = gym.make("PathFinder-v0")
    model = PPO.load("fc", custom_objects={"action_space": env.action_space}, device="cpu")
    onnxable_model = OnnxablePolicy(
        model.policy.features_extractor,
        model.policy.mlp_extractor.shared_net,
        model.policy.action_net
    )
    observation_size = model.observation_space.shape
    th.onnx.export(onnxable_model, 
                   th.randn([1, *observation_size]),
                   "fc.onnx",
                   input_names=["input"])

if __name__ == "__main__":
    run()
