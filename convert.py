import torch as th
from torch.distributions import Normal
from stable_baselines3 import PPO

class OnnxablePolicy(th.nn.Module):
    def __init__(self, extractor, mlp_extractor, action_net, log_std):
        super().__init__()
        self.extractor = extractor
        self.mlp_extractor = mlp_extractor
        self.action_net = action_net
        self.log_std = log_std

    def forward(self, observation):
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this. See `common.preprocessing.preprocess_obs`
        observation = self.extractor(observation)
        action_hidden = self.mlp_extractor(observation)
        action = self.action_net(action_hidden)
        action_std = th.ones_like(action) * self.log_std.exp()

        low = 0
        high = 1
        raw_action = Normal(action, action_std).mean
        scaled_action = low + (0.5 * (raw_action + 1.0) * (high - low))
        return scaled_action

def run():

    model = PPO.load("fc", device="cpu")
    onnxable_model = OnnxablePolicy(
        model.policy.features_extractor,
        model.policy.mlp_extractor.shared_net,
        model.policy.action_net,
        model.policy.log_std,
    )
    observation_size = model.observation_space.shape
    th.onnx.export(onnxable_model, 
                   th.randn([1, *observation_size]),
                   "fc.onnx",
                   input_names=["input"])

if __name__ == "__main__":
    run()
