import torch as th
from stable_baselines3 import TD3

class OnnxablePolicy(th.nn.Module):
    def __init__(self, action_net):
        super().__init__()
        self.action_net = action_net

    def forward(self, observation):
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this. See `common.preprocessing.preprocess_obs`
        scaled_action = self.action_net(observation)

        low = 0
        high = 1
        act = low + (0.5 * (scaled_action + 1.0) * (high - low))

        return act

def run():

    model = TD3.load("fc", device="cpu")
    print(model.actor)
    onnxable_model = OnnxablePolicy(
        model.actor
    )
    observation_size = model.observation_space.shape
    th.onnx.export(onnxable_model, 
                   th.randn([1, *observation_size]),
                   "fc.onnx",
                   input_names=["input"])

if __name__ == "__main__":
    run()
