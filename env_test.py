import torch
import torch.nn as nn
import gym

env = gym.make('Breakout-ramDeterministic-v4', render_mode="human")

EPISODE_NUM = 5


class Network(nn.Module):
    def __init__(self, in_features, out_features):
        super(Network, self).__init__()
        self.net = nn.Sequential(nn.Linear(in_features, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, out_features))

    def forward(self, x):
        return self.net(x)

    def act(self, state_in):
        state_in = torch.as_tensor(state_in, dtype=torch.float32, device=device)
        return torch.argmax(self(state_in.unsqueeze(0)), dim=1).item()


# getting and setting and displaying the used device
try:
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(torch.cuda.current_device())}")
except Exception as e:
    print(e)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_features = env.observation_space.shape[0]
out_features = env.action_space.n
target_net = Network(in_features, out_features).to(device)

model_dict = torch.load('model.pth')
target_net.load_state_dict(model_dict)

for episode in range(EPISODE_NUM):
    state = env.reset()[0] / 255
    done = False

    while not done:
        action = target_net.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = next_state / 255
        state = next_state
        env.render()
