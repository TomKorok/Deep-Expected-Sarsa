import torch
import torch.nn as nn
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import time


env = gym.make('Breakout-ramDeterministic-v4')


# getting and setting and displaying the used device
try:
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(torch.cuda.current_device())}")
except Exception as e:
    print(e)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        q_values = self(state_in.unsqueeze(0))
        max_q = torch.argmax((torch.softmax(q_values, dim=1) * q_values), dim=1)[0]
        return max_q.item()


BATCH_SIZE = 128
DISCOUNT_FACTOR = 0.99
replay_memory = deque(maxlen=10000)
TAU = 0.005
LR = 1e-4
MAX_STEPS = 5000

if torch.cuda.is_available():
    EPISODE_NUM = 1000
else:
    EPISODE_NUM = 50


in_features = env.observation_space.shape[0]
out_features = env.action_space.n
online_net = Network(in_features, out_features).to(device)
target_net = Network(in_features, out_features).to(device)
target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.AdamW(online_net.parameters(), lr=LR, amsgrad=True)

episode_duration = []


# prefill replay memory
def fill_replay_memory():
    state_frm = env.reset()[0]/255
    for _ in range(1000):
        action_frm = env.action_space.sample()
        next_state_frm, reward_frm, done_frm, _, _ = env.step(action_frm)
        next_state_frm = next_state_frm/255
        replay_memory.append((state_frm, action_frm, reward_frm, done_frm, next_state_frm))
        state_frm = next_state_frm

        if done_frm:
            state_frm = env.reset()[0]/255


# training
duration = np.zeros((EPISODE_NUM, 1))
cumulative_reward = []
fill_replay_memory()

start_time = time.time()

# episodes loop
for eps in range(EPISODE_NUM):
    if eps % 50 == 0:
        print(f"Episode {eps}")

    state = env.reset()[0]/255
    done = False
    step = 0
    eps_reward = 0
    # timestep loop
    while not done:
        action = online_net.act(state)

        next_state, reward, done, _, _ = env.step(action)
        eps_reward += reward
        next_state = next_state/255
        replay_memory.append((state, action, reward, done, next_state))
        state = next_state

        # sample and load a minibatch
        experiences = random.sample(replay_memory, k=BATCH_SIZE)
        states = np.asarray([e[0] for e in experiences])
        actions = np.asarray([e[1] for e in experiences])
        rewards = np.asarray([e[2] for e in experiences])
        dones = np.asarray([e[3] for e in experiences])
        new_states = np.asarray([e[4] for e in experiences])

        # convert them to tensors
        states = torch.as_tensor(states, dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(-1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1)
        new_states = torch.as_tensor(new_states, dtype=torch.float32, device=device)

        q_values = online_net(states)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions)

        target_q_values = target_net(new_states)
        action_prob_t = torch.softmax(target_q_values, dim=1)
        target_q_with_prob = torch.reshape(torch.sum(action_prob_t * target_q_values, dim=1), (BATCH_SIZE, 1))

        optimal_q_values = rewards + DISCOUNT_FACTOR * (1-dones) * target_q_with_prob
        criterion = nn.SmoothL1Loss()
        loss = criterion(action_q_values, optimal_q_values.detach())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(online_net.parameters(), 100)
        optimizer.step()

        if step >= MAX_STEPS:
            done = True

        if done:
            duration[eps] = step
        step += 1
        target_net_state_dict = target_net.state_dict()
        online_net_state_dict = online_net.state_dict()
        for key in online_net_state_dict:
            target_net_state_dict[key] = online_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

    cumulative_reward.append(eps_reward)

# Plotting and saving the curve
print(f"Run time: {time.time() - start_time:} seconds")
learning_curve = np.convolve(list(np.squeeze(duration, axis=1)), np.ones((1,))/1, mode='valid')

plt.title(f"Deep Expected Sarsa, training time: {round(time.time()-start_time, 2)} seconds")
plt.ylabel("Duration of an episode")
plt.xlabel("Training episodes")
plt.plot([j for j in range(len(learning_curve))], learning_curve, color='darkorange', alpha=1)
plt.savefig('learning_curve_for_breakout_desa.png')
plt.show()


learning_curve = np.convolve(cumulative_reward, np.ones((1,))/1, mode='valid')

plt.title(f"Collected rewards")
plt.ylabel("Reward per episode")
plt.xlabel("Training episodes")
plt.plot([j for j in range(len(learning_curve))], learning_curve, color='darkorange', alpha=1)
plt.savefig('collected_reward_for_breakout_desa.png')
plt.show()

# save target model
torch.save(target_net.state_dict(), 'model.pth')