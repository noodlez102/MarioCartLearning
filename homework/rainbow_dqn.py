import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from planner import Planner
from controller import control
from utils import PyTux
import pystk

class DQNAgent(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DQNAgent, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.network(x)


class RainbowAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network = DQNAgent(state_dim, action_dim).to(self.device)
        self.target_network = DQNAgent(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.action_dim = action_dim

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


def train_rl(args):
    agent = RainbowAgent(state_dim=10, action_dim=3)  # Example dimensions
    pytux = PyTux()
    num_episodes = args.num_episodes

    for episode in range(num_episodes):
        track = random.choice(args.track)
        steps, _ = pytux.rollout_rl(track, control, max_frames=1000, verbose=args.verbose)

        # Collect and preprocess data (states, actions, rewards, etc.)
        state = np.random.randn(10)  # Replace with actual state
        total_reward = 0

        for step in range(steps):
            action = agent.act(state)
            # Simulate environment step
            next_state = np.random.randn(10)  # Replace with actual next state
            reward = np.random.rand()        # Replace with actual reward
            done = step == steps - 1         # Set to True when episode ends
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Train the agent continuously after every step
            agent.replay(batch_size=32)
            
            if done:
                break

        # Update the target network periodically (every episode or every few steps)
        if episode % 10 == 0:  # Update the target network every 10 episodes (for example)
            agent.update_target_network()
        
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward}")

    pytux.close()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-e', '--num_episodes', type=int, default=100)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    train_rl(args)
