import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from utils import PyTux
import pystk

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10  # Update target network every 10 episodes
MAX_FRAMES = 1000
MAX_EPISODES = 500
TRACK_NAME = "zengarden"

ACTION_TO_AIM_POINT = {
    0: [-0.5, 0.0],  # Steer left
    1: [0.0, 0.0],   # Go straight
    2: [0.5, 0.0],   # Steer right
}

# Neural Network for Q-Learning
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# RL Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPSILON_START
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

        # Copy weights from Q-Network to Target Network
        self.target_network.load_state_dict(self.q_network.state_dict())


    def act(self, state):
        """Select an action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.q_network(state_tensor).detach().numpy()
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Compute target Q-values
        with torch.no_grad():
            target_q = rewards + GAMMA * (1 - dones) * self.target_network(next_states).max(dim=1, keepdim=True)[0]

        # Compute current Q-values
        current_q = self.q_network(states).gather(1, actions)

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        
    def controller(self, action_index):
        """Convert a discrete action index to a pystk.Action."""
        action = pystk.Action()

        # Define your action mapping here (e.g., steering, acceleration)
        if action_index == 0:  # Example: turn left
            action.steer = -1
            action.acceleration = 1
        elif action_index == 1:  # Example: go straight
            action.steer = 0
            action.acceleration = 1
        elif action_index == 2:  # Example: turn right
            action.steer = 1
            action.acceleration = 1

        return action

# Reward Function
def reward_function(state, track):
    kart = state.players[0].kart
    return kart.overall_distance / track.length


# Main Training Loop
if __name__ == "__main__":
    pytux = PyTux()
    state_dim = 3  # Example: distance to center, velocity, aim direction
    action_dim = 3  # Example: left, straight, right
    agent = DQNAgent(state_dim, action_dim)

    def control_rl(state, current_vel):
        # Extract necessary values from the state
        kart = state.players[0].kart
        aim_point = state.players[0].waypoint.position  # Assuming waypoint gives an aim point

        # Create an action
        action = pystk.Action()

        # Compute control logic (example)
        action.steer = aim_point[0]
        action.acceleration = 1.0 if current_vel < 20 else 0.0
        action.drift = abs(aim_point[0]) > 0.2

        return action

    for episode in range(MAX_EPISODES):
        print(f"Episode {episode + 1}/{MAX_EPISODES}")

        # Use rollout to run the simulation
        steps, total_reward = pytux.rollout(TRACK_NAME, control_rl, max_frames=MAX_FRAMES, verbose=True)

        print(f"Episode {episode + 1}: Steps = {steps}, Total Reward = {total_reward}")

        # Update agent based on experiences
        agent.replay()

        # Update target network periodically
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        # Decay exploration rate
        agent.decay_epsilon()

        print(f"Epsilon after episode {episode + 1}: {agent.epsilon:.2f}")

    pytux.close()
