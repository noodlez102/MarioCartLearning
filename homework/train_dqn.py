import torch
import numpy as np
from rainbow_dqn import DQNAgent  # Your DQN implementation
import pystk

def train_dqn(args):
    # Initialize the pystk environment
    pystk.init()
    config = pystk.RaceConfig(num_players=1)
    config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
    race = pystk.Race(config)
    race.start()

    # Initialize the agent
    state_shape = (3, 96, 128)  # Example image dimensions
    num_actions = 3  # Example: throttle, steer left, steer right
    agent = DQNAgent(state_shape, num_actions, lr=args.learning_rate)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    agent.model.to(device)

    # Training loop
    for episode in range(args.num_episodes):
        race.restart()
        state = preprocess_state(race.render_data[0].image)  # Preprocess image from pystk
        total_reward = 0
        done = False

        while not done:
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            # Select action using epsilon-greedy policy
            action = agent.select_action(state_tensor)

            # Apply action in the environment
            perform_action(race, action)  # Perform action based on action index

            # Observe next state and reward
            next_state = preprocess_state(race.render_data[0].image)
            reward = calculate_reward(race)  # Define your reward function
            done = race.finish_time > 0  # Check if race is finished

            # Store the transition in replay buffer
            agent.replay_buffer.add((state, action, reward, next_state, done))

            # Train the agent
            agent.train(args.batch_size)

            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}/{args.num_episodes}, Total Reward: {total_reward}")

        # Update the target network periodically
        if episode % args.target_update_freq == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

    # Save the trained model
    torch.save(agent.model.state_dict(), 'dqn_model.pth')
    race.stop()
    pystk.clean()

def preprocess_state(image):
    # Convert image to normalized (0, 1) tensor and resize to state shape
    image = image / 255.0  # Normalize pixel values
    return np.transpose(image, (2, 0, 1))  # Convert to (C, H, W) format

def perform_action(race, action):
    # Map the action index to actual controls in pystk
    steer, throttle = 0, 1  # Default controls
    if action == 0:  # Steer left
        steer = -1
    elif action == 1:  # Steer right
        steer = 1
    elif action == 2:  # Throttle forward
        throttle = 1

    # Apply control
    race.step([pystk.Action(steer=steer, throttle=throttle)])

def calculate_reward(race):
    # Example: reward based on forward velocity
    return race.players[0].state.velocity[0]  # x-direction velocity

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num_episodes', type=int, default=500)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('--target_update_freq', type=int, default=10)
    args = parser.parse_args()

    train_dqn(args)
