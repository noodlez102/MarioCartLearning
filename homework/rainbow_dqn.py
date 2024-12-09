import gym
from gym import spaces
import numpy as np
from utils import PyTux  # Adjust the import based on the actual file structure
import pystk  # Ensure you import pystk

class SuperTuxKartEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.pytux = PyTux()
        self.config = pystk.RaceConfig()
        
        # Define discrete action space: [steer left, steer right, accelerate, brake, drift]
        self.action_space = spaces.Discrete(5)
        # Observation space is an image (96 x 128 x 3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(96, 128, 3), dtype=np.uint8)
        
        # Initialize race configuration
        self.config.mode = pystk.RaceConfig.RaceMode.NORMAL_RACE
        self.config.num_kart = 1  # Example: 1 player, AI players will be filled automatically
        self.config.laps = 3  # Example: Set the number of laps
        
        # Ensure that we can modify the seed dynamically later
        self.config.seed = None  # Seed will be set during `seed()` method

    def reset(self):
        if self.config.seed is not None:
            self.pytux.reset(seed=self.config.seed)  # Pass the seed to reset if set
        else:
            self.pytux.reset()  # Otherwise, just reset normally

        state = np.array(self.pytux.render_data[0].image)
        return state

    def step(self, action):
        # Map discrete action to PyTux action
        action_map = [
            pystk.Action(steer=-1),  # Steer left
            pystk.Action(steer=1),   # Steer right
            pystk.Action(acceleration=1),  # Accelerate
            pystk.Action(brake=True),      # Brake
            pystk.Action(drift=True)       # Drift
        ]
        state, reward, done, info = self.pytux.step(action_map[action])
        return state, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.pytux.close()

    def seed(self, seed=None):
        """
        Set the seed for the environment to ensure reproducibility.
        Uses the seed property in RaceConfig if available.
        """
        self.config.seed = seed  # Set the seed in the RaceConfig
        self.pytux.reset()  # Re-initialize the environment with the new seed
        return [seed]  # Return the seed value to comply with Gym's seed method signature

from sb3_contrib import QRDQN  # Quantile Regression DQN (Rainbow-like)
from stable_baselines3.common.env_util import make_vec_env

# Create vectorized environment for parallel training
env = make_vec_env(SuperTuxKartEnv)

# Initialize Rainbow DQN (QR-DQN is a distributional DQN variant)
model = QRDQN("CnnPolicy", env, verbose=1, buffer_size=100000, learning_rate=1e-4, target_update_interval=1000)

# Train the model
model.learn(total_timesteps=200000)

# Save the model
model.save("rainbow_dqn_supertuxkart")

print("finished saving model")
# Load the model
model = QRDQN.load("rainbow_dqn_supertuxkart")

# Test the model
env = SuperTuxKartEnv()
state = env.reset()

done = False
while not done:
    action, _states = model.predict(state, deterministic=True)
    state, reward, done, info = env.step(action)
    env.render()
env.close()
