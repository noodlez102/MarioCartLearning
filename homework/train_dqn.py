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
