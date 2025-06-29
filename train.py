import gymnasium as gym
import ale_py


def policy(obs):
    return env.action_space.sample()


gym.register_envs(ale_py)  # unnecessary but helpful for IDEs

env = gym.make('ALE/Breakout-v5', render_mode="human")  # remove render_mode in training
obs, info = env.reset()
episode_over = False

while not episode_over:
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated
env.close()
