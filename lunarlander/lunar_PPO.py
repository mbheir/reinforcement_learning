import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Script params
USE_TRAINED_MODEL = True
MODEL_PATH = "ppo_lunar_v3"


# Initialize environment
# env = gym.make("LunarLander-v2",)# render_mode="human")
env = make_vec_env("LunarLander-v2",n_envs=8)

# Initialize and train PPO model
if not USE_TRAINED_MODEL:
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000_000)
    model.save(MODEL_PATH)
else:
    model = PPO.load(MODEL_PATH)


observation = env.reset()
while True:
    action, _states = model.predict(observation)
    observation, reward, dones, info = env.step(action)

    env.envs[0].render()
    # if dones:
    #     env.reset()
