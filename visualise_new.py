import gym
import gym_city

from stable_baselines3 import A2C, PPO
import numpy as np
def make_env():
    env_name = "MicropolisEnv-v0"
    #env_name = "Alien-v0"
    env = gym.make(env_name)
    env.setMapSize(16, render_gui=True)
    return env
def main():
    model = A2C.load(
        "logs/baselines/a2c/env_name=MicropolisEnv-v0_alpha=0.99_num_steps=50_map_width=16_gamma=0.95_value_loss_coef=0.5_entropy_coef=0.01_max_grad_norm=0.5_lr=0.0001_seed=1_2024-06-07_11-58-50/models/best_model/best_model.zip"
                   )
    env = make_env()
    obs = env.reset()
    for i in range(10000):
        action, _state = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)

        env.render("human")
        # VecEnv resets automatically
        if done:
            print("Episode finished after {} timesteps".format(i + 1))
            obs = env.reset()

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()