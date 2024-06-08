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
    env = make_env()
    model = A2C.load(
        "logs/baselines/june/power_puzzle/new_pp/a2c/n_steps=20_map_w=16_gamma=0.96_v_l_coef=0.5_e_coef=0.01_max_grad_norm=0.5_lr=0.0001_seed=1_eps=1e-05_lambda=0.98_vec_envs=64_2024-06-08_17-44-43/models/rl_model_749952_steps.zip"
        ,env=env)

    env = model.get_env()
    obs = env.reset()
    for i in range(10000):
        action, _state = model.predict(obs, deterministic=True)
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