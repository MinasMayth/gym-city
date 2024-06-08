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
        "logs/baselines/june/power_puzzle/new_pp/a2c/n_steps=5_map_w=16_gamma=0.98_v_l_coef=0.5_e_coef=0.05_max_grad_norm=0.5_lr=0.0005_seed=1_eps=1e-05_lambda=0.96_vec_envs=4_2024-06-08_10-42-11/models/rl_model_1250000_steps.zip"
                   )
    env = make_env()
    obs = env.reset()
    for i in range(10000):
        action, _state = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        print(reward)

        env.render("human")
        # VecEnv resets automatically
        if done:
            print("Episode finished after {} timesteps".format(i + 1))
            obs = env.reset()

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()