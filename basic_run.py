import gym
import gym_city
import time
from datetime import datetime
from arguments import get_args
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import os
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env

def make_env():
    env = gym.make("MicropolisEnv-v0")
    env.setMapSize(16, render_gui=True)
    print(check_env(env))
    return env

def main():
    env = make_env()
    obs = env.reset()
    n_steps = 10000
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
