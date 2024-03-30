import gym
import gym_city

from stable_baselines3 import A2C, PPO
def make_env():
    env = gym.make("MicropolisEnv-v0")
    env.setMapSize(64, render_gui=True)
    return env

def main():
    model = A2C.load("logs/baselines/a2c/CustomNetworkV1/logs/baselines/BaseToolSet/CustomNetworkV1/NewReward/a2c/gamma=0.98_num_steps=9_lr=0.001_eps=1e-05_2024-03-29_13-08-06/models/rl_model_6000000_steps.zip")
    env = make_env()
    obs = env.reset()
    for i in range(10000):
        action, _state = model.predict(obs)
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