import gym
import gym_city

from stable_baselines3 import A2C, PPO
def make_env():
    env = gym.make("MicropolisEnv-v0")
    env.setMapSize(32, render_gui=True)
    return env

def main():
    model = A2C.load(
        "logs/baselines/a2c/CustomNetworkV1/logs/baselines/BaseToolSet/CustomNetworkV1/NewReward/a2c/alpha=0.99_num_steps=11_map_width=32_lr=0.0009_eps=1e-05_2024-04-04_23-41-27/models/rl_model_1000000_steps.zip"

    )
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