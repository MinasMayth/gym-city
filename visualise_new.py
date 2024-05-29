import gym
import gym_city

from stable_baselines3 import A2C, PPO
def make_env():
    env = gym.make("MicropolisEnv-v0")
    env.setMapSize(24, render_gui=True)
    return env
def main():
    model = PPO.load(
    "logs/baselines/may/improvements/improvements/ppo/alpha=0.99_num_steps=9_map_width=24_clip_range=0.2_batch_size=32_n_epochs=4_lr=0.0001_eps=1e-05_2024-05-26_23-58-39/models/rl_model_3500000_steps.zip"
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