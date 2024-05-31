import gym
import gym_city

from stable_baselines3 import A2C, PPO
def make_env():
    env = gym.make("MicropolisEnv-v0")
    env.setMapSize(24, render_gui=True)
    return env
def main():
    model = PPO.load(
            "logs/baselines/may/nuevo_grid_search/ppo/alpha=0.99_num_steps=256_map_width=24_clip_range=0.2_batch_size=128_n_epochs=10_value_loss_coef=0.5_entropy_coef=0.01_lr=0.001_eps=1e-05_gamma=0.95_max_grad_norm=0.5_lambda=0.95_seed=1_2024-05-31_17-24-24/models/rl_model_2000000_steps.zip"
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