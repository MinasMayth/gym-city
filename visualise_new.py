import gym
import gym_city

from stable_baselines3 import A2C, PPO
def make_env():
    env = gym.make("MicropolisEnv-v0")
    env.setMapSize(24, render_gui=True)
    return env
def main():
    model = A2C.load(
    "logs/baselines/may/nuevo_grid_search/a2c/alpha=0.99_num_steps=5_map_width=24_gamma=0.9_value_loss_coef=0.5_entropy_coef=0.01_max_grad_norm=0.5_lr=1e-05_2024-05-29_19-16-35/models/rl_model_4500000_steps.zip"
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