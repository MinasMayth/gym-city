import gym
import gym_city

from stable_baselines3 import A2C, PPO
def make_env():
    env = gym.make("MicropolisEnv-v0")
    env.setMapSize(16, render_gui=True)
    return env
def main():
    model = A2C.load(
        "logs/baselines/a2c/alpha=0.99_num_steps=5_map_width=16_gamma=0.99_value_loss_coef=0.5_entropy_coef=0.01_max_grad_norm=0.5_lr=0.0001_seed=1_2024-06-05_21-15-37/models/best_model/best_model.zip"
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