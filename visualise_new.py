import gym
import gym_city

from stable_baselines3 import A2C, PPO
def make_env():
    env = gym.make("MicropolisEnv-v0")
    env.setMapSize(24, render_gui=True)
    return env
def main():
    model = PPO.load(
            "logs/baselines/june/new_reward_experiments/ppo/n_steps=2048_map_w=24_clip_range=0.3_batch_size=128_n_epochs=20_v_l_coef=0.5_e_coef=0.0_lr=0.001_eps=1e-05_gamma=0.95_max_grad_norm=0.5_lambda=0.95_seed=1_2024-06-01_11-58-02/models/rl_model_1000000_steps.zip"
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