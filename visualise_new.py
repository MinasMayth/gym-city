import gym
import gym_city

from stable_baselines3 import A2C, PPO
def make_env():
    env = gym.make("MicropolisEnv-v0")
    env.setMapSize(16, render_gui=True)
    return env
def main():
    model = A2C.load(
    "logs/baselines/june/custom_network/V3/a2c/n_steps=20_map_w=16_gamma=0.95_v_l_coef=0.5_e_coef=0.0_max_grad_norm=0.5_lr=0.0001_seed=1_eps=1e-05_lambda=0.95_2024-06-05_15-55-57/models/rl_model_1000000_steps.zip"
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