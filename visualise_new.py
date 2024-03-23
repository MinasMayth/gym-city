import gym
import gym_city

from stable_baselines3 import A2C
def make_env():
    env = gym.make("MicropolisEnv-v0")
    env.setMapSize(16, render_gui=True)
    return env

def main():
    model = A2C.load("logs/baselines/a2c/gamma=0.98_num_steps=9_value_loss_coef=0.5_entropy_coef=0.0_max_grad_norm=0.5_lr=0.0007_eps=1e-05_gae_lambda=0.98_2024-03-22_21-53-03/models/rl_model_1500000_steps.zip")
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