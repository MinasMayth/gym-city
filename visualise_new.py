import gym
import gym_city

from stable_baselines3 import A2C
def make_env():
    env = gym.make("MicropolisEnv-v0")
    env.setMapSize(16, render_gui=True)
    return env

def main():
    model = A2C.load("logs/baselines/a2c/2024-03-21_00-04-14/models/rl_model_975000_steps.zip")
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