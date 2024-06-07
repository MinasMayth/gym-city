import gym
import gym_city

from stable_baselines3 import A2C, PPO
def make_env():
    env_name = "MicropolisEnv-v0"
    #env_name = "Alien-v0"
    env = gym.make(env_name)
    env.setMapSize(16, render_gui=True)
    return env
def main():
    model = A2C.load(
        ""
                   )
    env = make_env()
    obs = env.reset()
    for i in range(10000):
        action, _state = model.predict(obs, deterministic=True)
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