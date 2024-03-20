import gym
import gym_city
import time
from datetime import datetime
from arguments import get_args
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import os
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env


def make_env():
    env = gym.make("MicropolisEnv-v0")
    env.setMapSize(16, render_gui=True)
    return env

def main():
    args = get_args()
    print(args)
    algorithm = "A2C"
    # Get current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Define the path with the current date and time
    # save_path = os.path.join("trained_models", "baseline_models", algorithm, current_datetime)
    log_path = os.path.join("logs", "baselines", algorithm, current_datetime)
    save_path = log_path

    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])

    # env = make_vec_env("MicropolisEnv-v0", n_envs=4, seed=0, vec_env_cls=SubprocVecEnv)
    env = make_env()

    if algorithm == "A2C":
        model = A2C("MlpPolicy", env, gamma=args.gamma, n_steps=args.num_steps,
                    vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                    learning_rate=args.lr, verbose=1, tensorboard_log=log_path)
    elif algorithm == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
    else:
        exit()

    # Save model parameters to a text file
    with open(os.path.join(log_path, "model_parameters.txt"), "w") as f:
        f.write(str(model.get_parameters()))
    model.set_logger(new_logger)
    model.learn(total_timesteps=1_000_000)
    model.save(save_path)
    #
    # vec_env = model.get_env()
    # obs = vec_env.reset()
    #
    # for i in range(1000):
    #     action, _state = model.predict(obs)
    #     obs, reward, done, info = vec_env.step(action)
    #     vec_env.render("human")
    #     # VecEnv resets automatically
    #     if done:
    #         print("Episode finished after {} timesteps".format(i + 1))
    #         obs = vec_env.reset()

    # Close the environment
    env.close()


if __name__ == "__main__":
    main()
