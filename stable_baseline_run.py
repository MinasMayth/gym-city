import gym
import gym_city
import time
from datetime import datetime
from arguments import get_args
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import os
from stable_baselines3 import A2C, PPO
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env

class ImageRecorderCallback(BaseCallback):
    def __init__(self, model, verbose=1):
        super(ImageRecorderCallback, self).__init__(verbose)
        self.init_callback(model)

    def _on_step(self):
        image = self.training_env.render(mode="rgb_array")
        # "HWC" specify the dataformat of the image, here channel last
        # (H for height, W for width, C for channel)
        # See https://pytorch.org/docs/stable/tensorboard.html
        # for supported formats
        self.logger.record("images", Image(image, "HWC"), exclude=("stdout", "log", "json", "csv"))
        return True

def make_env():
    env = gym.make("MicropolisEnv-v0")
    env.setMapSize(16, render_gui=False)
    return env

def main():
    args = get_args()
    print(args)
    algorithm = args.algo
    # Get current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Define the path with the current date and time
    # save_path = os.path.join("trained_models", "baseline_models", algorithm, current_datetime)
    log_path = os.path.join("logs", "baselines", algorithm, current_datetime)
    save_path = log_path

    if args.log:
        new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])

    # env = make_vec_env("MicropolisEnv-v0", n_envs=4, seed=0, vec_env_cls=SubprocVecEnv)
    env = make_env()

    if args.log:
        if algorithm == "a2c":
            model = A2C("MlpPolicy", env, gamma=args.gamma, n_steps=args.num_steps,
                        vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                        learning_rate=args.lr, rms_prop_eps=args.eps, verbose=1, tensorboard_log=log_path, create_eval_env=True)
        elif algorithm == "ppo":
            model = PPO("MlpPolicy", env, gamma=args.gamma, n_steps=args.num_steps,
                        vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                        learning_rate=args.lr, verbose=1, tensorboard_log=log_path)
        elif algorithm == "dqn":
            model = PPO("MlpPolicy", env, gamma=args.gamma, n_steps=args.num_steps,
                        vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                        learning_rate=args.lr, verbose=1, tensorboard_log=log_path)
        else:
            exit()
    else:
        if algorithm == "a2c":
            model = A2C("MlpPolicy", env, gamma=args.gamma, n_steps=args.num_steps,
                        vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                        learning_rate=args.lr, rms_prop_eps=args.eps, verbose=1)
        elif algorithm == "ppo":
            model = PPO("MlpPolicy", env, gamma=args.gamma, n_steps=args.num_steps,
                        vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                        learning_rate=args.lr, verbose=1)
        elif algorithm == "dqn":
            model = PPO("MlpPolicy", env, gamma=args.gamma, n_steps=args.num_steps,
                        vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                        learning_rate=args.lr, verbose=1)
        else:
            exit()

    if args.log:
        checkpoint_callback = CheckpointCallback(save_freq=25000, save_path=save_path + "/models")
        # Separate evaluation env
        eval_callback = EvalCallback(model.get_env(), best_model_save_path=save_path + '/models/best_model',
                                     log_path=save_path + '/models/best_model', eval_freq=500)
        # Create the callback list
        callback = CallbackList([checkpoint_callback, eval_callback])
        # Save model parameters to a text file
        with open(os.path.join(log_path, "model_parameters.txt"), "w") as f:
            f.write(str(model.get_parameters()))
        model.set_logger(new_logger)
        model.learn(total_timesteps=1_000_000, callback=callback, eval_log_path=log_path, reset_num_timesteps=True)
    else:
        model.learn(total_timesteps=1_000_000)

    if args.log:
        model.save(save_path)
    env.close()


if __name__ == "__main__":
    main()
