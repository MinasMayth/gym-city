import gym
import gym_city
import time
from datetime import datetime
from arguments import get_args
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import os
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env


def make_env(vec, args):
    if vec:
        env = make_vec_env(args.env_name, n_envs=4, vec_env_cls=DummyVecEnv)
        env.env_method("setMapSize", 16)
    else:
        env = gym.make(args.env_name)
        env.setMapSize(16, render_gui=args.visualise_training)
    return env


def main():
    args = get_args()
    print(args)
    algorithm = args.algo

    if args.log:
        verbose = 1
    else:
        verbose = 0
    # Get current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Define the path with the current date and time
    # save_path = os.path.join("trained_models", "baseline_models", algorithm, current_datetime)
    log_path = os.path.join("logs", "baselines", algorithm, current_datetime)
    save_path = log_path


    if args.save:
        os.makedirs(log_path, exist_ok=True)
        new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])

    env = make_env(vec=False, args=args)

    if args.save:
        if algorithm == "a2c":
            model = A2C("MlpPolicy", env, gamma=args.gamma, n_steps=args.num_steps,
                        vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                        learning_rate=args.lr, rms_prop_eps=args.eps, verbose=verbose, tensorboard_log=log_path,
                        create_eval_env=True, gae_lambda=args.gae)
        elif algorithm == "ppo":
            model = PPO("MlpPolicy", env, gamma=args.gamma, n_steps=args.num_steps,
                        vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                        learning_rate=args.lr, verbose=verbose, tensorboard_log=log_path)
        else:
            exit()
    else:
        if algorithm == "a2c":
            model = A2C("MlpPolicy", env, gamma=args.gamma, n_steps=args.num_steps,
                        vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                        learning_rate=args.lr, rms_prop_eps=args.eps, verbose=verbose, gae_lambda=args.gae)
        elif algorithm == "ppo":
            model = PPO("MlpPolicy", env, gamma=args.gamma, n_steps=args.num_steps,
                        vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                        learning_rate=args.lr, verbose=verbose)
        else:
            exit()

    if args.save:
        checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=save_path + "/models")
        # Separate evaluation env
        eval_callback = EvalCallback(model.get_env(), best_model_save_path=save_path + '/models/best_model',
                                     log_path=save_path + '/models/best_model', eval_freq=2500)
        # Create the callback list
        callback = CallbackList([checkpoint_callback, eval_callback])
        # Save model parameters to a text file
        with open(os.path.join(log_path, "model_parameters.txt"), "w") as f:
            f.write(str(model.get_parameters()))
        model.set_logger(new_logger)
        model.learn(total_timesteps=args.num_frames, callback=callback, eval_log_path=log_path)
    else:
        model.learn(total_timesteps=args.num_frames)

    if args.save:
        model.save(save_path)
    env.close()


if __name__ == "__main__":
    main()
