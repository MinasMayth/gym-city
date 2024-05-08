import gym
import gym_city
import time
from datetime import datetime
from arguments import get_args
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from networks import CustomActorCriticPolicy
from typing import Callable
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
        env.env_method("setMapSize", args.map_width)
    else:
        env = gym.make(args.env_name)
        env.setMapSize(args.map_width, render_gui=args.render)
    return env


def make_change_log(log_path, changes):
    """
    Function to save a text string to a .txt file detailing changes made.

    :param log_path: Path to the directory where the log file will be saved.
    :param changes: Text string describing the changes made.
    """
    change_log_file = os.path.join(log_path, "change_log.txt")
    with open(change_log_file, "a") as f:
        f.write(changes + "\n")


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def create_model(args, algorithm, env, verbose, log_path):

    policy_kwargs = dict(net_arch=[128, 128, 128, dict(vf=[64, 64], pi=[64])])
    if args.load_dir is None:
        if args.save:
            if args.lr_schedule:
                if algorithm == "a2c":
                    model = A2C("MlpPolicy", env, policy_kwargs=policy_kwargs, gamma=args.gamma, n_steps=args.num_steps,
                                vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                                learning_rate=linear_schedule(args.lr), rms_prop_eps=args.eps, verbose=verbose, tensorboard_log=log_path,
                                create_eval_env=True, gae_lambda=args.gae)
                elif algorithm == "ppo":
                    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, gamma=args.gamma, n_steps=args.num_steps,
                                batch_size=args.num_mini_batch, n_epochs=args.ppo_epoch, clip_range=args.clip_param,
                                vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                                learning_rate=linear_schedule(args.lr), verbose=verbose, tensorboard_log=log_path)
                else:
                    exit()
            else:
                if algorithm == "a2c":
                    model = A2C("MlpPolicy", env, policy_kwargs=policy_kwargs, gamma=args.gamma, n_steps=args.num_steps,
                                vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                                learning_rate=(args.lr), rms_prop_eps=args.eps, verbose=verbose, tensorboard_log=log_path,
                                create_eval_env=True, gae_lambda=args.gae)
                elif algorithm == "ppo":
                    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, gamma=args.gamma, n_steps=args.num_steps,
                                batch_size=args.num_mini_batch, n_epochs=args.ppo_epoch, clip_range=args.clip_param,
                                vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                                learning_rate=(args.lr), verbose=verbose, tensorboard_log=log_path)
                else:
                    exit()
        else:
            if algorithm == "a2c":
                model = A2C("MlpPolicy", env, policy_kwargs=policy_kwargs, gamma=args.gamma, n_steps=args.num_steps,
                            vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                            learning_rate=(args.lr), normalize_advantage=True
                            , rms_prop_eps=args.eps, verbose=verbose, gae_lambda=args.gae)
            elif algorithm == "ppo":
                model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, gamma=args.gamma, n_steps=args.num_steps,
                            vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                            learning_rate=(args.lr), verbose=verbose)
            else:
                exit()
    else:
        if algorithm == "a2c":
            model = A2C.load(args.load_dir, env)
        elif algorithm == "ppo":
            model = PPO.load(args.load_dir, env)
        else:
            exit()
    return model


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
    if algorithm == "a2c":
        parameter_values = {
            'alpha': str(args.alpha),
            'num_steps': str(args.num_steps),
            'map_width': str(args.map_width),
            'gamma': str(args.gamma),
            #'value_loss_coef': str(args.value_loss_coef),
            #'entropy_coef': str(args.entropy_coef),
            #'max_grad_norm': str(args.max_grad_norm),
            'lr': str(args.lr),
            #'eps': str(args.eps),
            'lambda': str(args.gae)
        }
        # Generate a string representation of parameters
        parameter_string = "_".join([f"{key}={value}" for key, value in parameter_values.items()])
        ALICE_path = '/home/s3458717/data1/'
        log_path = os.path.join(ALICE_path, "logs", "new", "reward_experiments", algorithm,
                                f"{parameter_string}_{current_datetime}")
        save_path = log_path
    elif algorithm == "ppo":
        parameter_values = {
            'alpha': str(args.alpha),
            'num_steps': str(args.num_steps),
            'map_width': str(args.map_width),
            'clip_range': str(args.clip_param),
            'batch_size': str(args.num_mini_batch),
            'n_epochs': str(args.ppo_epoch),
            'lr': str(args.lr),
            'eps': str(args.eps)
        }
        # Generate a string representation of parameters
        parameter_string = "_".join([f"{key}={value}" for key, value in parameter_values.items()])
        ALICE_path = '/home/s3458717/data1/'
        log_path = os.path.join(ALICE_path, "logs", "new", algorithm,
                                f"{parameter_string}_{current_datetime}")
        save_path = log_path
    else:
        ALICE_path = '/home/s3458717/data1/'
        log_path = os.path.join(ALICE_path, "logs", "new", algorithm, current_datetime)
        save_path = log_path

    if args.save:
        os.makedirs(log_path, exist_ok=True)
        new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
        save_to_text_file(args, os.path.join(save_path, "arguments.txt"))
        changes = "Limited Toolset. Gamespeed 1. Reward is simple total population + total zones with more advanced penalty for individual roads. Static Build."
        make_change_log(log_path, changes)

    env = make_env(vec=False, args=args)

    model = create_model(args, algorithm, env, verbose, log_path)
    print("POLICY:", model.policy)

    if args.save:
        checkpoint_callback = CheckpointCallback(save_freq=args.save_interval, save_path=save_path + "/models")
        # Separate evaluation env
        eval_callback = EvalCallback(model.get_env(), best_model_save_path=save_path + '/models/best_model',
                                     log_path=save_path + '/models/best_model', eval_freq=250_000)
        # Create the callback list
        callback = CallbackList([checkpoint_callback])
        # Save model parameters to a text file
        with open(os.path.join(log_path, "model_parameters.txt"), "w") as f:
            f.write(str(model.get_parameters()))
            f.write(str(model.policy))
        model.set_logger(new_logger)
        if args.load_dir is None:
            model.learn(total_timesteps=args.num_frames, callback=callback, eval_log_path=log_path)
        else:
            model.learn(total_timesteps=args.num_frames, callback=callback,
                        eval_log_path=log_path, reset_num_timesteps=True)
    else:
        if args.load_dir is None:
            model.learn(total_timesteps=args.num_frames)
        else:
            model.learn(total_timesteps=args.num_frames, reset_num_timesteps=True)

    if args.save:
        model.save(save_path + "/models")
    env.close()

def save_to_text_file(args, file_path):
    try:
        # Open the file in write mode
        with open(file_path, 'w') as file:
            # Write the contents of the args variable to the file
            file.write(str(args))
        print("Data successfully saved to", file_path)
    except Exception as e:
        print("Error occurred while saving to file:", str(e))

if __name__ == "__main__":
    main()
