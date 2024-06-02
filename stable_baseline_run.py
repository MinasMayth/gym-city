import gym
import gym_city
import time
from datetime import datetime
from arguments import get_args
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
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


def make_env(args, log_path):
    if args.vec_envs > 1:
        def make_env_vec(env_id):
            def _init():
                env = gym.make(env_id)
                env.setMapSize(args.map_width, render_gui=False)
                return env

            return _init

        env_id = args.env_name

        # List of environment creation functions
        env_fns = [make_env_vec(env_id) for _ in range(args.vec_envs)]

        # Create SubprocVecEnv
        env = SubprocVecEnv(env_fns)
        env.seed(args.seed)
        env = VecMonitor(env, os.path.join(log_path, "vec_monitor_log.csv"))
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
            # SCHEDULE
            if args.lr_schedule:
                if algorithm == "a2c":
                    model = A2C("MlpPolicy", env, policy_kwargs=policy_kwargs, gamma=args.gamma, n_steps=args.num_steps,
                                vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef,
                                max_grad_norm=args.max_grad_norm,
                                learning_rate=linear_schedule(args.lr), rms_prop_eps=args.eps, verbose=verbose,
                                tensorboard_log=log_path,
                                gae_lambda=args.gae, seed=args.seed, use_rms_prop=True, use_sde=False)
                elif algorithm == "ppo":
                    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, gamma=args.gamma, n_steps=args.num_steps,
                                batch_size=args.num_mini_batch, n_epochs=args.ppo_epoch, clip_range=args.clip_param,
                                vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef,
                                max_grad_norm=args.max_grad_norm,
                                learning_rate=linear_schedule(args.lr), verbose=verbose, tensorboard_log=log_path,
                                gae_lambda=args.gae, seed=args.seed, use_sde=False)
                elif algorithm == "dqn":
                    model = DQN("MlpPolicy", env, gamma=args.gamma, learning_rate=linear_schedule(args.lr),
                                buffer_size=args.buffer_size, learning_starts=args.learning_starts,
                                batch_size=args.batch_size,
                                tau=args.tau, target_update_interval=args.target_update_interval, verbose=verbose,
                                tensorboard_log=log_path, seed=args.seed)
                else:
                    raise NotImplementedError
            else:
                # NO SCHEDULE
                if algorithm == "a2c":
                    model = A2C("MlpPolicy", env, policy_kwargs=policy_kwargs, gamma=args.gamma, n_steps=args.num_steps,
                                vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef,
                                max_grad_norm=args.max_grad_norm,
                                learning_rate=args.lr, rms_prop_eps=args.eps, verbose=verbose,
                                tensorboard_log=log_path,
                                gae_lambda=args.gae, seed=args.seed, use_rms_prop=True, use_sde=False)
                elif algorithm == "ppo":
                    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, gamma=args.gamma, n_steps=args.num_steps,
                                batch_size=args.num_mini_batch, n_epochs=args.ppo_epoch, clip_range=args.clip_param,
                                vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef,
                                max_grad_norm=args.max_grad_norm,
                                learning_rate=(args.lr), verbose=verbose, tensorboard_log=log_path, gae_lambda=args.gae,
                                seed=args.seed, use_sde=False)
                elif algorithm == "dqn":
                    model = DQN("MlpPolicy", env, gamma=args.gamma, learning_rate=args.lr,
                                buffer_size=args.buffer_size, learning_starts=args.learning_starts,
                                batch_size=args.batch_size,
                                tau=args.tau, target_update_interval=args.target_update_interval, verbose=verbose,
                                seed=args.seed, tensorboard_log=log_path)

                else:
                    raise NotImplementedError
        else:  # NO SAVE
            if algorithm == "a2c":
                model = A2C("MlpPolicy", env, policy_kwargs=policy_kwargs, gamma=args.gamma, n_steps=args.num_steps,
                            vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                            learning_rate=(args.lr), rms_prop_eps=args.eps, verbose=verbose, gae_lambda=args.gae,
                            seed=args.seed, use_rms_prop=True, use_sde=False)
            elif algorithm == "ppo":
                model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, gamma=args.gamma, n_steps=args.num_steps,
                            vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                            learning_rate=(args.lr), verbose=verbose, gae_lambda=args.gae, seed=args.seed,
                            use_sde=False)
            elif algorithm == "dqn":
                model = DQN("MlpPolicy", env, gamma=args.gamma, learning_rate=(args.lr),
                            buffer_size=args.buffer_size, learning_starts=args.learning_starts,
                            batch_size=args.batch_size,
                            tau=args.tau, target_update_interval=args.target_update_interval, verbose=verbose,
                            seed=args.seed)
            else:
                raise NotImplementedError
    else:
        if algorithm == "a2c":
            model = A2C.load(args.load_dir, env)
        elif algorithm == "ppo":
            model = PPO.load(args.load_dir, env)
        elif algorithm == "dqn":
            model = DQN.load(args.load_dir, env)
        else:
            raise NotImplementedError
    return model


def obtain_log_path(args):
    # Get current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Define the path with the current date and time
    # save_path = os.path.join("trained_models", "baseline_models", algorithm, current_datetime)
    if args.algo == "a2c":
        parameter_values = {
            'n_steps': str(args.num_steps),
            'map_w': str(args.map_width),
            'gamma': str(args.gamma),
            'v_l_coef': str(args.value_loss_coef),
            'e_coef': str(args.entropy_coef),
            'max_grad_norm': str(args.max_grad_norm),
            'lr': str(args.lr),
            'seed': str(args.seed),
            'eps': str(args.eps),
            'lambda': str(args.gae)
        }
        # Generate a string representation of parameters
        parameter_string = "_".join([f"{key}={value}" for key, value in parameter_values.items()])
        ALICE_path = '/home/s3458717/data1/'
        log_path = os.path.join(ALICE_path, "logs", "new", "new_grid_search", args.algo,
                                f"{parameter_string}_{current_datetime}")
    elif args.algo == "ppo":
        parameter_values = {
            'n_steps': str(args.num_steps),
            'map_w': str(args.map_width),
            'clip_range': str(args.clip_param),
            'batch_size': str(args.num_mini_batch),
            'n_epochs': str(args.ppo_epoch),
            'v_l_coef': str(args.value_loss_coef),
            'e_coef': str(args.entropy_coef),
            'lr': str(args.lr),
            'eps': str(args.eps),
            'gamma': str(args.gamma),
            'max_grad_norm': str(args.max_grad_norm),
            'lambda': str(args.gae),
            'seed': str(args.seed)
        }
        # Generate a string representation of parameters
        parameter_string = "_".join([f"{key}={value}" for key, value in parameter_values.items()])
        ALICE_path = '/home/s3458717/data1/'
        log_path = os.path.join(ALICE_path, "logs", "new", "new_reward_experiments", args.algo,
                                f"{parameter_string}_{current_datetime}")
    elif args.algo == "dqn":
        parameter_values = {
            'lrng_starts': str(args.learning_starts),
            'map_w': str(args.map_width),
            'batch_size': str(args.batch_size),
            'updt_intvl': str(args.target_update_interval),
            'bffr_size': str(args.buffer_size),
            'lr': str(args.lr),
            'gamma': str(args.gamma),
            'seed': str(args.seed)
        }
        # Generate a string representation of parameters
        parameter_string = "_".join([f"{key}={value}" for key, value in parameter_values.items()])
        ALICE_path = '/home/s3458717/data1/'
        log_path = os.path.join(ALICE_path, "logs", "new", "testing", args.algo,
                                f"{parameter_string}_{current_datetime}")
        save_path = log_path
    else:
        raise NotImplementedError

    return log_path


def main():
    args = get_args()
    print(args)

    algorithm = args.algo

    if args.log:
        verbose = 1
    else:
        verbose = 0

    log_path = obtain_log_path(args)

    if args.save:
        os.makedirs(log_path, exist_ok=True)
        new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
        save_to_text_file(args, os.path.join(log_path, "arguments.txt"))
        #changes = ("Limited toolset. Gamespeed 3. Complex Reward. No Static Build")
        changes = ("Limited Toolset. Gamespeed 3. Reward is simple total population "
                   "+ total powered zones + road net reward of 1 and penalty of -1 + special traffic (5 roads). Static Build.")
        make_change_log(log_path, changes)

    env = make_env(args, log_path)

    model = create_model(args, algorithm, env, verbose, log_path)
    print("POLICY:", model.policy)

    if args.save:
        if args.vec_envs > 1:
            checkpoint_callback = CheckpointCallback(save_freq=max(args.save_interval // args.vec_envs, 1),
                                                     save_path=log_path + "/models")
        else:
            checkpoint_callback = CheckpointCallback(save_freq=max(args.save_interval // args.vec_envs, 1),
                                                     save_path=log_path + "/models")
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
        model.save(log_path + "/models")
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
