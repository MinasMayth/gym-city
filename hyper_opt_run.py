import random
import hyperopt
import stable_baselines3.common.evaluation
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope
import gym
import gym_city
import time
from datetime import datetime
from arguments import get_args
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
# from networks import CustomActorCriticPolicy
from typing import Callable
from CustomNetwork import CustomActorCriticPolicy, CustomCNN
import os
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike


def seed_everything(seed, env):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    env.seed(seed)


def make_env(args, log_path):
    if args.vec_envs > 1:
        def make_env_vec(env_id):
            def _init():
                env = gym.make(env_id)
                if "Micropolis" in env_id:
                    env.setMapSize(args.map_width, render_gui=False)
                return env

            return _init

        env_id = args.env_name

        # List of environment creation functions
        env_fns = [make_env_vec(env_id) for _ in range(args.vec_envs)]

        # Create SubprocVecEnv
        env = SubprocVecEnv(env_fns)

        seed_everything(args.seed, env)

        if args.save:
            env = VecMonitor(env, os.path.join(log_path, "vec_monitor_log.csv"))
    else:
        env = gym.make(args.env_name)
        if "Micropolis" in args.env_name:
            env.setMapSize(args.map_width, render_gui=args.render)
        seed_everything(args.seed, env)
        if args.save:
            env = Monitor(env, os.path.join(log_path, "monitor_log.csv"))
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
    # policy_kwargs = dict(net_arch=[128, 128, 128, dict(vf=[64, 64], pi=[64])])
    # if args.custom_extractor == True:
    #
    # else:
    #     policy_kwargs = dict(
    #         net_arch=[64, 64, dict(vf=[64], pi=[256])],
    #     )

    policy_kwargs = dict(
        net_arch=[64, 64, dict(vf=[64], pi=[256])],
        # optimizer_class=RMSpropTFLike,
        # optimizer_kwargs=dict(eps=1e-5),
        # features_extractor_class=CustomCNN,
        # features_extractor_kwargs=dict(features_dim=env.action_space.n),
    )

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
                                learning_rate=args.lr, verbose=verbose, tensorboard_log=log_path, gae_lambda=args.gae,
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
                            learning_rate=args.lr, rms_prop_eps=args.eps, verbose=verbose, gae_lambda=args.gae,
                            seed=args.seed, use_rms_prop=True, use_sde=False)
            elif algorithm == "ppo":
                model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, gamma=args.gamma, n_steps=args.num_steps,
                            vf_coef=args.value_loss_coef, ent_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                            learning_rate=args.lr, verbose=verbose, gae_lambda=args.gae, seed=args.seed,
                            use_sde=False)
            elif algorithm == "dqn":
                model = DQN("MlpPolicy", env, gamma=args.gamma, learning_rate=args.lr,
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
            'env': str(args.env_name),
            'n_steps': str(args.num_steps),
            'map_w': str(args.map_width),
            'gamma': str(args.gamma),
            'v_l_coef': str(args.value_loss_coef),
            'e_coef': str(args.entropy_coef),
            'max_grad_norm': str(args.max_grad_norm),
            'lr': str(args.lr),
            'seed': str(args.seed),
            'eps': str(args.eps),
            'lambda': str(args.gae),
            'vec_envs': str(args.vec_envs)
        }
        # Generate a string representation of parameters
        parameter_string = "_".join([f"{key}={value}" for key, value in parameter_values.items()])
        log_path = os.path.join("logs", "local", "hpo", args.algo,
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
            'seed': str(args.seed),
            'vec_envs': str(args.vec_envs)
        }
        # Generate a string representation of parameters
        parameter_string = "_".join([f"{key}={value}" for key, value in parameter_values.items()])
        log_path = os.path.join("logs", "local", "hpo", args.algo,
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
        log_path = os.path.join("logs", "local", "hpo", args.algo,
                                f"{parameter_string}_{current_datetime}")
        save_path = log_path
    else:
        raise NotImplementedError

    return log_path


def objective(params):
    args = get_args()
    args.lr = params['lr']
    args.gamma = params['gamma']
    args.num_steps = params['num_steps']
    args.entropy_coef = params['entropy_coef']
    args.value_loss_coef = params['value_loss_coef']
    args.max_grad_norm = params['max_grad_norm']
    args.gae = params['lambda']
    # args.num_mini_batch = params['num_mini_batch']
    # args.ppo_epoch = params['ppo_epoch']
    # args.clip_param = params['clip_param']
    # args.buffer_size = params['buffer_size']
    # args.batch_size = params['batch_size']
    # args.learning_starts = params['learning_starts']
    # args.target_update_interval = params['target_update_interval']

    if args.log:
        verbose = 1
    else:
        verbose = 0

    log_path = obtain_log_path(args)

    if args.save:
        os.makedirs(log_path, exist_ok=True)
        new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
        save_to_text_file(args, os.path.join(log_path, "arguments.txt"))
        changes = ("LT. Gamespeed 3. Reward is pop + road adjacency"
                   "+ No Forced Static Build & Old State Representation.")
        make_change_log(log_path, changes)

    env = make_env(args, log_path)
    model = create_model(args, args.algo, env, verbose, log_path)

    if args.save:
        checkpoint_callback = CheckpointCallback(save_freq=max(args.save_interval // args.vec_envs, 1),
                                                 save_path=log_path + "/models")
        eval_callback = EvalCallback(eval_env=env, best_model_save_path=log_path + '/models/best_model',
                                     log_path=log_path + '/models/best_model', eval_freq=max(2000 // args.vec_envs, 1),
                                     deterministic=False, render=False)
        callback = CallbackList([checkpoint_callback])
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

    mean_r, std_r = stable_baselines3.common.evaluation.evaluate_policy(model, env, deterministic=False)

    env.close()

    return {'loss': 1 / mean_r, 'status': STATUS_OK}


def save_to_text_file(args, file_path):
    try:
        # Open the file in write mode
        with open(file_path, 'w') as file:
            # Write the contents of the args variable to the file
            file.write(str(args))
        print("Data successfully saved to", file_path)
    except Exception as e:
        print("Error occurred while saving to file:", str(e))


def main():
    search_space = {
        'lr': hp.uniform('lr', 1e-5, 1e-3),
        'gamma': hp.uniform('gamma', 0.9, 0.999),
        'lambda': hp.uniform('lambda', 0.9, 0.999),
        'num_steps': scope.int(hp.quniform('num_steps', 5, 50, 5)),
        'entropy_coef': hp.uniform('entropy_coef', 0.00, 0.01),
        'value_loss_coef': hp.choice('value_loss_coef', [0.5, 1.0]),
        'max_grad_norm': hp.choice('max_grad_norm', [0.5, 1.0]),
        # 'num_mini_batch': scope.int(hp.quniform('num_mini_batch', 4, 64, 1)),
        # 'ppo_epoch': scope.int(hp.quniform('ppo_epoch', 1, 10, 1)),
        # 'clip_param': hp.uniform('clip_param', 0.1, 0.4),
        # 'batch_size': scope.int(hp.quniform('batch_size', 16, 256, 16)),
    }
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100,
        show_progressbar=True
    )
    print("Best parameters found: ", best)


if __name__ == "__main__":
    main()
