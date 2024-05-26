from multiprocessing import Pipe, Process
import numpy as np
from stable_baselines3.common.vec_env import CloudpickleWrapper, VecEnv
import gym

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_param_bounds':
            param_bounds = env.get_param_bounds()
            remote.send(param_bounds)
        elif cmd == 'set_params':
            env.set_params(data)
            remote.send(None)
        elif cmd == 'set_param_bounds':
            env.set_param_bounds(data)
            remote.send(None)
        elif cmd == 'render':
            env.render()
            remote.send(None)
        elif hasattr(env, cmd):
            cmd_fn = getattr(env, cmd)
            if isinstance(data, dict):
                ret_val = cmd_fn(**data)
            else:
                ret_val = cmd_fn(*data)
            remote.send(ret_val)
        else:
            raise NotImplementedError

class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def env_is_wrapped(self, wrapper_class, indices=None):
        if indices is None:
            indices = range(len(self.remotes))
        result = []
        for idx in indices:
            self.remotes[idx].send(('env_is_wrapped', wrapper_class))
        return [remote.recv() for remote in self.remotes]

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if indices is None:
            indices = range(len(self.remotes))
        for idx in indices:
            self.remotes[idx].send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in self.remotes]

    def get_attr(self, attr_name, indices=None):
        if indices is None:
            indices = range(len(self.remotes))
        for idx in indices:
            self.remotes[idx].send(('get_attr', attr_name))
        return [remote.recv() for remote in self.remotes]

    def seed(self, seed=None):
        for idx, remote in enumerate(self.remotes):
            remote.send(('seed', seed + idx if seed is not None else None))
        return [remote.recv() for remote in self.remotes]

    def set_attr(self, attr_name, value, indices=None):
        if indices is None:
            indices = range(len(self.remotes))
        for idx in indices:
            self.remotes[idx].send(('set_attr', (attr_name, value)))
        return [remote.recv() for remote in self.remotes]
