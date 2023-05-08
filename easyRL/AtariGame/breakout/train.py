import os
import random
import sys
import time

import gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from easyRL.AtariGame.breakout.wrapper import BreakOutWrapper

env_num = 1
log_dir = 'logs'
save_dir = 'trained_models_test'
total_study_step = 10000000
batch_size = 64
buffer_size = 10 * batch_size
gamma = 0.9
model_cache_path = sys.argv[1] if len(sys.argv) > 1 else None


def initial_env(env_name, seed=0):
    env = gym.make(env_name)
    wrap_env = BreakOutWrapper(env)
    monitor_env = Monitor(wrap_env)
    return monitor_env


def learning_rate_schedule(start=1e-6, end=1e-4):
    def schedule(process):
        return start + (end - start) * process

    return schedule


def main():
    init_dir()
    envs = [(lambda: initial_env("ALE/Breakout-v5", seed=i)) for i in range(env_num)]
    # 并行训练环境
    env = SubprocVecEnv(list(envs))
    # 模型算法
    model = get_model(env, path=model_cache_path)

    # 保存训练中间态
    checkpoint_interval = 300000
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="breakout_")

    # 日志与开始训练
    model.learn(total_timesteps=total_study_step, callback=[checkpoint_callback])


def get_model(env, path=None):
    if path:
        model = DQN.load(path, env=env)
    else:
        model = DQN(
            policy="CnnPolicy",
            env=env,
            device="cuda",
            verbose=1,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tensorboard_log=log_dir,
            learning_starts=buffer_size,
            learning_rate=learning_rate_schedule()
        )
    return model


def init_dir():
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)


def play():
    env: gym.Env = gym.make("ALE/Breakout-v5", render_mode="human")
    env = BreakOutWrapper(env)
    r = random.Random()
    for i in range(500):
        done = False
        env.reset()
        while not done:
            time.sleep(0.015)
            action = r.randint(0, 3)
            _, _, done, _ = env.step(action)


if __name__ == '__main__':
    main()
