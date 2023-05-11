import os
import random
import time

import gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, EpisodicLifeEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from easyRL.AtariGame.breakout.wrapper import BreakOutWrapper, FireResetWrapper

env_num = 1
log_dir = 'logs'
save_dir = 'trained_models_test'
total_study_step = 10000000
batch_size = 32
buffer_size = 100000
gamma = 0.99
tau = 1e-3
exploration_fraction = 0.1
exploration_initial_eps = 1
exploration_final_eps = 0.1


def initial_env(env_name, seed=0):
    env = gym.make(env_name)
    env = EpisodicLifeEnv(env)
    env = FireResetWrapper(env)
    env = MaxAndSkipEnv(env)
    wrap_env = BreakOutWrapper(env)
    monitor_env = Monitor(wrap_env)
    return monitor_env


def learning_rate_schedule(start=1e-5, end=1e-6):
    def schedule(process_remain):
        """
        process 由大到小
        """
        return end + (start - end) * process_remain

    return schedule


def main():
    init_dir()
    envs = [(lambda: initial_env("ALE/Breakout-v5", seed=i)) for i in range(env_num)]
    # 并行训练环境
    env = SubprocVecEnv(list(envs))
    # 模型算法
    model = load_model(env)

    # 保存训练中间态
    checkpoint_interval = 300000
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="breakout_")

    # 日志与开始训练
    model.learn(total_timesteps=total_study_step,
                callback=[checkpoint_callback])
    env.close()


def init_dir():
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)


def load_model(env, path: str = None):
    if path:
        model = DQN.load(path, env=env, device="cuda")
        model.exploration_schedule = lambda process: 0.01
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
            learning_rate=1e-5,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            policy_kwargs={},
            tau=tau
        )
    return model


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
