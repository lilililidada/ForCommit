import gym
import numpy as np
from PIL import Image

from easyRL.AtariGame.AirRaid.task import TrainTask
from easyRL.AtariGame.AirRaid.wrapper import EnvWrapper

if __name__ == '__main__':
    env: gym.Env = gym.make("ALE/AirRaid-v5", render_mode="human")
    env = EnvWrapper(env)
    env.reset(seed=11111)
    task = TrainTask(env, 500)
    task.load(r'D:\DayDayUp\easyRL\AtariGame\AirRaid\outputs\model\20220701-085212\models\\')
    task.test(1000)
