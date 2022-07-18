from typing import Tuple, Union

import gym
import numpy as np
from PIL import Image
from gym import Env
from gym.core import ActType, ObsType


class EnvWrapper(gym.Wrapper):

    def __init__(self, env: Env):
        super(EnvWrapper, self).__init__(env)
        self.life = 2
        self.shot_action = [1, 4, 5]
        self.peace_frame = 0

    def adjust_picture(self, state):
        adjust_state = Image.fromarray(state)
        # 转黑白
        zoom_state = adjust_state.convert("L")
        zoom_state.thumbnail((80, 125), Image.ANTIALIAS)
        width, height = zoom_state.size
        # 裁剪掉多余的图片部分
        crop_state = zoom_state.crop((0, 10, width, height - 35))
        return np.array(crop_state).reshape((1, 80, 80)) / 255

    def step(self, action):
        observation, reward, done, info = super().step(action)
        # 判断是否丢失生命，如果丢失就施加惩罚
        if self.is_loss_life(observation, done):
            reward -= 200
            self.life -= 1
        if reward:
            self.peace_frame = 0
        reward += self.peace_loss()
        # 计算射击价值
        reward += self.shot_cost(observation, action)
        # 调整图片大小
        observation = self.adjust_picture(observation)
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.life = 2
        return self.adjust_picture(observation)

    def is_loss_life(self, observation: np, done):
        """
        当有两条命时，在219行，有2两命以下时，就跳到了220

        @param observation: 图片
        @return: 是否失去生命
        """
        life1 = 1 if observation[219][58][0] else 0
        life2 = 1 if observation[219][66][0] else 0
        life_sum_1 = life1 + life2
        if life_sum_1:
            return life_sum_1 < self.life
        life3 = 1 if observation[220][58][0] else 0
        life4 = 1 if observation[220][66][0] else 0
        life_sum_2 = life3 + life4
        if life_sum_2:
            return life_sum_2 < self.life
        return done

    def shot_cost(self, observation: np, action):
        """
        这里先简单判断乱射击会扣分

        @param observation: 图片
        @param action: 动作
        @return:
        """
        if int(action) in self.shot_action:
            return -3
        return 0.

    def peace_loss(self):
        """
        如果长时间拿不到reward，则随时间增长惩罚越高
        """
        return self.peace_frame // 10 * -5
