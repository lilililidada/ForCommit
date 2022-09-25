import gym
import numpy as np
from PIL import Image
from gym import Env


class EnvWrapper(gym.Wrapper):

    def __init__(self, env: Env):
        super(EnvWrapper, self).__init__(env)
        self.life = 2
        self.shot_action = [1, 4, 5]
        self.peace_frame = 0

    def adjust_picture(self, state):
        adjust_state = Image.fromarray(state, "RGB")
        # 转黑白
        zoom_state = adjust_state.convert("L")
        zoom_state.thumbnail((32, 50), Image.ANTIALIAS)
        width, height = zoom_state.size
        # 裁剪掉多余的图片部分
        crop_state = zoom_state.crop((0, 3, width, height - 15))
        return np.array(crop_state).reshape((1, 32, 32)) / 255

    def step(self, action):
        # 这里因为有些敌机会闪烁，只取一帧的话可能导致时而有时而无，所以一步走两帧
        observation_1, reward_1, done, _ = super().step(action)
        if done:
            return self.adjust_picture(observation_1), reward_1, done, _
        observation_2, reward_2, done, info = super().step(0)
        observation, reward = ((observation_1 + observation_2) // 2, reward_1 + reward_2)
        # 有奖励，重置和平计时
        # if reward:
        #     self.peace_frame = 0
        # else:
        #     self.peace_frame += 1
        reward += self.peace_loss()
        # 计算射击价值
        # reward += self.shot_cost(observation, action)
        # 调整图片大小
        observation = self.adjust_picture(observation)
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.life = 2
        return self.adjust_picture(np.array(observation))

    def is_loss_life(self, observation: np):
        """
        判断是否失去生命

        @param observation: 图片
        @return: 是否失去生命
        """
        target_row = observation[0][29] * 255
        for num in target_row:
            if num >= 78:
                return True
        return False

    def shot_cost(self, observation: np, action):
        """
        这里先简单判断乱射击会扣分

        @param observation: 图片
        @param action: 动作
        @return:
        """
        if int(action) in self.shot_action:
            return -10
        return 0.

    def peace_loss(self):
        """
        如果长时间拿不到reward，则随时间增长惩罚越高
        """
        return max((self.peace_frame // -20), -5) if self.peace_frame else 0
