import collections

import gym
import numpy
from PIL import Image


class BreakOutWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 每一episode 有 N 帧图片
        self.frame_num = 4
        # 按压帧数
        self.press_frame = 1
        # 保持的历史frame数量
        self.max_frame_len = 4
        self.frame_stack = collections.deque(maxlen=self.max_frame_len)
        # 重定义observation大小
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(75, 75, self.frame_num), dtype=numpy.uint8)
        # 重定义动作 0 为什么都不做 1 为左 2 为右
        # self.action_space = gym.spaces.Discrete(3)
        self.max_reward = 200
        self.sum_removed_bricks = 0
        self.lives = 0

    @staticmethod
    def adjust_observation(obs):
        # 模糊化图片
        obs = obs[::2, ::2, ::]
        # 转为黑白
        gray_obs = numpy.array(Image.fromarray(obs).convert("L"), dtype=numpy.uint8)
        # (2, 25, 77, 100)
        crop_obs = gray_obs[25:100, 2:77]
        return crop_obs

    def reset(self, **kwargs):
        self.sum_removed_bricks = 0
        super().reset(**kwargs)
        # 当屏幕没有弹珠时，需要按1开始
        obs, _, _, info = super().step(1)
        self.lives = info['lives']
        adjust_obs = self.adjust_observation(obs)
        self.frame_stack.clear()
        for _ in range(self.frame_num):
            self.frame_stack.append(adjust_obs)
        return numpy.stack([self.frame_stack[i] for i in range(self.frame_num)], axis=-1)

    def step(self, action):
        sum_reward = 0
        done = False
        info = None
        # 保持按压固定步数
        for i in range(self.press_frame):
            # observation shape (210, 160, 3)
            frame_state, reward, done, info = super().step(action)
            # 调整后 shape (105, 80)
            adjust_state = self.adjust_observation(frame_state)
            self.frame_stack.append(adjust_state)
            sum_reward += reward
            if reward:
                self.sum_removed_bricks += 1
            if self.is_loss_life(info):
                done = True
            if done:
                for _ in range(i + 1, self.press_frame):
                    self.frame_stack.append(adjust_state)
                sum_reward += self.sum_removed_bricks
                break
        return numpy.stack([self.frame_stack[i] for i in range(self.frame_num)],
                           axis=-1), sum_reward / self.max_reward, done, info

    def is_loss_life(self, info) -> bool:
        return self.lives > info["lives"] > 0