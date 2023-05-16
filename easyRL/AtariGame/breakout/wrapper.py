import collections

import cv2
import gym
import numpy


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
        self.width = 105
        self.height = 80
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.width, self.height, self.frame_num),
                                                dtype=env.observation_space.dtype)
        # 重定义动作 0 为什么都不做 1 为左 2 为右
        # self.action_space = gym.spaces.Discrete(3)
        self.max_reward = 200
        self.sum_removed_bricks = 0

    def adjust_observation(self, obs):
        # down-sampling
        obs = obs[::2, ::2, ::]
        # 转为黑白
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # (2, 25, 77, 100)
        crop_obs = gray_obs[25:100, 2:77]
        frame = cv2.resize(crop_obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        result, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        return frame

    def reset(self, **kwargs):
        self.sum_removed_bricks = 0
        obs = self.env.reset(**kwargs)
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
            frame_state, reward, done, info = self.env.step(action)
            # 调整后 shape (105, 80)
            adjust_state = self.adjust_observation(frame_state)
            self.frame_stack.append(adjust_state)
            sum_reward += reward
            if reward:
                self.sum_removed_bricks += 1
            if done:
                for _ in range(i + 1, self.press_frame):
                    self.frame_stack.append(adjust_state)
                sum_reward += self.sum_removed_bricks
                break
        return numpy.stack([self.frame_stack[i] for i in range(self.frame_num)], axis=-1), sum_reward / self.max_reward, done, info


class FireResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        return obs
