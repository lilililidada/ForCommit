import gym
import keyboard

from easyRL.AtariGame.AirRaid.task import TrainTask
from easyRL.AtariGame.AirRaid.wrapper import EnvWrapper

model_path = r'D:\DayDayUp\easyRL\AtariGame\AirRaid\outputs\model\20220901-133224\models\\'


def play():
    env: gym.Env = gym.make("ALE/AirRaid-v5", render_mode="human")
    env = EnvWrapper(env)
    for i in range(500):
        done = False
        env.reset()
        while not done:
            action = from_keyboard()
            _, _, done, _ = env.step(action)


def from_keyboard() -> int:
    key_name = keyboard.read_key()
    print(key_name)
    if key_name == 'left':
        return 3
    if key_name == 'right':
        return 2
    if key_name == 'up':
        return 1
    return 0


def test():
    env: gym.Env = gym.make("ALE/AirRaid-v5", render_mode="human")
    env = EnvWrapper(env)
    env.reset(seed=11111)
    task = TrainTask(env, 500)
    task.load(model_path)
    task.agent.epsilon = lambda x: 0
    task.test(500)


if __name__ == '__main__':
    play()
