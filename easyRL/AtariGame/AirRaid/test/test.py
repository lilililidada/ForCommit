import gym

from easyRL.AtariGame.AirRaid.task import TrainTask
from easyRL.AtariGame.AirRaid.wrapper import EnvWrapper

model_path = r'D:\DayDayUp\easyRL\AtariGame\AirRaid\outputs\model\20220725-005745\models\\'

if __name__ == '__main__':
    env: gym.Env = gym.make("ALE/AirRaid-v5", render_mode="human")
    env = EnvWrapper(env)
    env.reset(seed=11111)
    task = TrainTask(env, 500)
    task.load(model_path)
    task.test(1000)
