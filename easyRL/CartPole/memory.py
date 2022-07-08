import random


class ExpBuffer:
    """
    存放 action, state, reward, next_state
    """

    def __init__(self, capacity) -> None:
        super().__init__()
        self.buffer = []
        self.capacity = capacity
        self.position = -1

    def push(self, state, action, reward, next_state, done) -> None:
        """
        队列，当超出capacity时，会丢弃队头数据

        @param state:  当前状态
        @param action:  动作
        @param reward:  奖励
        @param next_state:  执行action后的状态
        """
        item = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[(self.position + 1) % self.capacity] = item

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = zip(*batch)
        return states, actions, rewards, next_states, done

    def __len__(self):
        return len(self.buffer)
