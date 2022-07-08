import random


class ExperiencePool:
    def __init__(self, fix_size) -> None:
        super().__init__()
        self.fix_size = fix_size
        self.pool = []
        self.current = 0

    def put(self, state, reward, action, next_state, done):
        experience = (state, reward, action, next_state, done)
        if len(self.pool) < self.fix_size:
            self.pool.append(experience)
            return
        if self.current >= self.fix_size:
            self.current = self.current % self.fix_size
        self.pool[self.current] = experience
        self.current += 1

    def sample(self, batch_size: int) -> list:
        return random.sample(self.pool, batch_size)

    def __len__(self):
        return len(self.pool)
