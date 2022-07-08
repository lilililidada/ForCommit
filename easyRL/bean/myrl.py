import abc


class QLearning(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self):
        pass

    @abc.abstractmethod
    def choose_action(self, state):
        pass
