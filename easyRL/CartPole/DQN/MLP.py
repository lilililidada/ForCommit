from torch import nn
from torch.nn import functional as func


class MLP(nn.Module):
    """
    多层感知机
    Multilayer Perceptron
    """

    def __init__(self, n_states, n_actions, hidden_dim=128):
        """
        构造方法

        @param n_states: 输入的特征维度
        @param n_actions: 输出的动作维度
        @param hidden_dim 隐层维度
        """
        super().__init__()
        self.experience = []
        # 输入层
        self.fc1 = nn.Linear(n_states, hidden_dim)
        # 隐层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 输出层
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        input_out = func.relu(self.fc1(x))
        hidden_out = func.relu(self.fc2(input_out))
        return self.fc3(hidden_out)
