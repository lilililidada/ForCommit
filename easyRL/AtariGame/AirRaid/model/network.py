import torch.nn
import torch.nn.functional as F


class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, out_dim) -> None:
        super().__init__()
        self.kernel_size = 3
        self.padding = 1
        self.stride = 2
        self.hidden_dim = 32
        self.linear_input = 32 * 4 * 4
        self.linear_output = 512
        self.input_layer = torch.nn.Conv2d(input_dim, self.hidden_dim, self.kernel_size, padding=self.padding,
                                           stride=self.stride)
        self.hidden_one_layer = torch.nn.Conv2d(self.hidden_dim, self.hidden_dim, self.kernel_size,
                                                padding=self.padding,
                                                stride=self.stride)
        self.hidden_two_layer = torch.nn.Conv2d(self.hidden_dim, self.hidden_dim, self.kernel_size,
                                                padding=self.padding,
                                                stride=self.stride)
        self.hidden_pool_layer = torch.nn.MaxPool2d(self.kernel_size, stride=2, padding=0)
        self.hidden_linear_layer = torch.nn.Linear(self.linear_input, self.linear_output)
        self.out_layer = torch.nn.Linear(512, out_dim)

    def forward(self, state):
        input_out = F.relu(self.input_layer(state))
        hidden_one_out = F.relu(self.hidden_one_layer(input_out))
        hidden_two_out = F.relu(self.hidden_two_layer(hidden_one_out))
        hidden_pool_out: torch.Tensor = self.hidden_pool_layer(hidden_two_out)
        hidden_pool_out = hidden_pool_out.resize(hidden_two_out.size(0), self.linear_input)
        linear_out = F.relu(self.hidden_linear_layer(hidden_pool_out))
        return self.out_layer(linear_out)


class NeuralNetwork2(torch.nn.Module):

    def __init__(self, input_dim, out_dim) -> None:
        super().__init__()
        self.kernel_size = 3
        self.padding = 0
        self.stride = 2
        self.hidden_dim = 32
        self.linear_input = self.hidden_dim * 4 * 4
        self.linear_output = 512
        self.input_layer = torch.nn.Conv2d(input_dim, self.hidden_dim, kernel_size=self.kernel_size, stride=self.stride,
                                           padding=self.padding)

        self.hidden_layer_1 = torch.nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=self.kernel_size,
                                              stride=self.stride,
                                              padding=self.padding)
        self.hidden_layer_2 = torch.nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=self.kernel_size,
                                              stride=self.stride, padding=self.padding)
        self.hidden_layer_3 = torch.nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=self.kernel_size,
                                              stride=self.stride, padding=self.padding)
        self.hidden_linear = torch.nn.Linear(self.linear_input, self.linear_output)
        self.out_layer = torch.nn.Linear(self.linear_output, out_dim)

    def size_compute(self, input_size, cov_num):
        cur_size = input_size
        for i in range(cov_num):
            cur_size = (cur_size + 2 * self.padding - self.kernel_size) // self.stride + 1
        return cur_size * cur_size * self.hidden_dim

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        x = F.relu(self.hidden_layer_1(x))
        x = F.relu(self.hidden_layer_2(x))
        x = F.relu(self.hidden_layer_3(x))
        x = self.hidden_linear(x.view(x.size(0), self.linear_input))
        return self.out_layer(x)


class AdvantageActorCritic(torch.nn.Module):
    def __init__(self, input_dim, action_dim, state_shape) -> None:
        super().__init__()
        self.state_shape = state_shape
        self.kernel_size = 5
        self.padding = 1
        self.stride = 2
        self.conv_hidden_dim = 32
        self.linear_hidden_dim = 512
        self.conv_linear_input = 512
        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_dim, out_channels=self.conv_hidden_dim, kernel_size=self.kernel_size,
                            padding=self.padding, stride=self.stride),
            # 维度含义： 1、是否有物体; 2+N、类型（敌机1，敌机2，敌机3，子弹，我方）; 7、位置1(左开始); 8、位置2(右结束); 9、位置3(上边界); 10、位置4(下边界)
            # 输出数据维度为(6, 6, 10)
            torch.nn.Conv2d(in_channels=self.conv_hidden_dim, out_channels=10,
                            kernel_size=5, stride=self.stride),
            # 输出数据维度为(1, 1, linear_hidden_dim)
            torch.nn.Conv2d(in_channels=10, out_channels=self.linear_hidden_dim, kernel_size=5, stride=2),
            torch.nn.Conv2d(in_channels=self.linear_hidden_dim, out_channels=self.linear_hidden_dim, kernel_size=1),
            torch.nn.ReLU()
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(self.linear_hidden_dim, action_dim),
            torch.nn.Softmax(dim=1)

        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(self.linear_hidden_dim, 1)
        )
        self._initialize_weights()

    def forward(self, state):
        conv_feature = self.conv_layer(state)
        probs = self.actor(conv_feature.view(conv_feature.size(0), self.conv_linear_input))
        # 概率分布
        dist = torch.distributions.Categorical(probs)
        value = self.critic(conv_feature.view(conv_feature.size(0), self.conv_linear_input))
        return dist, value

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                torch.nn.init.orthogonal_(module.weight, torch.nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(module.bias, 0)


