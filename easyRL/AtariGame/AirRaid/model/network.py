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
        self.linear_input = self.hidden_dim * 5 * 5
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
