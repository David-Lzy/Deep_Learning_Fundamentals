import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


class SimpleCNNRNN(nn.Module):
    def __init__(
        self, num_channels, output_size, data_len, hidden_size, num_rnn_layers
    ):
        super(SimpleCNNRNN, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        self.pool_size = 2  # Assuming pool size is 2 for both max_pool1d
        reduced_data_len = data_len // self.pool_size // self.pool_size

        # RNN Layer
        self.rnn = nn.RNN(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
        )

        # Fully connected layer
        self.fc1 = nn.Linear(hidden_size * reduced_data_len, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        # Reshape x to (batch_size, channels, length)
        x = x.view(x.size(0), 23, -1)

        # CNN layers
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, self.pool_size)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, self.pool_size)

        # Flatten and reshape for RNN
        x = x.view(x.size(0), -1, 128)  # Reshape to (batch_size, seq_len, features)
        x, _ = self.rnn(x)

        # Flatten the tensor for the linear layer
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FlexibleRNN(nn.Module):
    def __init__(
        self,
        conv_layers_config,
        num_channels,
        output_size,
        data_len,
        hidden_size,
        num_rnn_layers,
    ):
        """
        :param conv_layers_config: List of tuples for convolution layers configuration.
                                   Each tuple is (num_filters, kernel_size, padding).
        :param num_channels: Number of input channels.
        :param output_size: The size of the output layer.
        :param data_len: The length of the input data.
        :param hidden_size: The size of the hidden layer in RNN.
        :param num_rnn_layers: The number of RNN layers.
        """
        super(FlexibleRNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.pool_size = 2

        in_channels = num_channels
        for out_channels, kernel_size, padding in conv_layers_config:
            # 计算卷积层之后的长度，假设步长为1
            data_len = (data_len - kernel_size + 2 * padding) + 1

            # 计算池化层之后的长度
            data_len = data_len // self.pool_size

            # 添加卷积层
            self.conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
            )
            in_channels = out_channels

        # RNN Layer
        self.rnn = nn.RNN(
            input_size=out_channels,
            hidden_size=hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
        )

        # Fully connected layer
        self.fc1 = nn.Linear(hidden_size * data_len, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        # 假设 x 的初始形状为 (batch_size, 47, 23)，无需重塑
        # Apply convolution layers
        x = x.permute(0, 2, 1)  # 重塑为 (batch_size, new_seq_len, new_channels)

        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
            x = F.max_pool1d(x, self.pool_size)

        # Flatten and reshape for RNN
        # 我们需要知道卷积层和池化层处理后的序列长度，这里假设为 new_seq_len
        # x 的形状现在应该是 (batch_size, new_channels, new_seq_len)
        # RNN 需要的形状是 (batch_size, seq_len, features)，所以进行适当的重塑

        x = x.permute(0, 2, 1)  # 重塑为 (batch_size, new_seq_len, new_channels)

        # RNN 层
        x, _ = self.rnn(x)

        # Flatten the tensor for the linear layer
        # 由于 RNN 层的输出形状为 (batch_size, seq_len, hidden_size)，我们需要进行展平
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
