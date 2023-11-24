import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

class FlexibleLSTM(nn.Module):
    def __init__(
        self,
        conv_layers_config,
        num_channels,
        output_size,
        data_len,
        hidden_size,
        num_lstm_layers,
    ):
        """
        :param conv_layers_config: List of tuples for convolution layers configuration.
                                   Each tuple is (num_filters, kernel_size, padding).
        :param num_channels: Number of input channels.
        :param output_size: The size of the output layer.
        :param data_len: The length of the input data.
        :param hidden_size: The size of the hidden layer in LSTM.
        :param num_lstm_layers: The number of LSTM layers.
        """
        super(FlexibleLSTM, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.pool_size = 2

        in_channels = num_channels
        for out_channels, kernel_size, padding in conv_layers_config:
            self.conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
            )
            in_channels = out_channels

        # Calculate the length of data after all convolution and pooling layers
        for _ in conv_layers_config:
            data_len = data_len // self.pool_size

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        # Fully connected layer
        self.fc1 = nn.Linear(hidden_size * data_len, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        # Reshape x to (batch_size, channels, length)
        x = x.view(x.size(0), 23, -1)

        # Apply convolution layers
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
            x = F.max_pool1d(x, self.pool_size)

        # Flatten and reshape for LSTM
        x = x.view(
            x.size(0), -1, x.size(1)
        )  # Reshape to (batch_size, seq_len, features)
        x, _ = self.lstm(x)

        # Flatten the tensor for the linear layer
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
