import torch
import torch.nn as nn


class MLP_fix(nn.Module):
    def __init__(
        self,
        input_dim,
        network_structure=(8, 8),
        act_fun=torch.relu,
        dropout_rate=0.0,  # 添加 dropout_rate 参数
    ):
        super(MLP_fix, self).__init__()
        self.act_fun = act_fun
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()  # 创建一个 ModuleList 用于存储 dropout 层
        if isinstance(network_structure, int):
            network_structure = (network_structure,)
        # 添加第一个线性层（从输入维度到隐藏维度）
        self.layers.append(nn.Linear(input_dim, network_structure[0]))
        self.dropouts.append(nn.Dropout(dropout_rate))  # 为每个线性层添加一个 dropout 层

        # 添加剩下的隐藏层
        for i in range(1, len(network_structure)):
            self.layers.append(
                nn.Linear(network_structure[i - 1], network_structure[i])
            )
            self.dropouts.append(nn.Dropout(dropout_rate))  # 为每个线性层添加一个 dropout 层

        # 添加最后一个线性层（从隐藏维度到输出维度）
        self.fc_last = nn.Linear(network_structure[-1], 1)

    def forward(self, x):
        for layer, dropout in zip(self.layers, self.dropouts):
            # Pass the input through the linear layer, activation function, and dropout layer
            x = dropout(self.act_fun(layer(x)))
        # Pass the output through the last linear layer
        # Because this is a binary classification problem,
        # sigmoid is a more appropriate activation function
        out = torch.sigmoid(self.fc_last(x))
        return out
