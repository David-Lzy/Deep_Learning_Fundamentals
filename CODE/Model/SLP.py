import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


# Define the Perceptron model
class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        # Define the single linear layer
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Pass the input through the linear layer
        # and then through the sigmoid activation function
        out = torch.sigmoid(self.fc1(x))
        return out
