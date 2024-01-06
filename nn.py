import torch
import torch.nn as nn
import torch.nn.functional as F

# ANN
class ANN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, activation, drop_ratio:float = 0.3):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        self.activation = activation
        self.drop_ratio = nn.Dropout(drop_ratio)

    def forward(self, x):
        x = self.lin1(x)
        x = self.activation(x)
        x = self.drop_ratio(x)
        x = self.lin2(x)
        x = F.sigmoid(x)
        return x