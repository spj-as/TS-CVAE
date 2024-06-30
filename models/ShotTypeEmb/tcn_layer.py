import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn.utils import weight_norm

class TCNLayer(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, dropout=0.2):
        super(TCNLayer, self).__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd to ensure same dimension output.")
        padding = (kernel_size - 1) // 2
        self.conv1 = weight_norm(nn.Conv1d(input_size, output_size, kernel_size, stride=1, padding=padding))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.dropout(self.relu(self.conv1(x)))
        return out