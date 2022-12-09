import torch
import torch.nn as nn
from torch.nn import Conv1d
from torch.nn.utils import weight_norm


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        # x (batch_size, in_channels, seq_len) -> (batch_size, out_channels, seq_len)
        self.conv_pre = weight_norm(Conv1d(1, 512, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        print(x.shape)

        return x