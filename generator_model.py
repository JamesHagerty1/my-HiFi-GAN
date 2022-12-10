import torch
import torch.nn as nn
from torch.nn import Conv1d
from torch.nn.utils import weight_norm


class Generator(torch.nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()

        # Embedding(K_discrete_units, embedding_dim)
        # x (batch_size, sequence_len) -> (batch_size, sequence_len, embedding_dim)
        self.lookup_table = nn.Embedding(config.num_embeddings, 
                                         config.embedding_dim)

        # Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        # x (batch_size, in_channels, seq_len) -> (batch_size, out_channels, seq_len)
        conv1 = Conv1d(config.embedding_dim, config.upsample_initial_channel, 
                       7, 1, padding=3)
        self.conv_pre = weight_norm(conv1)

    def forward(self, x):
        x = self.lookup_table(x)  
        x = x.transpose(1, 2)
        x = self.conv_pre(x)

        return x