# From https://github.com/facebookresearch/speech-resynthesis
# (minimally edited for different G input)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

from utils import get_padding, init_weights


LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                                        padding=get_padding(kernel_size, dilation[0]))), weight_norm(
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                   padding=get_padding(kernel_size, dilation[1]))), weight_norm(
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                   padding=get_padding(kernel_size, dilation[2])))])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
             weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
             weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                                       padding=get_padding(kernel_size, dilation[0]))), weight_norm(
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                   padding=get_padding(kernel_size, dilation[1])))])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.lookup_table = nn.Embedding(h.num_embeddings, 
                                         h.embedding_dim)
        conv1 = Conv1d(h.embedding_dim, h.upsample_initial_channel, 
                       7, 1, padding=3)
        self.conv_pre = weight_norm(conv1)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel // (2 ** i), 
                                h.upsample_initial_channel // (2 ** (i + 1)), 
                                k, u, padding=(k - u) // 2)))     
        
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.lookup_table(x)  
        x = x.transpose(1, 2)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses