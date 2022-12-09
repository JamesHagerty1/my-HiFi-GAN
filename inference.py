import torch
import numpy
from torch.nn import Conv1d

from generator_model import Generator


def main():
    G = Generator()

    hidden_units = torch.ones(1, 1, 500) # (batch_size, in_channels, sequence_len)
    G(hidden_units)


if __name__ == "__main__":
    main()