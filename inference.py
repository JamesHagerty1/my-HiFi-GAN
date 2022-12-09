import torch
import numpy
from torch.nn import Conv1d

from generator_model import Generator


def main():
    G = Generator()

    discrete_units = torch.ones(1, 42, dtype=torch.long) # (batch_size, sequence_len)
    G(discrete_units)


if __name__ == "__main__":
    main()