import json

import torch
from torch.utils.data import DataLoader

from data import dataloader_init
from generator_model import Generator
from utils import AttrDict


EPOCHS = 3100


def main():
    with open("hubert100_lut.json") as f:
        data = f.read()
    config = AttrDict(json.loads(data))

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    device = torch.device("cuda:0")

    dataloader = dataloader_init(config)
    generator = Generator(config).to(device)

    for epoch in range(EPOCHS):
        for i, batch in enumerate(dataloader):
            x, y, _, y_mel = batch

            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)
            print("* ok")

            y_g_hat = generator(x)
         
            break
        break    
    


if __name__ == "__main__":
    main()