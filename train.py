import json

from torch.utils.data import DataLoader

from data import dataloader_init
from generator_model import Generator
from utils import AttrDict


def main():
    with open("hubert100_lut.json") as f:
        data = f.read()
    config = AttrDict(json.loads(data))

    dataloader = dataloader_init(config)

    generator = Generator(config)

    for i, batch in enumerate(dataloader):
        x, y, _, y_mel = batch
        
        y_g_hat = generator(x)
        print(y_g_hat.shape)

        break


if __name__ == "__main__":
    main()