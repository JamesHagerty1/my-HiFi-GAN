import json

from torch.utils.data import DataLoader

from data import dataloader_init

def main():
    with open("hubert100_lut.json") as f:
        data = f.read()
    config = json.loads(data)

    dataloader = dataloader_init(config)

    # for i, batch in enumerate(dataloader):
    #     print(batch)



if __name__ == "__main__":
    main()