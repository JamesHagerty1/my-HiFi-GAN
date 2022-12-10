import json
import torch

from generator_model import Generator

from utils import AttrDict


CHECKPOINT_FILE = "checkpoints/testG0"
TEST_FILE_DIR = "test_files/"


def main():
    with open("hubert100_lut.json") as f:
        data = f.read()
    h = AttrDict(json.loads(data))

    device = torch.device("cuda:0")
    generator = Generator(h).to(device)
    state_dict_g = torch.load(CHECKPOINT_FILE, map_location=device)
    generator.load_state_dict(state_dict_g['generator'])

    # generator.eval()
    # generator.remove_weight_norm()
    # with torch.no_grad():
    #     pass


if __name__ == "__main__":
    main()