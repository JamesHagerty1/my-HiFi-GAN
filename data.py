import json

import torch.utils.data

from speech_resynthesis_data import get_speech_resynthesis_data


CONFIG_FILE = "config_v3.json"


class HifiGanDataset(torch.utils.data.Dataset):
    def __init__(self,
                 audio_file_list):
        self.audio_file_list = audio_file_list

    def __getitem__(self, index):
        return None

    def __len__(self):
        return len(self.audio_file_list)


def main():
    with open(CONFIG_FILE) as f:
        data = f.read()
    config = json.loads(data)

    audio_file_list, discrete_units_list, _ = get_speech_resynthesis_data(3)

    dataset = HifiGanDataset(audio_file_list)


if __name__ == "__main__":
    main()