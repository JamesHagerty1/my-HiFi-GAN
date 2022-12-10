import json

import torch
import torchaudio


AUDIO_DIR = "../data/LJSpeech-1.1/wavs_16khz/"
UNITS_AUDIO_MAP_FILE = \
    "../speech-resynthesis/datasets/LJSpeech/hubert100/train.txt"


# speech-resynthesis has a dataset showing 16khz LJSpeech file names and the
# discrete unit tensors that hubert100 outputs for them
def parse_line(line):
    d = json.loads(line.replace("'", '"'))
    audio_file = AUDIO_DIR + d["audio"].split("/")[-1]
    discrete_units = \
        torch.LongTensor([int(num) for num in d["hubert"].split(" ")]).numpy()
    duration = d["duration"]
    return audio_file, discrete_units, duration


def get_speech_resynthesis_data(N):
    audio_file_list = []
    discrete_units_list = []
    duration_list = []

    with open(UNITS_AUDIO_MAP_FILE) as f:
        for sample in range(N):
            audio_file, discrete_units, duration = parse_line(f.readline())
            audio_file_list.append(audio_file)
            discrete_units_list.append(discrete_units)
            duration_list.append(duration)

    return audio_file_list, discrete_units_list, duration_list