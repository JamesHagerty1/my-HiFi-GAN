import json

import torch
import torchaudio


AUDIO_DIR = "../data/LJSpeech-1.1/wavs_16khz/"
UNITS_AUDIO_MAP_FILE = "../speech-resynthesis/datasets/LJSpeech/hubert100/train.txt"


# speech-resynthesis has a dataset showing 16khz LJSpeech file names and the
# discrete unit tensors that hubert100 outputs for them
def parse_sr_dataset_line(line):
    d = json.loads(line.replace("'", '"'))
    audio_file = AUDIO_DIR + d["audio"].split("/")[-1]
    discrete_units = [int(num) for num in d["hubert"].split(" ")]
    duration = d["duration"]
    return audio_file, discrete_units, duration


def main():
    with open(UNITS_AUDIO_MAP_FILE) as f:
        audio_file, discrete_units, duration = parse_sr_dataset_line(f.readline())
        waveform, sample_rate = torchaudio.load(audio_file)

        print(audio_file)
        print(len(discrete_units))
        print(duration)
        print(waveform.shape)
        print(sample_rate)


if __name__ == "__main__":
    main()


# fb speech resyn uses these for discrete units
# codes += [torch.LongTensor(
#     [int(x) for x in sample[k].split(' ')]
# ).numpy()]