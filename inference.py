import json
import os

from librosa.util import normalize
from scipy.io.wavfile import write
import soundfile
import torch

from generator_model import Generator
from utils import AttrDict, mel_spectrogram, MAX_WAV_VALUE
from speech_resynthesis_data import get_speech_resynthesis_data


CHECKPOINT_FILE = "checkpoints/testG0"
GENERATED_FILE_DIR = "generated_files/"


def main():
    with open("hubert100_lut.json") as f:
        data = f.read()
    h = AttrDict(json.loads(data))

    torch.cuda.manual_seed(h.seed)
    device = torch.device("cuda:0")

    generator = Generator(h).to(device)
    state_dict_g = torch.load(CHECKPOINT_FILE, map_location=device)
    generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    generator.remove_weight_norm()

    audio_file_list, discrete_units_list, _ = get_speech_resynthesis_data(2)

    with torch.no_grad():
        for audio_file, discrete_units in zip(audio_file_list, discrete_units_list):
            print(audio_file)

            discrete_units = torch.LongTensor(discrete_units).to(device)
            x = discrete_units.view(1, discrete_units.shape[0])

            y_g_hat = generator(x)
            
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            write(GENERATED_FILE_DIR + audio_file.split("/")[-1], 
                  h.sampling_rate, audio)


if __name__ == "__main__":
    main()