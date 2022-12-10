import soundfile
from librosa.util import normalize
from torch.utils.data import Dataset, DataLoader

from speech_resynthesis_data import get_speech_resynthesis_data


MAX_WAV_VALUE = 32768.0


class HifiGanDataset(Dataset):
    def __init__(self, audio_file_list, discrete_units_list, config):
        self.audio_file_list = audio_file_list
        self.discrete_units_list = discrete_units_list
        self.sampling_rate = config["sampling_rate"]

    def __getitem__(self, index):
        audio_file = self.audio_file_list[index]
        print(audio_file)

        audio, sampling_rate = soundfile.read(audio_file, dtype="int16")
        if sampling_rate != self.sampling_rate:
            raise ValueError("Incorrect sampling_rate")

        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95

        
        print("audio shape ", audio.shape)

        return 

    def __len__(self):
        return len(self.audio_file_list)


def dataloader_init(config):
    audio_file_list, discrete_units_list, _ = get_speech_resynthesis_data(16)
    dataset = HifiGanDataset(audio_file_list, discrete_units_list, config)

    dataset.__getitem__(0)
    dataset.__getitem__(1)

    dataloader = DataLoader(dataset, 
                            num_workers=config["num_workers"], 
                            batch_size=config["batch_size"])
    return dataloader