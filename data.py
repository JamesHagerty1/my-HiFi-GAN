import random

import soundfile
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from speech_resynthesis_data import get_speech_resynthesis_data


TRAINING_SAMPLE_COUNT = 100
MAX_WAV_VALUE = 32768.0


# From github.com/facebookresearch/speech-resynthesis/blob/main/dataset.py
mel_basis = {}
hann_window = {}
def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, 
                    fmax, center=False):
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, 
                             fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = \
            torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
    y = torch.nn.functional.pad(y.unsqueeze(1), 
        (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, 
                      window=hann_window[str(y.device)], center=center, 
                      pad_mode='reflect', normalized=False, onesided=True, 
                      return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = torch.log(torch.clamp(spec, min=1e-5) * 1)
    return spec


class HifiGanDataset(Dataset):
    def __init__(self, audio_file_list, discrete_units_list, config):
        self.audio_file_list = audio_file_list
        self.discrete_units_list = discrete_units_list
        self.sampling_rate = config["sampling_rate"]
        self.code_hop_size = config["code_hop_size"]
        self.segment_size = config["segment_size"]
        self.n_fft = config["n_fft"]
        self.num_mels = config["num_mels"]
        self.hop_size = config["hop_size"]
        self.win_size = config["win_size"]
        self.fmin = config["fmin"]
        self.fmax_for_loss = config["fmax_for_loss"]

    # From github.com/facebookresearch/speech-resynthesis/blob/main/dataset.py
    def _sample_interval(self, seqs, seq_len=None):
        N = max([v.shape[-1] for v in seqs])
        if seq_len is None:
            seq_len = self.segment_size if self.segment_size > 0 else N
        hops = [N // v.shape[-1] for v in seqs]
        lcm = np.lcm.reduce(hops)
        interval_start = 0
        interval_end = N // lcm - seq_len // lcm
        start_step = random.randint(interval_start, interval_end)
        new_seqs = []
        for i, v in enumerate(seqs):
            start = start_step * (lcm // hops[i])
            end = (start_step + seq_len // lcm) * (lcm // hops[i])
            new_seqs += [v[..., start:end]]
        return new_seqs

    def __getitem__(self, index):
        audio_file = self.audio_file_list[index]

        audio, sampling_rate = soundfile.read(audio_file, dtype="int16")

        if (sampling_rate != self.sampling_rate):
            raise Exception("Incorrect sampling_rate")

        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95

        code_length = min(audio.shape[0] // self.code_hop_size, 
                          self.discrete_units_list[index].shape[0])
        code = self.discrete_units_list[index][:code_length]
        audio = audio[:code_length * self.code_hop_size]

        if (audio.shape[0] // self.code_hop_size != code.shape[0]):
            raise Exception("Incorrect shape")

        while (audio.shape[0] < self.segment_size):
            audio = np.hstack([audio, audio])
            code = np.hstack([code, code])

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if (audio.size(1) < self.segment_size):
            raise Exception("Incorrect audio size")

        audio, code = self._sample_interval([audio, code])

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
            self.sampling_rate, self.hop_size, self.win_size, self.fmin, 
            self.fmax_for_loss, center=False)

        return code.squeeze(), audio.squeeze(0), audio_file, mel_loss.squeeze()

    def __len__(self):
        return len(self.audio_file_list)


def dataloader_init(config):
    audio_file_list, discrete_units_list, _ = \
        get_speech_resynthesis_data(TRAINING_SAMPLE_COUNT)
    dataset = HifiGanDataset(audio_file_list, discrete_units_list, config)
    dataloader = DataLoader(dataset, num_workers=config["num_workers"], 
                            batch_size=config["batch_size"])
    return dataloader