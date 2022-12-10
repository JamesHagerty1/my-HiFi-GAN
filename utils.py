# From https://github.com/facebookresearch/speech-resynthesis

from librosa.filters import mel as librosa_mel_fn
import torch


MAX_WAV_VALUE = 32768.0


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


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


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)