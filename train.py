import itertools
import json

import torch
from torch.utils.data import DataLoader

from data import dataloader_init
from generator_model import Generator
from discriminator_models import MultiPeriodDiscriminator, MultiScaleDiscriminator, discriminator_loss
from utils import AttrDict, mel_spectrogram


EPOCHS = 3100


def main():
    with open("hubert100_lut.json") as f:
        data = f.read()
    h = AttrDict(json.loads(data))

    torch.manual_seed(h.seed)
    torch.cuda.manual_seed(h.seed)
    device = torch.device("cuda:0")

    dataloader = dataloader_init(h)
    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), h.learning_rate,
                                betas=[h.adam_b1, h.adam_b2])

    for epoch in range(EPOCHS):
        for i, batch in enumerate(dataloader):
            x, y, _, y_mel = batch

            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            y_g_hat = generator(x)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)

            # Discriminators' losses
            optim_d.zero_grad()

            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            
         
            # if (i == 100):
            #     torch.save({'generator': generator.state_dict()},
            #                f"checkpoints/testG{i}")

            break
        break    
    

if __name__ == "__main__":
    main()