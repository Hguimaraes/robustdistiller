import os
import numpy as np
import pandas as pd
from glob import glob
from scipy import signal

import torch
import torchaudio
import torchaudio.functional as F


class Reverb:
    def __init__(
        self,
        reverb_root:str,
        sample_rate:int=16000,
        **kwargs
    ) -> None:
        self.root = reverb_root
        self.sample_rate = sample_rate

        self.fname_list = glob(os.path.join(self.root, "SLR28/**/*.wav"), recursive=True)

    def _sample(self) -> np.array:
        idx = torch.randint(0, len(self.fname_list), (1,))
        return self.fname_list[idx]

    def reverb(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        device = x.device
        rir_path = self._sample()
        rir, sr = torchaudio.load(rir_path)

        # Randomly select one channel for real RIR
        num_channels = rir.shape[0]
        if num_channels > 1:
            selected_channel = torch.randint(0, num_channels, (1,))
            rir = rir[selected_channel, :]

        # Resample if necessary
        if sr != self.sample_rate:
            rir = F.resample(
                waveform=rir,
                orig_freq=sr,
                new_freq=self.sample_rate,
                resampling_method="kaiser_window"
            )

        x, rir = x.cpu(), rir.cpu()
        reverbed = signal.convolve(x, rir, mode="full")
        return torch.Tensor(reverbed[:, :x.shape[1]]).to(device), rir_path


class AdditiveNoise:
    """
    This class applies additive from the noise using the
    MUSAN  and UrbanSound8K datasets.
    Information on where to download the dataset:
        https://www.openslr.org/17/
        https://urbansounddataset.weebly.com/
    Arguments
    ---------
    sample_rate : int
        Desired sample_rate for the noise waveform. 
        Should be the same as the waveform to be corrupted (e.g., clean speech)
    """
    def __init__(
        self, 
        sample_rate:int=16000,
        musan_root:str=None,
        urbansound_root:str=None,
        **kwargs
    ) -> None:
        self.musan_root = musan_root
        self.urbansound_root = urbansound_root
        self.sample_rate = sample_rate

        self.fname_list = self.__get_filepaths()

    def __get_filepaths(self):
        # Get files from MUSAN only using the noise folder
        musan_noise_list = glob(os.path.join(self.musan_root, "noise/**/*.wav"), recursive=True)

        # Get Urbansound
        metadata = os.path.join(self.urbansound_root, "metadata/UrbanSound8K.csv")
        df = pd.read_csv(metadata)
        
        remove_classes = ["children_playing", "street_music"]
        df = df[~df['class'].isin(remove_classes)]
        fname_list = df.apply(
            lambda row: 
            f'{self.urbansound_root}/audio/fold{row["fold"]}/{row["slice_file_name"]}',
        axis=1).tolist()
        fname_list.extend(musan_noise_list)

        return fname_list

    def insert_controlled_noise(
        self,
        signal:torch.Tensor,
        noise:torch.Tensor,
        desired_snr_db:float=5,
        eps:float=1e-8,
    ) -> torch.Tensor:
        # 1D signal
        signal, noise = signal.squeeze(0), noise.squeeze(0)

        # Calculate the power of signal and noise
        n = signal.shape[0]
        S_signal = signal.dot(signal) / n
        S_noise = noise.dot(noise) / n

        # Proportion factor
        K = (S_signal / (S_noise + eps)) * (10 ** (-desired_snr_db / 10))

        # Rescale the noise
        new_noise = torch.sqrt(K) * noise
        noisy = new_noise + signal
        return noisy.unsqueeze(0)

    def sample(self) -> torch.Tensor:
        fname_size = len(self.fname_list)
        fname = self.fname_list[torch.randint(0, fname_size, (1, ))]

        sig, sr = torchaudio.load(fname)
        num_channels = sig.shape[0]
        
        if num_channels > 1:
            # stereo to mono
            sig = sig.mean(dim=0).unsqueeze(0)
        
        if sr != self.sample_rate:
            sig = F.resample(
                waveform=sig,
                orig_freq=sr,
                new_freq=self.sample_rate,
                resampling_method="kaiser_window"
            )

        return sig, fname


class RobustEnvironment:
    def __init__(
        self,
        config
    ) -> None:
        self.config = config
        self.low_snr = config['low_snr']
        self.high_snr = config['high_snr']
        self.verbose = config['verbose']
        self.inc_steps = config['inc_steps_noises']
        self.step_counter = 0

        if self.inc_steps > 0:
            self.low_range = torch.round(
                torch.linspace(self.high_snr, self.low_snr, self.inc_steps)
            )
            self.prob_reverb_range = torch.linspace(0, 1, self.inc_steps)

        self.noise_sampler = AdditiveNoise(**self.config)
        self.reverb_sampler = Reverb(**self.config)

    def transform(self, waveform_input):
        x = []
        for wave in waveform_input:
            x.append(self.apply_transform(wave))

        self.step_counter = self.step_counter + 1
        return torch.stack(x)

    def apply_transform(self, x):
        # Insert number of channels
        x = x.unsqueeze(0)

        # Randomly select which operation is going to be performed
        op = torch.randint(0, 4, (1,)).item()
        func_map = {
            0: self.apply_noise,
            1: self.apply_reverberation,
            2: self.apply_noise_and_reverberation,
            3: self.apply_clean,
        }

        x, msg = func_map[op](x)
        if self.verbose:
            print(" ".join(['[Robustness]', msg]))

        # remove number of channels and return
        return x.squeeze(0)

    def apply_noise(self, x):
        if self.inc_steps > 0:
            idx = self.step_counter if self.step_counter < self.inc_steps else -1
            low = int(self.low_range[idx].item())
        else:
            low = self.low_snr

        snr = torch.randint(low, self.high_snr + 1, (1,)).item()
        if torch.rand(1) < 0.7:
            noise, n_fname = self.noise_sampler.sample()
        else:
            noise = torch.randn_like(x)
            n_fname = "white_noise"

        # If signals do not have the same side, make compatible
        if noise.shape != x.shape:
            n_samples = x.shape[1]
            noise = self.shape_noise(noise, n_samples)

        noise = noise.to(x.device)
        x = self.noise_sampler.insert_controlled_noise(x, noise, snr)

        msg = f'Applied Noise. Random SNR = {snr}. File: {n_fname}'
        return x, msg

    def apply_reverberation(self, x):
        idx, p = -1, 2
        if self.inc_steps > 0:
            idx = self.step_counter if self.step_counter < self.inc_steps else -1
            p = self.prob_reverb_range[idx]

        msg = 'Tried to apply Reverb. But probability was less than threshold'
        if torch.rand(1) < p:
            x, r_fname = self.reverb_sampler.reverb(x)
            msg = f'Applied Reverb. File: {r_fname}'

        return x, msg

    def apply_noise_and_reverberation(self, x):
        x, n_msg = self.apply_noise(x)
        x, r_msg = self.apply_reverberation(x)
        return x, " ".join([n_msg, r_msg])
    
    def apply_clean(self, x):
        msg = "No detrimental factors applied"
        return x, msg

    def shape_noise(self, noise, n_samples):
        m = noise.shape[1]

        # If the audio signal is longer then the noise
        if n_samples >= m:
            # Tile the noise signal
            factor = int(np.ceil(n_samples / m))
            noise = torch.tile(noise, dims=(factor,))
            noise = noise[:, :n_samples]
        else:
            start = torch.randint(0, m - n_samples + 1, (1,))
            noise = noise[:, start:start + n_samples]

        return noise
