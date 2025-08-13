from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import random
import torch
import numpy as np
import lightning as L
from librosa.filters import mel as librosa_mel_fn
import math
import os
import random
import torch
import torch.utils.data
import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn
import pathlib
from tqdm import tqdm
from typing import List, Tuple, Optional

MAX_WAV_VALUE = 32767.0  # NOTE: 32768.0 -1 to prevent int16 overflow (results in popping sound in corner cases)


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)


mel_basis_cache = {}
hann_window_cache = {}


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int = None,
    center: bool = False,
) -> torch.Tensor:
    """
    Calculate the mel spectrogram of an input signal.
    This function uses slaney norm for the librosa mel filterbank (using librosa.filters.mel) and uses Hann window for STFT (using torch.stft).

    Args:
        y (torch.Tensor): Input signal.
        n_fft (int): FFT size.
        num_mels (int): Number of mel bins.
        sampling_rate (int): Sampling rate of the input signal.
        hop_size (int): Hop size for STFT.
        win_size (int): Window size for STFT.
        fmin (int): Minimum frequency for mel filterbank.
        fmax (int): Maximum frequency for mel filterbank. If None, defaults to half the sampling rate (fmax = sr / 2.0) inside librosa_mel_fn
        center (bool): Whether to pad the input to center the frames. Default is False.

    Returns:
        torch.Tensor: Mel spectrogram.
    """
    if torch.min(y) < -1.0:
        print(f"[WARNING] Min value of input waveform signal is {torch.min(y)}")
    if torch.max(y) > 1.0:
        print(f"[WARNING] Max value of input waveform signal is {torch.max(y)}")

    device = y.device
    key = f"{n_fft}_{num_mels}_{sampling_rate}_{hop_size}_{win_size}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
        hann_window_cache[key] = torch.hann_window(win_size).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(
        y.unsqueeze(1), (padding, padding), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = spectral_normalize_torch(mel_spec)

    return mel_spec


class audioDatasetModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers,train_path,val_path):
        super(audioDatasetModule).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset_path = train_path 
        self.val_dataset_path = val_path    
        self.prepare_data_per_node = True
        self._log_hyperparams=False
        self.allow_zero_length_dataloader_with_multiple_devices = False

    def pad_collate_fn(self,data):
        # return pad_sequence(data, batch_first=True)
        # return pad_sequence(*data)
        is_one_data = not isinstance(data[0], tuple)
        outputs = []
        if is_one_data:
            for datum in data:
                if isinstance(datum, torch.Tensor):
                    output = datum.unsqueeze(0)
                else:
                    output = torch.tensor([datum])
                outputs.append(output)
            return tuple(outputs)        
        for datum in zip(*data):
            if isinstance(datum[0], torch.Tensor):
                output = pad_sequence(datum, batch_first=True)
            else:
                output = torch.tensor(list(datum))
            outputs.append(output)
        return tuple(outputs)
    

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_file_list = []
            for path in self.train_dataset_path:
                with open(path, 'r') as f:
                    train_file_list.extend(f.readlines())
            random.shuffle(train_file_list)
            self.train = audioDataset(train_file_list,65536,False)
        if stage == "test" or stage is None:
            self.test = audioDataset(self.val_dataset_path,65536,False)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,collate_fn=self.pad_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False, collate_fn=self.pad_collate_fn)

class audioDataset(Dataset):
    def __init__(self,
                 file_list,
                 segment_size,
                 if_val):
        super().__init__()
        self.file_list = file_list
        self.segment_size = segment_size
        self.sample_rate = 24000
        self.downsample_rate = 320
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        file = self.file_list[index].strip()
        audio_file, feature_file = file.split('\t')
        audio, sr = torchaudio.load(audio_file)
        feature = torch.from_numpy(np.load(feature_file)).squeeze(0)
        audio = audio.mean(axis=0)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        if audio.size(-1) > self.segment_size:
            max_audio_start = audio.size(-1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio_start_16k = int(audio_start * 16 / 24)
            segment_size_16k= 43691
            audio= audio[audio_start:audio_start+self.segment_size]
            feature_start = min(int(audio_start_16k / self.downsample_rate), feature.size(0) - 136)
            feature = feature[feature_start:feature_start + 136, :]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(-1)), 'constant')
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)  # [B(1), self.segment_size]
        
        
        mel = mel_spectrogram(
                    audio,
                    1024,
                    100,
                    24000,
                    256,
                    1024,
                    0,
                    None,
                    center=False,
                )
        return audio.squeeze(0),feature,mel.squeeze()
    

