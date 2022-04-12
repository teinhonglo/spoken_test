import os
import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchaudio

#ORIGINAL_SAMPLE_RATE = 16000
SAMPLE_RATE = 16000
EXAMPLE_WAV_MAX_SEC = 10


class AudioSLUDataset(Dataset):
    def __init__(self, df, base_path, So_fluency):
        self.df = df
        self.base_path = base_path
        self.max_length = SAMPLE_RATE * EXAMPLE_WAV_MAX_SEC
        self.So_fluency = So_fluency
        #self.speaker_name = speaker_name
        #self.resampler = torchaudio.transforms.Resample(ORIGINAL_SAMPLE_RATE, SAMPLE_RATE)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wav_path = self.df.loc[idx]['wav.scp']
        utt_id = self.df.loc[idx]['utt_id']
        wav, sr = torchaudio.load(wav_path)
        wav = wav.squeeze(0)
        
        label = []

        for slot in ["sent.fluency"]:
            value = self.df.loc[idx][slot]
            label.append(self.So_fluency[slot][value])
        
        return wav, torch.tensor(label).long(), utt_id

    def collate_fn(self, samples):
        wavs, labels, utt_ids = [], [], []

        for wav, label, utt_id in samples:
            wavs.append(wav)
            labels.append(label)
            utt_ids.append(utt_id)

        return wavs, labels, utt_ids