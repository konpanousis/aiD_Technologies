# A simple python script to load the ccs10 greek dataset for training an ASR model.
# Developed in the context of the aiD project
import os

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
import codecs
import pandas as pd
import librosa
import re

chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\»\«\´\'\́a-z]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower() + " "
    return batch

def load_css10_item(line, path, folder_audio):
    # Each line as the following data:
    # client_id, path, sentence, up_votes, down_votes, age, gender, accent


    line = line.strip()
    wav_path, original_script, normalized_script, audio_duration = line.strip().split('|')
    speaker_id = wav_path.split('/')[0]
    wav_path_22k = os.path.join(path,  wav_path)
    waveform, sample_rate = torchaudio.load(wav_path_22k)
    resample_wav = librosa.resample(waveform.numpy(), sample_rate, 16000)

    data = {
        'path': wav_path_22k,
        'audio': resample_wav,
        'path_16k': None,
        'sentence': remove_special_characters(normalized_script),
        'sample_rate': 16000,
        'duration': audio_duration,
        'speaker_id': speaker_id,
        'language': 'el',
    }

    return data


class CSS10(Dataset):
    """
    Create a Dataset for CSS10.
    """

    _ext_txt = ".txt"
    _ext_audio = ".wav"
    _folder_audio = "Paramythi_horis_onoma"

    def __init__(self, path, transform = None):

        fpaths, text_lengths, texts = [], [], []
        transcript = os.path.join(path, 'transcript.txt')
        lines = codecs.open(transcript, 'r', 'utf-8').readlines()
        self._lines = list(lines)
        self._path = path
        self._folder_audio = "Paramythi_horis_onoma"


    def __getitem__(self,idx):
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, dictionary)``,  where dictionary is built
            from the TSV file with the following keys: ``client_id``, ``path``, ``sentence``,
            ``up_votes``, ``down_votes``, ``age``, ``gender`` and ``accent``.
        """
        line = self._lines[idx]
        return load_css10_item(line,  self._path, self._folder_audio)


    def __len__(self):
        return len(self._lines)


if __name__=='__main__':

    ds = CSS10('archive/el')
    for i in range(len(ds)):
        sample = ds[i]

        print(sample)

        if i == 3:
            break
    print(len(ds))
