# A python script to combine two different greek datasets to train an ASR model.
# Developed in the context of the aiD project

import torch, re

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import  Wav2Vec2Processor
from datasets import load_dataset, Audio, Dataset, concatenate_datasets
from torch.utils.data import ConcatDataset
import json
from css_load import CSS10
import codecs
import pandas as pd, os
import torchaudio, librosa

chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\»\«\´\'\́a-z]'


def load_css10(path):
    fpaths, text_lengths, texts = [], [], []
    transcript = os.path.join(path, 'transcript.txt')
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()
    _lines = list(lines)
    _folder_audio = "Paramythi_horis_onoma"

    datafr = pd.DataFrame(columns = ['path', 'audio', 'path_16k', 'sentence', 'sample_rate', 'duration', 'speaker_id', 'language'])
    datadicts = []
    index = 0
    for line in _lines:
        line = line.strip()
        wav_path, original_script, normalized_script, audio_duration = line.strip().split('|')
        speaker_id = wav_path.split('/')[0]
        wav_path_22k = os.path.join(path,  wav_path)
        #waveform, sample_rate = librosa.load(wav_path_22k)
        if float(audio_duration) > 8.:
            continue
        with open(wav_path_22k, 'rb') as f:
            data = {
                'audio':{
                    'path': wav_path_22k,
                    'bytes': f.read()
                },
                'path': wav_path_22k,
                #'audio': waveform,
                'path_16k': None,
                'sentence': normalized_script,
                'sample_rate': 22050,
                'duration': audio_duration,
                'speaker_id': speaker_id,
                'language': 'el',
            }
        datadicts.append(data)
        index +=1
        if index>10000:
            break


    return pd.DataFrame.from_records(datadicts)


def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower() + " "
    return batch

def replace_hatted_characters(batch):
    batch["sentence"] = re.sub('[ά]', 'α', batch["sentence"])
    batch["sentence"] = re.sub('[έ]', 'ε', batch["sentence"])
    batch["sentence"] = re.sub('[ί]', 'ι', batch["sentence"])
    batch["sentence"] = re.sub('[ύ]', 'υ', batch["sentence"])
    batch["sentence"] = re.sub('[ϊ]', 'ι', batch["sentence"])
    batch["sentence"] = re.sub('[ό]', 'ο', batch["sentence"])
    batch["sentence"] = re.sub('[ϋ]', 'υ', batch["sentence"])
    batch["sentence"] = re.sub('[ή]', 'η', batch["sentence"])
    batch["sentence"] = re.sub('[ώ]', 'ω', batch["sentence"])
    batch["sentence"] = re.sub('  ', ' ', batch["sentence"])


    return batch

def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))

  return {"vocab": [vocab], "all_text": [all_text]}




def load_data(path = 'archive/el'):
    common_voice_train = load_dataset("common_voice", 'el', split="train+validation")
    common_voice_test = load_dataset("common_voice", 'el',  split="test")

    common_voice_train = common_voice_train.remove_columns(
        ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    common_voice_test = common_voice_test.remove_columns(
        ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])


    css_set = load_css10(path)

    css_train = Dataset.from_pandas(css_set)
    css_train = css_train.remove_columns(['path_16k', 'duration', 'sample_rate', 'speaker_id', 'language'])


    # remove special characters like question marks
    common_voice_train = common_voice_train.map(remove_special_characters)
    common_voice_test = common_voice_test.map(remove_special_characters)
    css_train = css_train.map(remove_special_characters)

    print(common_voice_train.info)

    css_train = css_train.cast_column("audio", Audio(sampling_rate=16_000))
    common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16_000))
    common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16_000))

    # extract the vocabulary characters when using the CTC loss
    vocab_train_css = css_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                                         remove_columns=common_voice_train.column_names)
    vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                                         remove_columns=common_voice_train.column_names)
    vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                                       remove_columns=common_voice_test.column_names)

    # keep the intersection of characters
    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]) | set(vocab_train_css['vocab'][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    # some unknown and padding characters
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    return concatenate_datasets([common_voice_train, css_train]), \
           common_voice_test, \
           vocab_dict


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


if __name__ == '__main__':
    ds, _, _ = load_data()
