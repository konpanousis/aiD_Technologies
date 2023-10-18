# A python script to finetune the XLSR model towards Automatic Speech Recognition using an LM.
# Developed in the context of the aiD project

from datasets import load_metric
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ProcessorWithLM, Wav2Vec2Processor
from data_util_comb import DataCollatorCTCWithPadding, load_data
import numpy as np
from transformers import Wav2Vec2ForCTC, TrainingArguments, Trainer,  AutoProcessor, AutoFeatureExtractor
import random
import torch
import itertools
from pyctcdecode import build_ctcdecoder
import torchaudio.functional as F

wer_metric = load_metric("wer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def train(model, train_set, test_set,path):
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=common_voice_train,
        eval_dataset=common_voice_test,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()


def evaluate(model, processor, test_aud, test_tr, load=False):
    if load:
        model = Wav2Vec2ForCTC.from_pretrained(model).to("cuda")
        processor = Wav2Vec2Processor.from_pretrained(processor)

    input_dict = processor(test_aud[0]['input_values'], return_tensors='pt', padding=True)
    logits = model(input_dict.input_values.to('cuda')).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]

    print("Prediction:")
    print(processor.decode(pred_ids))

    print("\nReference:")
    print(test_tr[0]["sentence"].lower())


if __name__ == '__main__':

    common_voice_train, common_voice_test, vocab_dict = load_data()

    train_flag = False
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]",
                                     word_delimiter_token="|")

    model =  Wav2Vec2ForCTC.from_pretrained("./combined_datasets/checkpoint-best", local_files_only=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
                "./combined_datasets/checkpoint-best", local_files_only=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained("./combined_datasets/checkpoint-best/")
    vocab_dict = tokenizer.get_vocab()

    sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
    
    unigrams = []
    with open('language_models/unigrams.txt', 'r') as f:
        for line in f:
            unigrams.append(line.strip().split("\n")[0])
            
    decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()),
        kenlm_model_path="language_models/4gram_cleaned.bin",
        unigrams = unigrams
        )
    
    processor_with_lm = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    

    rand_int = random.randint(0, len(common_voice_test) - 1)
    print("Target text:", common_voice_test[rand_int]["sentence"])
    print("Input array shape:", common_voice_test[rand_int]["audio"]["array"].shape)
    print("Sampling rate:", common_voice_test[rand_int]["audio"]["sampling_rate"])

    print(" ".join(sorted(processor.tokenizer.get_vocab())))
    

    for a in [0.2, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for b in [2.0, 3.0, 4.0]:
            wer = 0. 
            decoder.reset_params(alpha=a, beta=b)
            for sample in common_voice_test:
                #sample = common_voice_test[i]
                #print(sample)

                #    print(wer)
                print('new sample')
                print(sample['audio']['array'])
                target_text = sample["sentence"]

                input_values = processor_with_lm(sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"],  return_tensors="pt", padding = True)

                with torch.no_grad():
                    logits = model(**input_values).logits


                transcription = processor_with_lm.batch_decode(logits.numpy()).text

                print('pred text: ', transcription[0])
                print('pred text: ', transcription)

                wer += wer_metric.compute(references = [target_text], predictions = [transcription[0].lower()]) / len(common_voice_test)

            print((a,b,wer))
        
