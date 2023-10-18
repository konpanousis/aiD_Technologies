# A python script to finetune the XLSR model towards Automatic Speech Recognition without using an LM.
# Developed in the context of the aiD project
import torch
import itertools

from datasets import load_metric
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from data_util_comb import DataCollatorCTCWithPadding, load_data
import numpy as np
from transformers import Wav2Vec2ForCTC, TrainingArguments, Trainer
import random

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

    print('GPU Available:', torch.cuda.is_available())
    path = '../archive/el'
    common_voice_train, common_voice_test, vocab_dict = load_data(path)

    train_flag = True
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]",
                                     word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


    def prepare_dataset(batch):
        audio = batch["audio"]

        # batched output is "un-batched"
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch


    rand_int = random.randint(0, len(common_voice_train) - 1)
    print("Target text:", common_voice_train[rand_int]["sentence"])
    print("Input array shape:", common_voice_train[rand_int]["audio"]["array"].shape)
    print("Sampling rate:", common_voice_train[rand_int]["audio"]["sampling_rate"])

    common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names)
    common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names)

    best_eval_acc = 0.
    lrs = [ 9e-5, 8e-5]
    wds = [0.01]
    layerdrop = [0.1]
    attention_dropout = [0.1]
    hidden_dropout = [0.]

    hyperparams = [lrs, wds, layerdrop, attention_dropout, hidden_dropout]
    hyperparams = list(itertools.product(*hyperparams))

    for lr, wd, ldrop, adrop, hdrop in hyperparams:
        output_dir = './combined_datasets_final/lr_{}_wd_{}_ldrop_{}_adrop_{}_hdrop_{}/'.format(str(lr), str(wd), str(ldrop),
                                                                                      str(adrop), str(hdrop))

        if train_flag:
            training_args = TrainingArguments(
                output_dir=output_dir,
                group_by_length=True,
                per_device_train_batch_size=32,
                gradient_accumulation_steps=2,
                evaluation_strategy="steps",
                num_train_epochs=200,
                gradient_checkpointing=True,
                fp16=True,
                save_steps=400,
                eval_steps=400,
                logging_steps=400,
                learning_rate=lr,
                warmup_steps=500,
                save_total_limit=3,
                push_to_hub=False,
                logging_dir = output_dir
            )

            data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

            model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-xls-r-300m",
                attention_dropout=adrop,
                hidden_dropout=hdrop,
                feat_proj_dropout=0.0,
                mask_time_prob=0.05,
                layerdrop=ldrop,
                ctc_loss_reduction="mean",
                pad_token_id=processor.tokenizer.pad_token_id,
                vocab_size=len(processor.tokenizer),
            )

            model.freeze_feature_encoder()

            train(model, common_voice_train, common_voice_test, output_dir)

            del data_collator, model
        else:
            pass
