# The inference script for the Sign Language Translation module.
# It loads all the necessary modules from the SLT package, and the main recognition function is ''recognition''.
# Developed in the context of the aiD project
import argparse
import os
import sys
import logging

from SLTModel.TranslationModel.signjoey.prediction import validate_on_data

sys.path.append(os.getcwd()+'/SLTModel/TranslationModel')

from signjoey.helpers import (
    bpe_postprocess,
    load_config,
    get_latest_checkpoint,
    load_checkpoint,
)
from signjoey.compression import compute_compression_rate,compute_reduced_weights
from signjoey.vocabulary import GlossVocabulary, TextVocabulary
from signjoey.model import build_model, SignModel
from signjoey.data import load_test_data, make_data_iter, read_gloss_vocab
from signjoey.vocabulary import PAD_TOKEN, SIL_TOKEN

from signjoey.loss import XentLoss
from signjoey.EnsembleTransformer import BuildEnsebleCKPT


mode = 'test'
config_path = os.getcwd()+'/SLTModel/TranslationModel/configs/PlayGround_alphapose.yaml'
ckpt = os.getcwd()+'/SLTModel/TranslationModel/PreTrained/best.ckpt'
output_path = 'results_SLT/'

##### LOGGING STUFF #######################
logger = logging.getLogger(__name__)
if not logger.handlers:
    FORMAT = "%(asctime)-15s - %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)

# load configuration
cfg = load_config(config_path)

# Ensemble or simple model
if type(cfg['training']['model_dir'])==type([]):
        ensemble=True
        ensembleN=len(cfg['training']['model_dir'])
        model_checkpoint=BuildEnsebleCKPT([get_latest_checkpoint(model_dir) for model_dir in cfg['training']['model_dir'] ])
        print('\n##########################\n')
        print('   Ensemble')
        print('  ',ensembleN,' Models')
        print('\n##########################\n')

else:
    ensemble=False
    ensembleN=1

# use gpu
use_cuda = cfg["training"].get("use_cuda", False)
translation_beam_sizes = cfg["testing"].get("translation_beam_sizes", [1])

# i don't know why is this necessary
if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")
if not ensemble:
    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)

        if ckpt is None:
            raise FileNotFoundError(
                "No checkpoint found in directory {}.".format(model_dir)
            )
    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

batch_size = cfg["training"]["batch_size"]
batch_type = cfg["training"].get("batch_type", "sentence")

level = cfg["data"]["level"]
dataset_version = cfg["data"].get("version", "phoenix_2014_trans")
translation_max_output_length = cfg["training"].get(
    "translation_max_output_length", None
)


gls_vocab= GlossVocabulary(file=cfg["data"].get("gls_vocab", None))
txt_vocab = TextVocabulary(file=cfg["data"].get("txt_vocab", None))
#print(txt_vocab)

# build model and load parameters into it
do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
model = build_model(
    cfg=cfg["model"],
    gls_vocab=gls_vocab,
    txt_vocab=txt_vocab,
    sgn_dim=sum(cfg["data"]["feature_size"])
    if isinstance(cfg["data"]["feature_size"], list)
    else cfg["data"]["feature_size"],
    do_recognition=do_recognition,
    do_translation=do_translation,
    ensemble=ensemble,
    ensembleN=ensembleN
)

model.load_state_dict(model_checkpoint["model_state"])

if use_cuda:
    model.cuda()


frame_subsampling_ratio = cfg["data"].get("frame_subsampling_ratio", None)

# whether to use beam search for decoding, 0: greedy decoding
if "testing" in cfg.keys():
    recognition_beam_sizes = cfg["testing"].get("recognition_beam_sizes", [1])
    translation_beam_sizes = cfg["testing"].get("translation_beam_sizes", [1])
    translation_beam_alphas = cfg["testing"].get("translation_beam_alphas", [-1])
else:
    recognition_beam_sizes = [1]
    translation_beam_sizes = [1]
    translation_beam_alphas = [-1]

if "testing" in cfg.keys():
    max_recognition_beam_size = cfg["testing"].get(
        "max_recognition_beam_size", None
    )
    if max_recognition_beam_size is not None:
        recognition_beam_sizes = list(range(1, max_recognition_beam_size + 1))

if do_recognition:
    recognition_loss_function = torch.nn.CTCLoss(
        blank=model.gls_vocab.stoi[SIL_TOKEN], zero_infinity=True
    )
    if use_cuda:
        recognition_loss_function.cuda()
if do_translation:
    translation_loss_function = XentLoss(
        pad_index=txt_vocab.stoi[PAD_TOKEN], smoothing=0.0
    )
    if use_cuda:
        translation_loss_function.cuda()


def recognition(test_path):
    dev_data = load_test_data(cfg["data"], test_path, gls_vocab, txt_vocab)

    if True:
        logger.info("=" * 60)
        dev_translation_results = {}
        dev_best_bleu_score = float("-inf")
        dev_best_translation_beam_size = 1
        dev_best_translation_alpha = 1
        for tbw in translation_beam_sizes:
            dev_translation_results[tbw] = {}
            for ta in translation_beam_alphas:
                dev_translation_results[tbw][ta] = validate_on_data(
                    model=model,
                    data=dev_data,
                    batch_size=batch_size,
                    use_cuda=use_cuda,
                    level=level,
                    sgn_dim=sum(cfg["data"]["feature_size"])
                    if isinstance(cfg["data"]["feature_size"], list)
                    else cfg["data"]["feature_size"],
                    batch_type=batch_type,
                    dataset_version=dataset_version,
                    do_recognition=do_recognition,
                    recognition_loss_function=recognition_loss_function
                    if do_recognition
                    else None,
                    recognition_loss_weight=1 if do_recognition else None,
                    recognition_beam_size=1 if do_recognition else None,
                    do_translation=do_translation,
                    translation_loss_function=translation_loss_function,
                    translation_loss_weight=1,
                    translation_max_output_length=translation_max_output_length,
                    txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
                    translation_beam_size=tbw,
                    translation_beam_alpha=ta,
                    frame_subsampling_ratio=frame_subsampling_ratio,
                )

                if (
                    dev_translation_results[tbw][ta]["valid_scores"]["bleu"]
                    >= dev_best_bleu_score
                ):
                    dev_best_bleu_score = dev_translation_results[tbw][ta][
                        "valid_scores"
                    ]["bleu"]
                    dev_best_translation_beam_size = tbw
                    dev_best_translation_alpha = ta
                    dev_best_translation_result = dev_translation_results[tbw][ta]
                    logger.info(
                        "[DEV] partition [Translation] results:\n\t"
                        "New Best Translation Beam Size: %d and Alpha: %d\n\t"
                        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
                        "CHRF %.2f\t"
                        "ROUGE %.2f",
                        dev_best_translation_beam_size,
                        dev_best_translation_alpha,
                        dev_best_translation_result["valid_scores"]["bleu"],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu1"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu2"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu3"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu4"
                        ],
                        dev_best_translation_result["valid_scores"]["chrf"],
                        dev_best_translation_result["valid_scores"]["rouge"],
                    )
                    logger.info("-" * 60)

    return dev_best_translation_result["txt_hyp"]

if __name__=='__main__':
    read_gloss_vocab(cfg['data'])
    load_video(cfg)





