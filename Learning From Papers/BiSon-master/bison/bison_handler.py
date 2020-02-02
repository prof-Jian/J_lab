# coding=utf-8
#        BiSon

"""
Entry point for running BiSon, can call training and prediction from here.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch

from .predict import predict
from .masking import get_masker
from .model_helper import load_model, argument_sanity_check, set_seed, get_tokenizer
from .dataset_handlers.datasets_factory import get_data_handler
from .train import train

LOGGER = logging.getLogger(__name__)


def bison_runner(bison_args):
    """
    Main function to run training or prediction for Bison.

    :param bison_args: instance of :py:class:BisonArguments
    :return: A tuple of:
            1. the best score of the validation set during training
            2. the best score after prediction
    """
    # Set up masker which decides what parts in the input will be masked
    masker = get_masker(bison_args)
    LOGGER.info("Masker: %s", masker)

    # Set up needed data handlers
    data_handler = None
    data_handler_predict = None
    if bison_args.do_train:
        data_handler = get_data_handler(bison_args)
        LOGGER.info("Data Handler for training: %s", data_handler)
    if bison_args.do_predict or bison_args.valid_every_epoch:
        data_handler_predict = get_data_handler(bison_args)
        LOGGER.info("Data Handler for prediction: %s", data_handler_predict)

    # Runs some sanity check on the argument, returns 0 on success, else raises error
    argument_sanity_check(bison_args)

    # Set up where computations will be run (gpu vs cpu, number of gpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bison_args.train_batch_size = int(bison_args.train_batch_size /
                                      bison_args.gradient_accumulation_steps)

    # Set seed
    set_seed(bison_args.seed)

    # Set up tokenizer
    tokenizer = get_tokenizer(bison_args)

    # Training
    deciding_score_train = -1
    if bison_args.do_train:
        # Prepare model
        model = load_model(bison_args, device, data_handler,
                           output_model_file=bison_args.load_prev_model)

        # Move model
        model.to(device)

        data_handler.read_examples(input_file=bison_args.train_file, is_training=True)
        deciding_score_train = train(bison_args, data_handler, data_handler_predict, model, masker,
                                     tokenizer, device)

    # Prediction
    deciding_score = -1
    if bison_args.do_predict and (bison_args.local_rank == -1 or torch.distributed.get_rank() == 0):
        LOGGER.info("bison_args.output_dir: %s", bison_args.output_dir)
        output_model_file = os.path.join(bison_args.output_dir, "pytorch_model.bin")
        model = load_model(bison_args, device, data_handler_predict,
                           output_model_file=output_model_file)
        model.to(device)
        model.eval()
        deciding_score = predict(bison_args, data_handler_predict, masker, tokenizer, model, device)

    return deciding_score_train, deciding_score
