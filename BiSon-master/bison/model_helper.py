# coding=utf-8
#        BiSon
"""
Implements methods that help with handling the BERT models.
Note: Usage of
        # pylint: disable=not-callable
        # pylint: disable=no-member
to remove pytorch error warnings that ocur pre 1.0.1 where accessing torch causes these issues.
"""

import logging
import os
import re
import random

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from pytorch_pretrained_bert.modeling import BertForMaskedLM, BertConfig

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

LOGGER = logging.getLogger(__name__)


def save_model(bison_args, model, prefix=None):
    """
    Saves a model.

    :param bison_args: instance of :py:class:BisonArguments
    :param model: the model to save
    :param prefix: the prefix to attach to the file name
    :return: the location of the output file
    """
    model_to_save = model.module if hasattr(model, 'module') else model
    # Only save the model it-self
    if prefix:
        output_model_file = os.path.join(bison_args.output_dir, "%s.pytorch_model.bin" % prefix)
    else:
        output_model_file = os.path.join(bison_args.output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)
    return output_model_file


def set_seed(seed):
    """
    Sets the seed.

    :param seed: seed to set, set -1 to draw a random number
    :return: 0 on success
    """
    if seed == -1:
        seed = random.randrange(2**32 - 1)
    LOGGER.info("Seed: %s", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return 0


def create_tensor_dataset(data_handler):
    """
    Using a data_handler, whose features have been filled via the function
    convert_examples_to_features from a subclass instance of :py:class:Masking,
    convert the features into a TensorDataset
    :param data_handler: instance or subclass instance of :py:class:Bitext
    :return: the features represented as a TensorDataset
    """
    # pylint: disable=not-callable
    # pylint: disable=no-member
    all_input_ids = torch.tensor([f.input_ids for f in data_handler.features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in data_handler.features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in data_handler.features], dtype=torch.long)
    all_gen_label_ids = torch.tensor([f.gen_label_ids for f in data_handler.features],
                                     dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    data_set = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                             all_gen_label_ids, all_example_index)
    return data_set


def get_tokenizer(bison_args):
    """
    Based on the command line arguments, gets the correct Bert tokenizer.

    :param bison_args: an instance of :py:class:BisonArguments
    :return: a Bert tokenizer
    """
    if bison_args.bert_tokenizer is None:  # then tokenizer is the same name as the bert model
        if bison_args.bert_model == 'bert-vanilla':
            raise ValueError('For the bert-vanilla model, '
                             'a seperate bert_tokenizer must be specified.')
        tokenizer = BertTokenizer.from_pretrained(bison_args.bert_model,
                                                  do_lower_case=bison_args.do_lower_case)
    else:  # then a separate tokenizer has been specified
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",
                                                  do_lower_case=bison_args.do_lower_case)
    return tokenizer


def prepare_train_data_loader(bison_args, masker, data_handler, tokenizer):
    """
    Prepares the TensorDataset for training.

    :param bison_args: instance of :py:class:BisonArguments
    :param masker: the masker which will mask the data as appropriate, an instance of a subclass of
    :py:class:Masking
    :param data_handler: the dataset handler, an instance of :py:class:BitextHandler or a subclass
    :param tokenizer: the BERT tokenizer
    :return: train_dataloader, an instance of :py:class:TensorDataset
    """
    masker.convert_examples_to_features(
        data_handler=data_handler,
        tokenizer=tokenizer,
        max_seq_length=bison_args.max_seq_length,
        max_part_a=bison_args.max_part_a,
        is_training=True)

    train_data = create_tensor_dataset(data_handler)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=bison_args.train_batch_size)
    return train_dataloader


def load_model(bison_args, device, data_handler, output_model_file=None):
    """
    Load a model.

    :param bison_args: instance of :py:class:BisonArguments
    :param device: the device to move the model to
    :param data_handler: the dataset handler, an instance of :py:class:BitextHandler or a subclass
    :param output_model_file: the location of the model to load
    :return: the loaded model
    """

    model_state_dict = None
    if output_model_file is not None:
        model_state_dict = torch.load(output_model_file)

    if bison_args.bert_model == 'bert-vanilla':
        # randomly initialises BERT weights instead of using a pre-trained model
        model = BertForMaskedLM(BertConfig.from_default_settings())
    else:
        model = BertForMaskedLM.from_pretrained(bison_args.bert_model, state_dict=model_state_dict)
    model.to(device)
    return model


def argument_sanity_check(bison_args):
    """
    Performs a couple of additional sanity check on the provided arguments.

    :param bison_args: instance of :py:class:BisonArguments
    :return: 0 on success (else an error is raise)
    """
    if bison_args.do_train:
        if not bison_args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if bison_args.do_predict:
        if not bison_args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if os.path.exists(os.path.join(bison_args.output_dir, "pytorch_model.bin")) \
            and bison_args.do_train:
        if not bison_args.load_prev_model:
            raise ValueError("Output directory already contains a saved model (pytorch_model.bin).")
    os.makedirs(bison_args.output_dir, exist_ok=True)
    return 0


def prepare_optimizer(bison_args, model, num_train_steps):
    """
    Prepares the optimizer for training.
    :param bison_args: instance of :py:class:BisonArguments
    :param model: the model for which the optimizer will be created
    :param num_train_steps: the total number of training steps that will be performed
            (need for learning rate schedules that depend on this)
    :return: the optimizer and the number of total steps
    """
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    t_total = num_train_steps
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=bison_args.learning_rate,
                         warmup=0.1,
                         t_total=t_total)

    return optimizer, t_total
