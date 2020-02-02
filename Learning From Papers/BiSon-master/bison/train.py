# coding=utf-8
#        BiSon

"""
Handles training of BiSon models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from tqdm import tqdm, trange

from .predict import predict
from .model_helper import save_model, prepare_train_data_loader, prepare_optimizer

LOGGER = logging.getLogger(__name__)


def get_loss(model, batch):
    """
    Given a batch, gets the loss for the chosen BERT model.

    :param bison_args: an instance of :py:class:BisonArguments
    :param model: a BERT model
    :param batch: the current batch
    :return: the loss of the current batch for the chosen model.
    """
    input_ids, input_mask, segment_ids, gen_label_ids, _ = batch

    # Get loss
    loss = model(input_ids, segment_ids, input_mask, gen_label_ids)
    return loss


def train(bison_args, data_handler, data_handler_predict, model, masker, tokenizer, device):
    """
    Runs training for a model.

    :param bison_args: Instance of :py:class:BisonArguments
    :param data_handler: instance or subclass of :py:class:Bitext, for training
    :param data_handler_predict: instance or subclass of :py:class:Bitext, for validation
    :param model: the model that will be trained
    :param masker: subclass instance of :py:class:Masking
    :param tokenizer: instance of BertTokenzier
    :param device: the device to run the computation on
    :return: 0 on success
    """
    train_examples = data_handler.examples
    num_train_steps = \
        int(len(train_examples) / bison_args.train_batch_size /
            bison_args.gradient_accumulation_steps * bison_args.num_train_epochs)

    optimizer, t_total = prepare_optimizer(bison_args, model, num_train_steps)

    train_dataloader = prepare_train_data_loader(bison_args, masker, data_handler, tokenizer)

    best_valid_score = 0.0  # if validation is run during training, keep track of best

    model.train()
    n_params = sum([p.nelement() for p in model.parameters()])
    LOGGER.info("Number of parameters: %d", n_params)

    for epoch in trange(int(bison_args.num_train_epochs), desc="Epoch"):
        LOGGER.info("Starting Epoch %s:", epoch)

        # some masking changes at every epoch, thus reload if necessary
        if bison_args.masking_strategy is not None and epoch != 0:  # already done for first epoch
            LOGGER.info("Recreating masks")
            train_dataloader = prepare_train_data_loader(bison_args,
                                                         masker,
                                                         data_handler,
                                                         tokenizer)

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(instance.to(device) for instance in batch)  # move batch to gpu

            loss = get_loss(model, batch)  # access via loss.item()

            if bison_args.gradient_accumulation_steps > 1:
                loss = loss / bison_args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % bison_args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Validate on the dev set if desired.
        if bison_args.valid_every_epoch:
            best_valid_score = validate(best_valid_score, bison_args, data_handler_predict, masker,
                                        tokenizer, model, device, epoch)

    # save last model if we didn't pick the best during training
    if not bison_args.valid_every_epoch:
        LOGGER.info("Saving final model")
        save_model(bison_args, model)

    return best_valid_score


def validate(best_valid_score, bison_args, data_handler_predict, masker, tokenizer, model, device,
             epoch):
    """
    After an epoch of training, validate on the validation set.

    :param best_valid_score: the currently best validation score
    :param bison_args: an instance of :py:class:BisonArguments
    :param data_handler_predict: instance or subclass instance of :py:class:Bitext,
     on which to run prediction
    :param masker: an instance of a subclass of :py:class:Masking
    :param tokenizer: the BERT tokenizer
    :param model: the BERT model
    :param device: where to run computations
    :param epoch: the current epoch
    :return: the new best validation score
    """
    model.eval()
    if best_valid_score == 0.0:  # then first epoch, save model
        save_model(bison_args, model)
    deciding_score = predict(bison_args, data_handler_predict, masker, tokenizer, model,
                             device, epoch)
    if best_valid_score < deciding_score:
        LOGGER.info("Epoch %s: Saving new best model: %s vs. previous %s",
                    epoch, deciding_score, best_valid_score)
        save_model(bison_args, model)
        best_valid_score = deciding_score
    model.train()
    return best_valid_score
