# coding=utf-8
#        BiSon
"""
Implements various maskers for BiSon.
"""

import random
import logging
import numpy as np

LOGGER = logging.getLogger(__name__)


class GenInputFeatures:
    """Features for one data point."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 gen_label_ids):
        """
        General possible structure of an input sentence:
        [CLS] Part A [SEP] Part B [SEP] <Padding until max_seq_length>
        :param input_ids: contains the vocabulary id for each unmasked token,
        masked tokens receive the value of [MASK]
        :param input_mask: 1 prior to padding, 0 for padding
        :param segment_ids: 0 for Part A, 1 for Part B, 0 for padding.
        :param gen_label_ids: -1 for unmasked tokens, vocabulary id for masked tokens
        """
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.gen_label_ids = gen_label_ids


def get_masker(bison_args):
    """
    Factory for returning a masker.
    :param bison_args: an instance of :py:class:BisonArguments
    :return: an instance of a subclass of :py:class:Masking
    """
    masker = None
    if bison_args.masking == 'gen':
        masker = GenerationMasking(bison_args)
    else:
        LOGGER.error("Unknown masking name: %s", bison_args.masking)
        exit(1)
    return masker


class Masking:
    """
    Superclass for maskers.
    They take a data_handler, subclass instance of :py:class:DatasetHandler,
    and convert the elements of data_handler.examples into a set of features,
    which are stored in data_handler.features.

    Same index indicates same example/feature.

    data_handler.examples is a list of subclass instances of :py:class:GenExample
    data_hanlder.examples ist a list of instances of :py:class:GenInputFeatures

    Subclasses should implement handle_masking, should call this class's init in its own.
    convert_examples_to_features should stay the same.
    """
    def __init__(self, bison_args):
        """
        Keeps track of some statistics.

        violate_max_part_a_len: how often in data_handler, the maximum query
        length (Part A) was violated
        violate_max_gen_len: how often in data_handler, the maximum generation
        length (Part B) was violated
        trunc_part_b: how often Part B was truncated
        trunc_part_a: how often Part A was truncated
        max_gen_length: the maximum generation length
        max_part_a: the maximum length of part a

        """
        self.violate_max_part_a_len = 0
        self.violate_max_gen_len = 0
        self.trunc_part_b = 0
        self.trunc_part_a = 0
        self.max_gen_length = bison_args.max_gen_length
        self.max_part_a = bison_args.max_part_a

    def handle_masking(self, part_a, part_b, is_training, max_seq_length, tokenizer, example_index):
        """
        Convert a part_a and a part_b into 4 lists needed to instantiate :py:class:GenInputFeatures
        :param part_a: a string of text of Part A, i.e. part_a of a subclass instance of
        :py:class:GenExample
        :param part_b: a string of text of Part B, i.e. part_b of a subclass instance of
        :py:class:GenExample
        :param is_training: true if training, part_b is only considered for training
        :param max_seq_length: the maximum sequence length (Part A + Part B)
        :param tokenizer: an instance of :py:class: BertTokenizer
        :param example_index: the index of the current sample, e.g. i when iterating over
        data_handler.examples[i]
        :return: a 4-tuple of lists, each with length max_seq_length
            input_ids: ids of "[cls] part a [sep] part b [sep]" or a masking thereof
            input_mask: 1 for all spots that should be attended to
            segment_ids: 0 up to and including the first [sep], 1 until second [sep] or for
            remainder of sequence
            gen_label_ids: -1 for positions in input_ids that should not be predicted,
                           the id of the to-be-predicted token,
                           should be always -1 at test time
        """
        raise NotImplementedError

    def convert_examples_to_features(self, data_handler, tokenizer, max_seq_length, max_part_a,
                                     is_training):
        """
        From a list of examples (subclass instances of :py:class:GenExample),
        creates a list of instances of :py:class:GenInputFeatures
        :param data_handler: a subclass instance of :py:class:DatasetHandler;
                will access data_handler.examples (list of subclass instances of
                :py:class:GenExample)
                and will set data_handler.features (list of instances of :py:class:GenInputFeatures)
        :param tokenizer: an instance of :py:class: BertTokenizer
        :param max_seq_length: the maximum sequence length ([CLS] + Part A + [SEP] + Part B + [SEP])
        :param max_part_a: the maximum length of Part A
        :param is_training: true if training, handles gold label construction
        :return:0 on success
        """
        data_handler.features = []
        max_a = 0
        max_b = 0

        # iterate over subclass instances of :py:class:GenExample
        for i, instance in enumerate(data_handler.examples):
            # Part A
            part_a = tokenizer.tokenize(instance.part_a)
            max_a = max(max_a, len(part_a))
            if len(part_a) > max_part_a:
                if data_handler.truncate_end:
                    part_a = part_a[0:max_part_a]
                    self.trunc_part_a += 1
                else:  # truncate beginning
                    # +2 because we save space for [CLS] and [SEP]
                    first_trunc_index = len(part_a) - max_part_a + 2
                    part_a = part_a[first_trunc_index:]
                    self.trunc_part_a += 1

            # Part B
            part_b = tokenizer.tokenize(instance.part_b)
            max_b = max(max_b, len(part_b))

            # Masking for one instance, handled by subclass of :py:class:Masker
            input_ids, input_mask, segment_ids, gen_label_ids = \
                self.handle_masking(part_a, part_b, is_training, max_seq_length, tokenizer, i)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(gen_label_ids) == max_seq_length

            feature = GenInputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                gen_label_ids=gen_label_ids)
            data_handler.features.append(feature)

        # Every instance has exactly one corresponding features at the same index
        assert len(data_handler.examples) == len(data_handler.features)
        LOGGER.info("Maximum Part A is: %s", max_a)
        LOGGER.info("Maximum Part B is: %s", max_b)
        LOGGER.warning("Couldn't encode query length %s times.", self.violate_max_part_a_len)
        LOGGER.warning("Couldn't encode generation length %s times.", self.violate_max_gen_len)
        LOGGER.warning("Truncated part b %s times.", self.trunc_part_b)
        LOGGER.warning("Truncated part a %s times.", self.trunc_part_a)
        return 0


class GenerationMasking(Masking):
    """
    [CLS] Part A [SEP] [Mask] * max_gen_length <Padding>
    Introduces max_gen_length to keep the maximum generation length fixed across examples.

    input_ids: vocabulary id up to first [SEP], then [MASK] ID until max_gen_length
    input_mask: 1 for every position until max_gen_length
    segment_ids: 0 for Part A and first [SEP], 1 until max_gen_length
    gen_label_ids: -1 first [SEP], vocabulary id for actually existing masked tokens,
    including second [SEP]
                   then -1 until max_gen_length
                   at test time: always -1 until max_gen_length
    """
    def __init__(self, bison_args):
        """
        Masking scheme for generation.

        :param bison_args: instance of :py:class:GeneralArguments
        """
        super(GenerationMasking, self).__init__(bison_args)
        self.max_gen_length = bison_args.max_gen_length
        self.max_part_a = bison_args.max_part_a

        self.masking_strategy = ""
        if bison_args.masking_strategy is not None:
            self.do_percentage_per_example = True
            self.masking_strategy = bison_args.masking_strategy
            LOGGER.info("Using %s sampling for masking threshold.", bison_args.masking_strategy)
        else:
            LOGGER.info("Not using a percentage list.")
            self.do_percentage_per_example = False
            self.masking_strategy = 'all'
        self.mean = bison_args.distribution_mean
        self.stdev = bison_args.distribution_stdev
        LOGGER.info("Mean: %s, Variance: %s", self.mean, self.stdev)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "GenerationMasking"

    def create_mask(self, len_mask_list):
        """
        Given a length, it uses the specified masking strategy to create a corresponding
        masking list.

        :param len_mask_list: the length the masking list will have to be.
        :return: the masking list, where it is 1.0 if a mask should be placed in that position
        """
        mask_list = [0.0] * len_mask_list
        if self.masking_strategy == 'bernoulli':
            #1.0 means mask
            for i, _ in enumerate(mask_list):
                sample = random.random()
                if sample < self.mean:
                    mask_list[i] = 1.0
        elif self.masking_strategy == 'gaussian':
            current_threshold = np.random.normal(self.mean, self.stdev)
            nr_masks = int(round(current_threshold * len_mask_list))
            mask_list = [1.0] * nr_masks + [0.0] * (len_mask_list - nr_masks)
            random.shuffle(mask_list)
        return mask_list

    def handle_masking(self, part_a, part_b, is_training, max_seq_length, tokenizer,
                       example_index=-1):
        """
        Given a part_a and a part_b, performs masking.
        If is_training is False, everything is masked in part_b.

        :param part_a: Part A as taken from a subclass instance of :py:class:GenExample
        :param part_b: Part B as taken from a subclass instance of :py:class:GenExample
        :param is_training: Set True for training, else false
        :param max_seq_length: the maximum sequence length ([CLS] Part A [SEP] Part B [SEP])
        :param tokenizer: the tokenizer to use
        :param example_index: the index of the current example, use for debugging only
        :return: a 4-tuple of lists, all of length max_seq_length:
                1. input_ids: a list of word IDs, Part A is whole, Part B can contain [MASK] IDs
                2. input_mask: 1 until second [SEP], 0 rest
                3. segment_ids: 0 until first [SEP], 1 until second [SEP], then 0
                4. gen_label_ids: -1 at test time, correct word IDs where input_ids has [MASK] ID
        """
        # sample from the percentage list for every example
        mask_list = None
        if is_training is True:
            mask_list = self.create_mask((len(part_b) + 1))
        tokens = []
        segment_ids = []

        # Part A
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in part_a:
            tokens.append(token)
            segment_ids.append(0)
            if len(tokens) == self.max_part_a - 1:  # save space for [SEP]
                LOGGER.debug("Can't encode the maximum Part A length of example number %s",
                             example_index)
                self.violate_max_part_a_len += 1
                break
        tokens.append("[SEP]")
        segment_ids.append(0)

        part_b_index = len(tokens)
        max_gen_index = len(tokens) + self.max_gen_length - 1

        if not is_training:
            assert not part_b

        # Part B: Assembles [MASK]
        for i in range(self.max_gen_length):
            # i < len(part_b) ensures that no non-mask is done at test time,
            # since len(part_b) == 0 at test time
            # (see assert above)

            # we always mask the second [sep] token, else change len(part_b) to len_part_b_and_sep
            if self.do_percentage_per_example is True and i < len(part_b):
                #prob = random.random()  #uncomment to reproduce EMNLP results

                # only mask if the probability falls on masking
                if mask_list is not None:
                    if mask_list[i] == 1.0:
                        tokens.append('[MASK]')
                    else:
                        tokens.append(part_b[i])
                else:
                    tokens.append(part_b[i])
            else:
                tokens.append('[MASK]')
            segment_ids.append(1)
            if len(tokens) == max_seq_length:
                LOGGER.debug("Can't encode the maximum generation length of example number %s",
                             example_index)
                self.violate_max_gen_len += 1
                break

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        gen_label_ids = [-1] * len(input_ids)

        # Pad to maximum sequence length
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            gen_label_ids.append(-1)

        # Fill gen_label_ids correctly if is_training is True
        if is_training is True:
            for token in part_b:
                gen_label_ids[part_b_index] = tokenizer.vocab[token]
                part_b_index += 1
                if part_b_index == max_gen_index:  # truncate
                    LOGGER.debug("Warning: Truncated Part b of example number %s",
                                 example_index)
                    self.trunc_part_b += 1
                    break
            gen_label_ids[part_b_index] = tokenizer.vocab["[SEP]"]  # [SEP] is always masked

        return input_ids, input_mask, segment_ids, gen_label_ids
