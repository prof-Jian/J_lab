# coding=utf-8
#        BiSon
"""
Implements the handler for Daily Dialog dataset (http://yanran.li/dailydialog.html)
"""

import logging
import os

from bison.util import write_list_to_file, write_json_to_file, read_lines_in_list
from .dataset_bitext import BitextHandler, GenExample

LOGGER = logging.getLogger(__name__)


class DailyDialogHandler(BitextHandler):
    """
    Inherits from src.textHandler to get the functions:
    arrange_generated_output, evaluate, select_deciding_score

    Handles the Daily Dialog dataset (http://yanran.li/dailydialog.html)
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, classify_type='act'):
        super().__init__()
        self.examples = []
        self.features = []
        self.write_predictions = write_list_to_file
        self.write_eval = write_json_to_file
        self.truncate_end = False

    def evaluate(self, output_prediction_file, valid_gold, mode='generation'):
        """
        Given the location of the prediction and gold output file, calls the BLEU
        evaluation script to evaluate the generation output.

        Assumes that valid_gold points to a folder with 'dev.gold' the targets for generation.

        :param output_prediction_file: the file location of the predictions
        :param valid_gold: the folder location (see above)
        :param mode: implements generation
        :return: BLEU scores
        """
        results = super(DailyDialogHandler, self).evaluate(output_prediction_file,
                                                           os.path.join(valid_gold, 'gold'),
                                                           mode=mode)
        return results

    # pylint: disable=too-many-locals
    def read_examples(self, input_file, is_training=False):
        """
        Reads a DailyDialog dataset, each instance in self.examples holds a
        :py:class:DailyDialogExample object
        We assume a tsv file of: dialogues_act_*.txt  dialogues_emotion_*.txt  dialogues_*.txt
        :param input_file: the file containing DailyDialog data
        :param is_training: True for training, then we read in gold labels, else we do not.
        :return: 0 on success
        """
        self.examples = []  # reset previous lot
        LOGGER.info("Part a: history")
        LOGGER.info("Part b: last utterance")
        all_data = read_lines_in_list(input_file)

        example_counter = 0

        for instance in all_data:
            topic = ""  # not handled atm
            next_utterance = ""

            split_tsv = instance.split("\t")
            assert len(split_tsv) == 3
            # -2 because the last position is empty
            act = int(split_tsv[0].split(" ")[-2]) - 1  # minus 1 because 0 indexed
            emotion = int(split_tsv[1].split(" ")[-2])
            dialogue = split_tsv[2]

            dialogue = dialogue.replace("__eou__", "\t")
            split_dialogue = dialogue.split("\t")
            # the last one is always empty because dialogue finished with __eou__
            split_dialogue.pop()

            if is_training is True:
                next_utterance = split_dialogue[-1].strip()
                next_utterance = next_utterance.replace("  ", " ")

            split_dialogue.pop()  # remove last utterance, which is the one we are trying to predict

            history = "".join(split_dialogue).strip()
            history = history.replace("  ", " ")

            example = DailyDialogHandler.DailyDialogExample(
                example_index=example_counter,
                example_counter=example_counter,
                topic=topic,
                act=act,
                emotion=emotion,
                history=history,
                next_utterance=next_utterance)
            self.examples.append(example)
            example_counter += 1

        return 0

    class DailyDialogExample(GenExample):
        """A single training/test example from src.e Daily Dialog corpus.
        """
        # pylint: disable=too-many-arguments
        # pylint: disable=too-few-public-methods
        def __init__(self,
                     example_index,
                     example_counter,
                     topic,
                     act,
                     emotion,
                     history,
                     next_utterance):
            super().__init__()
            self.example_index = example_index
            self.example_counter = example_counter
            self.topic = topic
            self.act = act
            self.emotion = emotion
            self.history = history
            self.next_utterance = next_utterance

            self.part_a = self.history
            self.part_b = self.next_utterance
