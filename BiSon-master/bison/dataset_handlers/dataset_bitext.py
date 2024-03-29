# coding=utf-8
#        BiSon
"""
Handles any bitext, where input (Part A) and output (Part B) are separated by a tab.
"""

import logging
import re
import subprocess
import os

from bison.util import write_list_to_file, write_json_to_file, read_lines_in_list

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

LOGGER = logging.getLogger(__name__)


class GenExample():
    """
    This class encodes the bare minimum an instance needs to specify for a BERT model to run on it.
    """
    def __init__(self):
        """
        A single set of data. Subclasses should overwrite as appropriate.
        :param part_a: text string for Part A
        :param part_b: text string for Part B
        """
        self.part_a = None
        self.part_b = None


class BitextHandler():
    """
    Base class for bitext data sets.
    Other classes should inherit from this.

    A subclass should have the following variables
    examples: a list of examples, example should be dataset specific, for a very generic version see
                :py:class:GenExample
    features: a list of features, where a feature at index i maps to the example at index i
                in examples list
                See :py:class:GenInputFeature for an example.
    write_predictions: How to write predictions, either write_list_to_file or write_json_to_file
    write_eval: How to write evaluations, either write_list_to_file or write_json_to_file
    """
    def __init__(self):
        """
        examples: a list of examples of type :py:class:BitextExample
        features: a list of features of type :py:class:GenInputFeature
        """
        self.examples = []
        self.features = []
        self.write_predictions = write_list_to_file
        self.write_eval = write_json_to_file
        # if True, convert_examples_to_features in masking.py will truncate the end
        # if it exceeds max_part_a
        self.truncate_end = True

    def read_examples(self, input_file, is_training=False):
        """
        Reads a bitext that is separated by a tab, e.g. word1 word2 \t word3 word4
        Everything before the tab will become Part A, rest Part B
        :param input_file: the file containing the tab-separated data for training,
        and just Part A for predict
        :param is_training: True for training, then we expect \t, else we do not.
        :return: 0 on success
        """
        self.examples = []  # reset previous lot
        LOGGER.info("Part a: prior to tab")
        LOGGER.info("Part b: post tab")
        all_data = read_lines_in_list(input_file)

        example_counter = 0
        for instance in all_data:
            part_a = instance
            part_b = ""
            if is_training is True:
                split_line = instance.split("\t")
                assert len(split_line) == 2
                part_a = split_line[0]
                part_b = split_line[1]

            example = BitextHandler.BitextExample(
                example_index=example_counter,
                part_a=part_a,
                part_b=part_b)
            self.examples.append(example)
            example_counter += 1
        return 0

    # pylint: disable=no-self-use
    def arrange_generated_output(self, current_example, generated_text):
        """
        Simply returns generated_text, other data sets can arrange the output
        ready for evaluation here.
        :param current_example: The current example
        :param generated_text: The text generated by the model
        :return: generated_text
        """
        del current_example
        return generated_text

    def evaluate(self, output_prediction_file, valid_gold, mode='generation'):
        """
        Given the location of the prediction and gold output file,
        calls a dataset specific evaluation script.
        Here it calls case-sensitive BLEU script and F1 word match.
        :param output_prediction_file: the file location of the predictions
        :param valid_gold: the file location of the gold outputs
        :param mode: possible values: generation
        :return: a dictionary with various statistics
        """
        def _convert_to_float(convert):
            try:
                convert = float(convert)
            except OverflowError:
                convert = 0.0
            return convert

        # BLEU
        with open(output_prediction_file, "r") as file:
            eval_process = \
                subprocess.Popen([DIR_PATH+"/../evals/multi-bleu.perl", "-lc", valid_gold],
                                 stdin=file, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = eval_process.communicate()
        #format example:
        # BLEU = 26.27, 59.8/38.8/32.5/27.1 (BP=0.695, ratio=0.733, hyp_len=4933, ref_len=6729)
        bleu_all = stdout.decode("utf-8")
        if bleu_all.startswith("Illegal division"):
            results = {mode+"_"+'moses_bleu': 0.0, mode+"_"+'moses_bleu_1': 0.0,
                       mode+"_"+'moses_bleu_2': 0.0, mode+"_"+'moses_bleu_3': 0.0,
                       mode+"_"+'moses_bleu_4': 0.0}
        else:
            bleu = 0.0
            try:
                bleu = float(re.compile('BLEU = (.*?),').findall(bleu_all)[0])
            except OverflowError:  # if all translations are the empty string
                pass

            bleu_all = re.sub(r".*?, ", '', bleu_all, 1)
            bleu_all = re.sub(r" .BP.*\n", '', bleu_all)
            #format now: 159.8/38.8/32.5/27.1
            bleu_all = bleu_all.split("/")
            try:
                results = {mode+"_"+'moses_bleu': bleu,
                           mode+"_"+'moses_bleu_1': _convert_to_float(bleu_all[0]),
                           mode+"_"+'moses_bleu_2': _convert_to_float(bleu_all[1]),
                           mode+"_"+'moses_bleu_3': _convert_to_float(bleu_all[2]),
                           mode+"_"+'moses_bleu_4': _convert_to_float(bleu_all[3])}
            except OverflowError:
                results = {mode+"_"+'moses_bleu': 0.0, mode+"_"+'moses_bleu_1': 0.0,
                           mode+"_"+'moses_bleu_2': 0.0, mode+"_"+'moses_bleu_3': 0.0,
                           mode+"_"+'moses_bleu_4': 0.0}

        return results

    def select_deciding_score(self, results):
        """
        Returns the score that should be used to decide whether or not
        a model is best compared to a previous score.
        Here we return BLEU-4
        :param results: what is returned by the method evaluate,
        a dictionary that should contain 'bleu_4'
        :return: BLEU-4 value
        """
        return results['generation_moses_bleu_4']

    class BitextExample(GenExample):
        """A single training/test example from src.Bitext.
        """

        # pylint: disable=too-few-public-methods
        def __init__(self,
                     example_index,
                     part_a,
                     part_b):
            super().__init__()
            self.example_index = example_index
            self.part_a = part_a
            self.part_b = part_b
