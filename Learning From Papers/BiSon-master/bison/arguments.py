# coding=utf-8
#        BiSon

"""
Handles the possible command line arguments for both BiSon and GPT2.
"""

import argparse


class GeneralArguments():
    """
    Settings relevant for every training and prediction scenario
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._add_arguments()
        bison_args = vars(self.parser.parse_args())
        for key in bison_args:
            setattr(self, key, bison_args[key])

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.__class__.__name__

    def _add_arguments(self):
        # Required parameters
        self.parser.add_argument("--output_dir", default=None, type=str, required=True,
                                 help="The output directory where all relevant files will"
                                      "be written to.")

        self.parser.add_argument('--valid_every_epoch',
                                 action='store_true',
                                 help="Whether to validate on the validation set after every "
                                      "epoch, save best model according to the evaluation metric "
                                      "indicated by each specific dataset class.")
        self.parser.add_argument("--load_prev_model", default=None, type=str,
                                 help="Provide a file location if a previous model should be "
                                      "loaded. (Note that Adam Optimizer paramters are lost.")

        ## Other parameters
        self.parser.add_argument("--data_set", type=str, required=True,
                                 choices=['sharc', 'daily_dialog', 'bitext'],
                                 help="Which dataset to expect.")
        self.parser.add_argument("--train_file", default=None, type=str,
                                 help="Input file for training")
        self.parser.add_argument("--predict_file", default=None, type=str,
                                 help="Input file for prediction.")
        self.parser.add_argument("--valid_gold", default=None, type=str,
                                 help="Location of gold file for evaluating predictions.")
        self.parser.add_argument("--max_seq_length", default=384, type=int,
                                 help="The maximum total sequence length (Part A + B) after "
                                      "tokenization. "
                                      "Note: For daily_dialog we truncate the beginning.")
        self.parser.add_argument("--max_part_a", default=64, type=int,
                                 help="The maximum number of tokens for Part A. Sequences longer "
                                      "than this will be truncated to this length.")
        self.parser.add_argument("--do_train", action='store_true', help="Should be true to run taining.")
        self.parser.add_argument("--do_predict", action='store_true',
                                 help="Should be true to run predictition.")
        self.parser.add_argument("--train_batch_size", default=16, type=int,
                                 help="Batch size to use for training. "
                                      "Actual batch size will be divided by "
                                      "gradient_accumulation_steps and clipped to closest int.")
        self.parser.add_argument("--predict_batch_size", default=16, type=int,
                                 help="Batch size to use for predictions.")
        self.parser.add_argument("--learning_rate", default=1e-5, type=float,
                                 help="The learning rate for Adam.")
        self.parser.add_argument("--num_train_epochs", default=3.0, type=float,
                                 help="How many training epochs to run.")
        self.parser.add_argument('--seed',
                                 type=int,
                                 default=42,
                                 help="Random seed for initialization, "
                                      "set to -1 to draw a random number.")
        self.parser.add_argument('--gradient_accumulation_steps',
                                 type=int,
                                 default=1,
                                 help="Number of updates steps to accumulate before performing "
                                      "a backward/update pass.")
        self.parser.add_argument("--do_lower_case",
                                 action='store_true',
                                 help="Whether to lower case the input text. "
                                      "Should be True for uncased models, False for cased models.")


class BisonArguments(GeneralArguments):
    """
    Arguments relevant for generating with BERT."""

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.__class__.__name__

    def _add_arguments(self):
        super(BisonArguments, self)._add_arguments()
        # Required parameters
        self.parser.add_argument("--bert_model", default=None, type=str, required=True,
                                 choices=['bert-large-uncased', 'bert-base-uncased',
                                          'bert-base-cased', 'bert-large-cased',
                                          'bert-base-multilingual-uncased',
                                          'bert-base-multilingual-cased',
                                          'bert-base-chinese', 'bert-vanilla'],
                                 help="Bert pre-trained model to use.")
        self.parser.add_argument("--bert_tokenizer", default=None, const=None, nargs='?',
                                 type=str,
                                 choices=['bert-large-uncased', 'bert-base-cased',
                                          'bert-large-cased',
                                          'bert-base-multilingual-uncased',
                                          'bert-base-multilingual-cased',
                                          'bert-base-chinese', 'bert-vanilla'],
                                 help="If the tokenizer should differ from the model, "
                                      "e.g. when initializing weights randomly but still want to "
                                      "use the vocabulary of a pre-trained BERT model.")

        #pertaining masking
        self.parser.add_argument("--masking", default=None, type=str, required=True,
                                 choices=['gen'],
                                 help="Selection of: 'gen' for generating sequences.")
        self.parser.add_argument("--max_gen_length", default=50, type=int,
                                 help="Maximum length for output generation sequence (Part B).")
        # for GenerationMasking and RandomAllPartsMasking
        self.parser.add_argument("--masking_strategy", default=None, type=str, const=None,
                                 nargs='?',
                                 choices=['bernoulli', 'gaussian'],
                                 help="Which masking strategy to us, options are: "
                                      "bernoulli, gaussian")
        self.parser.add_argument("--distribution_mean", default=1.0, type=float,
                                 help="The mean (for Bernoulli and Gaussian sampling).")
        self.parser.add_argument("--distribution_stdev", default=0.0, type=float,
                                 help="The standard deviation (for Gaussian sampling).")

        #pertaining prediction
        self.parser.add_argument("--predict", type=str, default='one_step_greedy',
                                 const='one_step_greedy',
                                 nargs='?',
                                 choices=['one_step_greedy', 'left2right', 'max_probability',
                                          'min_entropy', 'right2left', 'no_look_ahead'],
                                 help="How perdiction should be run.")
