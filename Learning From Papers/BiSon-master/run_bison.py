# coding=utf-8
#        BiSon

"""
Main script that starts BiSon.
"""

import logging
import os

from bison.arguments import BisonArguments
from bison.bison_handler import bison_runner
from bison.util import write_list_to_file

logging.basicConfig(format='%(name)s - %(message)s',
                    level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def start_bison_runner(bison_args):
    """
    Starts BiSon training or prediction.
    :param bison_args: instance of :py:class:BisonArguments
    :return: A tuple of:
            1. the best score of the validation set during training
            2. the best score after prediction
    """
    write_list_to_file([BISON_ARGS.parser.parse_args()],
                       os.path.join(bison_args.output_dir, "bison_args"))
    return bison_runner(BISON_ARGS)


if __name__ == "__main__":
    BISON_ARGS = BisonArguments()
    LOGGER.info("Arguments: %s", BISON_ARGS.parser.parse_args())
    start_bison_runner(BISON_ARGS)
