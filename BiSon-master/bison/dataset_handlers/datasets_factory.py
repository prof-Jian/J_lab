# coding=utf-8
#        BiSon
"""
Returns a dataset_handler as specified in command line arguments.
"""

from .dataset_sharc import SharcHandler
from .dataset_bitext import BitextHandler
from .dataset_daily import DailyDialogHandler


def get_data_handler(bert_args):
    """
    Factory for returning dataset specific handlers.
    :param bert_args: instance of :py:class:Arguments
    :return: an instance or a subclass instance of :py:class:BitextHandler
    """
    dataset_handler = None
    if bert_args.data_set == 'sharc':
        dataset_handler = SharcHandler()
    elif bert_args.data_set == 'daily_dialog':
        dataset_handler = DailyDialogHandler(bert_args.dataset_dailydialog_classify_type)
    return dataset_handler
