# coding=utf-8
#        BiSon
"""
Provides some utility functions.
"""

import codecs
import json
import numpy as np


def write_list_to_file(list_to_write, file_to_write):
    """
    Write a list to a file.
    :param list_to_write: the list to be written to a file
    :param file_to_write: the file to write to
    :return: 0 on success
    """
    with codecs.open(file_to_write, 'w', encoding='utf8') as file:
        for line in list_to_write:
            print(line, file=file)
    return 0


def write_json_to_file(json_object, file_to_write):
    """
    Write a json object to a file.
    :param json: the json object to write
    :param file_to_write: the location to write to
    :return: 0 on success
    """
    with open(file_to_write, "w") as writer:
        writer.write(json.dumps(json_object, indent=4) + "\n")
    return 0


def read_lines_in_list(file_to_read):
    """
    Reads a file into a list.
    :param file_to_read: the location of the file to be read
    :return: a list where each entry corresponds to a line in the file
    """
    read_list = []
    with codecs.open(file_to_read, 'r', encoding='utf8') as file:
        for line in file:
            read_list.append(line.rstrip('\n'))
    return read_list


def read_json(json_to_read):
    """
    Read a json file
    :param json_to_read: the json object to read
    :return: the json object
    """
    with open(json_to_read, "r") as reader:
        json_object = json.loads(reader)
    return json_object


def compute_softmax(scores, alpha=1.0):
    """
    Computes softmax probaility over raw logits
    :param scores: a numpy array with logits
    :param alpha: temperature parameter, values >1.0 approach argmax, values <1.0 approach uniform
    :return: a numpy array with probability scores
    """
    scores = scores * float(alpha)
    scores = scores - np.max(scores)
    scores = np.exp(scores)
    probs = scores / np.sum(scores)

    return probs
