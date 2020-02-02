# -*- coding: utf-8 -*-

from __future__ import print_function
from hyperparams import Hyperparams as hp
import numpy as np
import codecs
import re

def load_vocab(file_name, min_cnt=hp.min_cnt):
    vocab = [line.split()[0] for line in codecs.open(file_name, 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=min_cnt]
    word_index = {word: idx for idx, word in enumerate(vocab)}
    index_word = {idx: word for idx, word in enumerate(vocab)}
    return word_index, index_word

def load_data(file_name, maxlen):
    fin = codecs.open(file_name, 'r', 'utf-8')
    source_sents = []
    target_sents = []
    while True:
        text = fin.readline().strip()
        if text == "":
            break
        splits = text.split('\t')
        #if not re.match(ur"[\u4e00-\u9fa5 ]+\t[\u4e00-\u9fa5 ]+\t[0-1]", text):
        #    continue
        if len(splits) != 2:
            continue
        source_sents.append(splits[0])
        target_sents.append(splits[1])
    
    source_word_index, source_index_word = load_vocab('./source_vocab.tsv')
    target_word_index, target_index_word = load_vocab('./target_vocab.tsv')
    
    # Index
    x_list, y_list, sources, targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [source_word_index.get(word, 1) for word in (source_sent + u" </S>").split()] # 1: OOV, </S>: End of Text
        y = [target_word_index.get(word, 1) for word in (target_sent + u" </S>").split()] 
        if max(len(x), len(y)) <= maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            sources.append(source_sent)
            targets.append(target_sent)
    
    # Pad      
    X = np.zeros([len(x_list), maxlen], np.int32)
    Y = np.zeros([len(y_list), maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, maxlen-len(y)], 'constant', constant_values=(0, 0))
    return X, Y, sources, targets
    
if __name__ == "__main__":
    train_file = '../data/train.txt'
    X, Y, s, t = load_data(train_file, 10)
    print(s[0])
    print(X[0])
    print(t[0])
    print(Y[0])
