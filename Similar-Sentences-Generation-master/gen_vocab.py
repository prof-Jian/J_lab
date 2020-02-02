# -*- coding: utf-8 -*-
'''
This file is used to make a vocabulary from a given corpus.

June.30 2017 by ymthink
yinmiaothink@gmail.com
'''

import codecs
import os
from collections import Counter
from hyperparams import Hyperparams as hp

def gen_vocab(files, source_file, target_file):
    source_word_cnt = Counter()
    target_word_cnt = Counter()
    for file in files:
        fin = codecs.open(file, 'r', 'utf-8')
        while True:
            text = fin.readline()
            if not text:
                break
            sources, targets = text.split('\t')
            source_word_cnt.update(sources.split())
            target_word_cnt.update(targets.split())

    with codecs.open('{}'.format(source_file), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in source_word_cnt.most_common(len(source_word_cnt)):
            if(cnt == 1):
                continue
            fout.write(u"{}\t{}\n".format(word, cnt))

    with codecs.open('{}'.format(target_file), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in target_word_cnt.most_common(len(target_word_cnt)):
            if(cnt == 1):
                continue
            fout.write(u"{}\t{}\n".format(word, cnt))
        

if __name__ == '__main__':
    files = []
    files.append(hp.train_file)
    files.append(hp.test_file)
    gen_vocab(files, 'source_vocab.tsv', 'target_vocab.tsv')








