# -*- coding: utf-8 -*-
#/usr/bin/python2
class Hyperparams:
    batch_size = 64 # alias = N
    D_learning_rate = 0.0005
    G_learning_rate = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    source_vocab_file = './source_vocab.tsv'
    target_vocab_file = './target_vocab.tsv'
    train_file = './data/train.txt'
    test_file = './data/test.txt'
    
    # model
    maxlen = 15 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 2 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    
    
    
    
