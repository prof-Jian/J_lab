# -*- coding: utf-8 -*-

from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from load_data import *
from model import *
from nltk.translate.bleu_score import corpus_bleu
import heapq

def eval(out_file, is_multi=True):
    X, Y, sources, targets = load_data(hp.test_file, hp.maxlen)
    source_word_index, source_index_word = load_vocab(hp.source_vocab_file)
    target_word_index, target_index_word = load_vocab(hp.target_vocab_file)

    # Load graph
    gan = TransGAN(
        source_vocab_size = len(source_word_index),
        target_vocab_size = len(target_word_index),
        SIGMA = 1e-3,
        LAMBDA = 10,
        is_training = False
    )
    
     
    # Start session         
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        ## Restore parameters
        if tf.train.get_checkpoint_state('./backup/latest/'):
            saver = tf.train.Saver()
            saver.restore(sess, './backup/latest/')
            print('********Restore the latest trained parameters.********')
        else:
            print('********The model is not existed.********')
          
        ## Inference
        with codecs.open(out_file, "w", "utf-8") as fout:
            list_of_refs, hypotheses = [], []
            for i in range(len(X)):
                 
                ### Get mini-batches
                x = X[i: (i+1)]
                source = sources[i: (i+1)]
                target = targets[i: (i+1)]
                if is_multi:
                 
                    for max_idx in range(1, 3):
                        for k in range(-1, (hp.maxlen // 2 - 2)):
                            ### Autoregressive inference
                            preds = np.zeros((1, hp.maxlen), np.int32)
                            for j in range(hp.maxlen):
                                _logits, _preds = sess.run([gan.g, gan.preds], {gan.x: x, gan.y: preds})
                                _logits = _logits[0,j,:]
                                word_indices = heapq.nlargest(3, range(len(_logits)), _logits.take)
                                if j == k:
                                    preds[0, j] = word_indices[max_idx]
                                else:
                                    preds[:, j] = _preds[:, j]
                                #print(target_index_word[_preds[0,j]], target_index_word[preds[0,j]])
                                #print(target_index_word[word_indices[0]], target_index_word[word_indices[1]])
                             
                            ### Write to file
                            for s, t, pred in zip(source, target, preds): # sentence-wise
                                got = " ".join(target_index_word[idx] for idx in pred).split("</S>")[0].strip()
                                fout.write(got +"\t" + s + "\n")
                                #fout.write("- expected: " + t + "\n")
                                #fout.write("- got: " + got + "\n\n")
                                fout.flush()
                                  
                                # bleu score
                                #ref = t.split()
                                #hypothesis = got.split()
                                #if len(ref) > 3 and len(hypothesis) > 3:
                                #    list_of_refs.append([ref])
                                #    hypotheses.append(hypothesis)
                else:
                    preds = np.zeros((1, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds = sess.run(gan.preds, {gan.x: x, gan.y: preds})
                        preds[:, j] = _preds[:, j]
                     
                    ### Write to file
                    for s, t, pred in zip(source, target, preds): # sentence-wise
                        got = " ".join(target_index_word[idx] for idx in pred).split("</S>")[0].strip()
                        fout.write(got +"\t" + s + "\n")
                        fout.flush()
          

if __name__ == '__main__':
    eval('./results/out.txt', True)
