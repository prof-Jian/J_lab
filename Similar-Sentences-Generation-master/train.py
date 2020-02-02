# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import sys
sys.path.append('./')
import tensorflow as tf
import heapq
from hyperparams import Hyperparams as hp
from model import *

from load_data import *
import os, codecs
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

step_num = 80000


def train():
    if not os.path.exists('./backup/'):
        os.mkdir('./backup/')
    if not os.path.exists('./backup/latest/'):
        os.mkdir('./backup/latest/')
    if not os.path.exists('./summaries'):
        os.mkdir('./summaries')
    # Load vocabulary    
    source_word_index, source_index_word = load_vocab(hp.source_vocab_file) #用已知文件做vocab
    target_word_index, target_index_word = load_vocab(hp.target_vocab_file)
    
    # Construct graph
    gan = TransGAN(
        source_vocab_size = len(source_word_index),
        target_vocab_size = len(target_word_index),
        SIGMA = 1e-3,
        LAMBDA = 10,
        is_training = True
    )
    
    summary_writer = tf.summary.FileWriter('./summaries/')
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True # 根据需要分配显存
    #config.allow_soft_placement = True # 自动选择设备

    # Start session
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    #sess.run(tf.initialize_local_variables())

    if tf.train.get_checkpoint_state('./backup/latest/'):
        saver = tf.train.Saver()
        saver.restore(sess, './backup/latest/')
        print('********Restore the latest trained parameters.********')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for step in tqdm(range(step_num), total=step_num, ncols=70, leave=False, unit='b'):
        gs = sess.run(gan.global_step)   
        if coord.should_stop():
            break
        for _ in range(5):
            _, loss_d = sess.run([gan.D_opt, gan.D_loss])
        _, loss_g = sess.run([gan.G_opt, gan.G_loss])
        sess.run(gan.gs_op)
        if gs % 200 == 0:
            summary = sess.run(gan.merged)
            summary_writer.add_summary(summary, gs)
            print('********step: {}, D_loss = {:.8f}, G_loss = {:.8f}********'.format(gs, loss_d, loss_g))
        if gs % 1500 == 0:
                saver = tf.train.Saver()
                saver.save(sess, './backup/latest/', write_meta_graph=False)
    coord.request_stop()
    coord.join(threads)
    saver = tf.train.Saver()
    saver.save(sess, './backup/latest/', write_meta_graph=False)
    sess.close()
    
    print("Done")    

if __name__ == '__main__':
    train()
