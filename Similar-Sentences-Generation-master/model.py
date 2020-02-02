# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import sys
sys.path.append('./')

from hyperparams import Hyperparams as hp
from load_data import *
from modules import *


class TransGAN(object):
    def __init__(self, source_vocab_size, target_vocab_size, SIGMA, LAMBDA, is_training):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.SIGMA = SIGMA
        self.LAMBDA = LAMBDA
        self.is_training = is_training

        if self.is_training:
            X, Y, _, _ = load_data(hp.train_file, hp.maxlen)

            
            # calc total batch count
            self.num_batch = len(X) // hp.batch_size
            
            # Convert to tensor
            X = tf.convert_to_tensor(X, tf.int32)
            Y = tf.convert_to_tensor(Y, tf.int32)
            
            # Create Queues
            input_queues = tf.train.slice_input_producer([X, Y])
                    
            # create batch queues
            self.x, self.y = tf.train.shuffle_batch(input_queues,
                                        num_threads=8,
                                        batch_size=hp.batch_size, 
                                        capacity=hp.batch_size*64,   
                                        min_after_dequeue=hp.batch_size*32, 
                                        allow_smaller_final_batch=False)
        else: # inference
            self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

        self._creat_model()


    def generator(self):
        # define decoder inputs
        decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1])*2, self.y[:, :-1]), -1) # 2:<S>

        # Encoder
        with tf.variable_scope("encoder"):
            ## Embedding
            enc = embedding(self.x, 
                                  vocab_size=self.source_vocab_size, 
                                  num_units=hp.hidden_units, 
                                  scale=True,
                                  scope="enc_embed")
            
            ## Positional Encoding
            enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                  vocab_size=hp.maxlen, 
                                  num_units=hp.hidden_units, 
                                  zero_pad=False, 
                                  scale=False,
                                  scope="enc_pe") 
             
            ## Dropout
            enc = tf.layers.dropout(enc, 
                                        rate=hp.dropout_rate, 
                                        training=tf.convert_to_tensor(self.is_training))
            
            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    enc = multihead_attention(queries=enc, 
                                                    keys=enc, 
                                                    num_units=hp.hidden_units, 
                                                    num_heads=hp.num_heads, 
                                                    dropout_rate=hp.dropout_rate,
                                                    is_training=self.is_training,
                                                    causality=False)
                    
                    ### Feed Forward
                    enc = feedforward(enc, num_units=[4*hp.hidden_units, hp.hidden_units])
        
        # Decoder
        with tf.variable_scope("decoder"):
            ## Embedding
            dec = embedding(decoder_inputs, 
                                  vocab_size=self.target_vocab_size, 
                                  num_units=hp.hidden_units,
                                  scale=True, 
                                  scope="dec_embed")
            
            ## Positional Encoding
            dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(decoder_inputs)[1]), 0), [tf.shape(decoder_inputs)[0], 1]),
                                  vocab_size=hp.maxlen, 
                                  num_units=hp.hidden_units, 
                                  zero_pad=False, 
                                  scale=False,
                                  scope="dec_pe")
            
            ## Dropout
            dec = tf.layers.dropout(dec, 
                                        rate=hp.dropout_rate, 
                                        training=tf.convert_to_tensor(self.is_training))
            
            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention ( self-attention)
                    dec = multihead_attention(queries=dec, 
                                                    keys=dec, 
                                                    num_units=hp.hidden_units, 
                                                    num_heads=hp.num_heads, 
                                                    dropout_rate=hp.dropout_rate,
                                                    is_training=self.is_training,
                                                    causality=True, 
                                                    scope="self_attention")
                    
                    ## Multihead Attention ( vanilla attention)
                    dec = multihead_attention(queries=dec, 
                                                    keys=enc, 
                                                    num_units=hp.hidden_units, 
                                                    num_heads=hp.num_heads,
                                                    dropout_rate=hp.dropout_rate,
                                                    is_training=self.is_training, 
                                                    causality=False,
                                                    scope="vanilla_attention")
                    
                    ## Feed Forward
                    dec = feedforward(dec, num_units=[4*hp.hidden_units, hp.hidden_units])
        logits = tf.layers.dense(dec, self.target_vocab_size)
        return logits
        
    def discriminator(self, x):
        with tf.variable_scope('dense_1'):
            h = tf.layers.dense(x, hp.hidden_units)
        h = tf.reshape(h, [-1, hp.maxlen, hp.hidden_units, 1])
        with tf.variable_scope('conv2d_2_1'):
            hc1 = tf.layers.conv2d(
                h, 
                filters=16, 
                kernel_size=[2, hp.hidden_units],
                strides=[1,1],
                padding='SAME',
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )
            hc1 = tf.nn.relu(h)

        with tf.variable_scope('conv2d_2_2'):
            hc2 = tf.layers.conv2d(
                h,
                filters=16,
                kernel_size=[3, hp.hidden_units],
                strides=[1,1],
                padding='SAME',
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )
            hc2 = tf.nn.relu(h)

        with tf.variable_scope('conv2d_2_3'):
            hc3 = tf.layers.conv2d(
                h,
                filters=16,
                kernel_size=[4, hp.hidden_units],
                strides=[1,1],
                padding='SAME',
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )
            hc3 = tf.nn.relu(h)
        with tf.variable_scope('conv2d_2_4'):
            hc4 = tf.layers.conv2d(
                h,
                filters=16,
                kernel_size=[5, hp.hidden_units],
                strides=[1,1],
                padding='SAME',
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )
            hc4 = tf.nn.relu(h)
        
        h = tf.concat([hc1,hc2,hc3,hc4], 2)

        with tf.variable_scope('max_pooling_3'):
            h = tf.layers.max_pooling2d(
                h,
                pool_size=[hp.maxlen,1],
                strides=1
            )

        with tf.variable_scope('dense_4'):
            h = tf.layers.dense(h, 32)
            h = tf.nn.sigmoid(h)
        with tf.variable_scope('dense_5'):
            h = tf.layers.dense(h, 1)
        return h
        
    def _creat_model(self):
        self.y_ = tf.one_hot(indices=self.y, depth=self.target_vocab_size)
        
        with tf.variable_scope('generator'):
            self.g = self.generator()
        with tf.variable_scope('discriminator') as scope:
            self.D_real = self.discriminator(self.y_)
            scope.reuse_variables()
            self.D_fake = self.discriminator(self.g)

        self.G_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 
            scope='generator'
        )
        self.D_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 
            scope='discriminator'
        )

        self.preds = tf.to_int32(tf.arg_max(self.g, dimension=-1))
        self.istarget = tf.to_float(tf.not_equal(self.y, 0))
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))
        tf.summary.scalar('acc', self.acc)
        if self.is_training:
            # Loss
            self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=self.target_vocab_size))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.g, labels=self.y_smoothed)
            self.content_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))
            tf.summary.scalar('mean_loss', self.content_loss)

            disc_loss = -tf.reduce_mean(self.D_real) + tf.reduce_mean(self.D_fake)
            gen_loss = -tf.reduce_mean(self.D_fake)

            alpha = tf.random_uniform(
                shape=[hp.batch_size, 1, 1],
                minval=0.,
                maxval=1.
            )

            differences = self.y_ - self.g
            interpolates = self.y_ + alpha * differences
            gradients = tf.gradients(self.discriminator(interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

            self.global_step = tf.Variable(0, name='global_step')
            #====
            # tf.assign(ref, value, validate_shape=None, use_locking=None, name=None)
            #函数完成了将value赋值给ref的作用。其中：ref 必须是tf.Variable创建的tensor，如果ref=tf.constant()会报错！
            #同时，shape（value）==shape（ref）
            #=====
            self.gs_op = tf.assign(self.global_step,tf.add(self.global_step, 1))

            self.D_loss = self.SIGMA * (disc_loss + self.LAMBDA * gradient_penalty)
            self.G_loss = self.content_loss + self.SIGMA * gen_loss
            self.D_opt = tf.train.AdamOptimizer(
                learning_rate=hp.D_learning_rate,
                beta1=0.5,
                beta2=0.9
            ).minimize(self.D_loss, var_list=self.D_params)
            self.G_opt = tf.train.AdamOptimizer(
                learning_rate=hp.G_learning_rate,
                beta1=0.8,
                beta2=0.98,
                epsilon=1e-8
            ).minimize(self.G_loss, var_list=self.G_params)
            self.merged = tf.summary.merge_all()


            
        





