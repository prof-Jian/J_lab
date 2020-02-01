# -*- coding: utf-8 -*-

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    """A CNN for text classification

    architecture: Embedding >> Convolution >> Max-pooling >> Softmax
    """

    def __init__(self, num_classes, vocab_size, emb_dim, filter_sizes, num_filters, dropout):
        super(Discriminator, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, emb_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(sum(num_filters), num_classes) # num_classes
        self.softmax = nn.LogSoftmax()
        self.init_parameters()

    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)
        我的一些想法：
            - 为什么我不可以用conv1d来写这些？
            - 原文中如是说：We can use various numbers of kernels with different window filter_sizes
            - highway architecture是一篇文章里的名字，但这里highway只是一个全连接；
        """
        emb = self.emb(x).unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim
        # 也就是说conv的每一个stride的结果是按照stride的方向排的，否则也不需要squeeze了；
        # 另外，这里并没有衔接，而是在取最值；
        # length因为取最大值不用担心维度具体是多少，或者矩阵接不上
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        """
        以下两行是highway architercture,
        知乎上说在全连接层，并且输入输出一样大小时候，将全连接替换成highway对结果有一点提升
        “原理是给输出加一个gate来控制信息的流动”
        另外，文章里面说优化Discriminator要用cross entropy
        """
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) *  F.relu(highway) + (1. - torch.sigmoid(highway)) * pred
        pred = self.softmax(self.lin(self.dropout(pred)),dim = 1)
        return pred

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
