# -*- coding:utf-8 -*-

import os
import random
import math

import argparse
import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from generator import Generator
from discriminator import Discriminator
from target_lstm import TargetLSTM
from rollout import Rollout
from data_iter import GenDataIter, DisDataIter
# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=0, type=int) # default = None | cuda = GPU数量
opt = parser.parse_args()
print(opt)

# Basic Training Paramters
SEED = 88
BATCH_SIZE = 64
TOTAL_BATCH = 50
GENERATED_NUM = 3000
POSITIVE_FILE = 'real_jieba.data'
NEGATIVE_FILE = 'gene.data'
EVAL_FILE = 'eval.data'
VOCAB_SIZE = 3800
PRE_EPOCH_NUM = 150

PATH_GPU = 'E:\\github lab\\Generation of Synthetic EMRecord-PyTorch-master\\model\\jieba_padding_generator.pkl'

if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True

# Genrator Parameters
g_emb_dim = 32
g_hidden_dim = 32
g_sequence_len = 40

# Discriminator Parameters
d_emb_dim = 64
d_filter_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
d_num_filters = [200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

d_dropout = 0.75
d_num_class = 2



def generate_samples(model, batch_size, generated_num, output_file):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, g_sequence_len).cpu().data.numpy().tolist() # batch_size * seq_len
        samples.extend(sample) #注意这里不是append，我先默认这个extend变动的就是dim=0，我看gene.data:(9984 * 20)
    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)

def train_epoch(model, data_iter, criterion, optimizer):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:#tqdm(data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data) # batch_size * (1 + seq_len)
        target = Variable(target) # batch_size * (seq_len + 1)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1) # target.shpe = [N,] ,其中N = batch_size * (seq_len + 1)
        pred = model.forward(data) # pred.shape = (batch_size * (1+seq_len))*num_emb
        loss = criterion(pred, target) #  具体是NLLLoss函数的输入和输出；就是看类one-hot vector和整数c对应得是否够好，其中整数c对应one-hot vector中第c个值；
        # loss是tensor型，shape为[1,]
        total_loss += loss.item() # tensor.item()将type变成float；
        total_words += data.size(0) * data.size(1) # 扫过单词数量
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()
    return math.exp(total_loss / total_words) #平均loss率

def eval_epoch(model, data_iter, criterion):
    total_loss = 0.
    total_words = 0.
    with torch.no_grad():
        for (data, target) in data_iter:#tqdm(
            #data_iter, mininterval=2, desc=' - Training', leave=False):
            data = Variable(data)
            target = Variable(target)
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            target = target.contiguous().view(-1)
            pred = model.forward(data)
            loss = criterion(pred, target)
            total_loss += loss.item()
            total_words += data.size(0) * data.size(1)
        data_iter.reset()

    assert total_words > 0  # Otherwise NullpointerException
    return math.exp(total_loss / total_words)

class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable C：Vocabulary的大小 原句前面加了个0，得到原句每一个词被前面的句子生成的概率分布
            target : (N, ), torch Variable，是原句
            reward : (N, ), torch Variable，从原句的每个字开始，生成句子能骗过Discriminator的得分
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor) # torch.ByteTensor
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot) # 到这里就是根据target将prob里的对应值取出来
        loss = loss * reward 
        '''
        有种预感，这里是可以修改得更好得地方：
        首先，133行的loss是Log_softmax的结果，如果产生该词的概率低（上一行loss负得多），但reward大，那么generator的问题就越大
        但是，如果产生该词的概率大（上一行loss负得少），但是    reward大  比  reward小    产生得loss还要多一些
        '''
        loss =  -torch.sum(loss)
        return loss


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Define Networks
    generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
    discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)
    if opt.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    # Generate toy data using target lstm 也就是新建一个data，假装他就是真实数据
    print('啊啊')

    # Load data from file
    # 每个iter输出一个data和一个target，其中data是每个point前填个0，每个target是后面添个0
    gen_data_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE)

    # Pretrain Generator using MLE
    gen_criterion = nn.NLLLoss(reduction='sum') #You may use CrossEntropyLoss instead, if you prefer not to add an extra LogSoftmax layer .
    gen_optimizer = optim.Adam(generator.parameters())
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()
    print('Pretrain with MLE ...')
    for epoch in range(PRE_EPOCH_NUM): #PRE_EPOCH_NUM =120
        loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer) #使得generator的参数更新，使之适应gen_data_iter
        print('Epoch [%d] Model Loss: %f'% (epoch, loss)) # 9月1日

    # Pretrain Discriminator
    dis_criterion = nn.NLLLoss(reduction='sum')
    dis_optimizer = optim.Adam(discriminator.parameters())
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    print('Pretrain Discriminator ...')
    for epoch in range(5):
        generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
        dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
        for _ in range(3):
            loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
            print('Epoch [%d], loss: %f' % (epoch, loss))
    # Adversarial Training
    rollout = Rollout(generator, 0.8)
    print('#####################################################')
    print('Start Adeversatial Training...\n')
    gen_gan_loss = GANLoss()
    gen_gan_optm = optim.Adam(generator.parameters())
    if opt.cuda:
        gen_gan_loss = gen_gan_loss.cuda()
    gen_criterion = nn.NLLLoss(reduction='sum')
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()
    dis_criterion = nn.NLLLoss(reduction='sum')
    dis_optimizer = optim.Adam(discriminator.parameters())
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    for total_batch in range(TOTAL_BATCH):
        ## Train the generator for one step
        for it in range(1):
            samples = generator.sample(BATCH_SIZE, g_sequence_len)
            # construct the input to the genrator, add zeros before samples and delete the last column
            zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
            if samples.is_cuda:
                zeros = zeros.cuda()
            inputs = Variable(torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())
            targets = Variable(samples.data).contiguous().view((-1,)) # 这里我不明白，为什么inputs：（batch_size,seq_len)，而targets是一个序列
            # calculate the reward 为什么在rollout里reward不能直接直接由generator来sample
            rewards = rollout.get_reward(samples, 16, discriminator) # rewards:(batch_size,seq_len)
            rewards = Variable(torch.Tensor(rewards))
            rewards = torch.exp(rewards).contiguous().view((-1,)) # 这是因为Discriminator的最后一层是log_Softmax
            if opt.cuda:
                rewards = rewards.cuda()
            prob = generator.forward(inputs)
            loss = gen_gan_loss(prob, targets, rewards) #这里是点睛之笔
            gen_gan_optm.zero_grad()
            loss.backward()
            gen_gan_optm.step() #这里更新的是rollout里面的ori_model吗？

        rollout.update_params() #这里理解了的话，基本没问题了

        for _ in range(4):
            generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
            dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
            for _ in range(2):
                loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer)

        print('Adversarial Training %d complete \n' % (total_batch))

    print('保存模型genetor')
    torch.save(generator.state_dict(), PATH_GPU)

if __name__ == '__main__':
    main()
