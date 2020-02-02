import torch
from scipy.stats import truncnorm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#A truncated distribution has its domain (the x-values) restricted to a certain range of values.
#For example, you might restrict your x-values to between 0 and 100, written in math terminology as {0 > x > 100}. 
#There are several types of truncated distributions(截取顶端了的分布):
def truncated_normal(shape, lower=-0.2, upper=0.2):
    size = 1
    for dim in shape:
        size *= dim
    w_truncated = truncnorm.rvs(lower, upper, size=size)
    w_truncated = torch.from_numpy(w_truncated).float()
    w_truncated = w_truncated.view(shape)
    return w_truncated

class Highway(nn.Module):
    #Highway Networks = Gating Function To Highway = y = xA^T + b
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.fc1 = nn.Linear(in_size, out_size)
        self.fc2 = nn.Linear(in_size, out_size)
    def forward(self, x):
        #highway = F.sigmoid(highway)*F.relu(highway) + (1. - transform)*pred # sets C = 1 - T
        g = F.relu(self.fc1)
        t = torch.sigmoid(self.fc2)
        out = g*t + (1. - t)*x
        return out

class Discriminator(nn.Module):
    """
    A CNN for text classification
    num_filters (int): This is the output dim for each convolutional layer, which is the number
          of "filters" learned by that layer.
    """
    def __init__(self, seq_len, num_classes, vocab_size, dis_emb_dim, 
                    filter_sizes, num_filters, start_token, goal_out_size, step_size, dropout_prob, l2_reg_lambda):
        '''
        'discriminator_params': 
         {'seq_len': 20, 
         'num_classes': 2, 
         'vocab_size': 5258, 
         'dis_emb_dim': 64, 
         'filter_sizes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20], 
         'num_filters': [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160], 
         'start_token': 0, 
         'goal_out_size': sum(discriminator_params["num_filters"]) 
         'step_size': 5, 
         'dropout_prob': 0.8,
         'l2_reg_lambda': 0.2

         gen_corpus是一个9984*20的矩阵；
        '''
        super(Discriminator, self).__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.dis_emb_dim = dis_emb_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.start_token = start_token #这是什么东西
        self.goal_out_size = goal_out_size
        self.step_size = step_size
        self.dropout_prob = dropout_prob
        self.l2_reg_lambda = l2_reg_lambda
        self.num_filters_total = sum(self.num_filters)
        
        #Building up layers
        self.emb = nn.Embedding(self.vocab_size + 1, self.dis_emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_f, (f_size, self.dis_emb_dim)) for f_size, num_f in zip(self.filter_sizes, self.num_filters)
        ])
        self.highway = nn.Linear(self.num_filters_total, self.num_filters_total)
        #in_features = out_features = sum of num_festures
        self.dropout = nn.Dropout(p = self.dropout_prob)
        #Randomly zeroes some of the elements of the input tensor with probability p using Bernouli distribution
        #Each channel will be zeroed independently onn every forward call
        self.fc = nn.Linear(self.num_filters_total, self.num_classes)
        
    def forward(self, x):
        """
        Argument:
            x: shape(batch_size * self.seq_len)
               type(Variable containing torch.LongTensor)
        Return:
            pred: shape(batch_size * 2)
                  For each sequence in the mini batch, output the probability
                  of it belonging to positive sample and negative sample.
            feature: shape(batch_size * self.num_filters_total)
                     Corresponding to f_t in original paper
            score: shape(batch_size, self.num_classes)
              
        """
        #1. Embedding Layer
        #2. Convolution + maxpool layer for each filter size
        #3. Combine all the pooled features into a prediction
        #4. Add highway
        #5. Add dropout. This is when feature should be extracted
        #6. Final unnormalized scores and predictions
        emb = self.emb(x).unsqueeze(1) # [batch_size * 1 * seq_len * emb_dim] because of the in_channels = 1 for the convs By Jian
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs] # the size of the element in the List is [batch_size * num_filter * (seq_len - kernel_size + 1)]
        pooled_out = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] #  the size of the element in the List is [batch_size * num_filter]
        pred = torch.cat(pooled_out, 1) # batch_size * sum(num_filters)
        #print("Pred size: {}".format(pred.size()))
        highway = self.highway(pred)
        #print("highway size: {}".format(highway.size()))
        highway = torch.sigmoid(highway)* F.relu(highway) + (1.0 - torch.sigmoid(highway))*pred
        features = self.dropout(highway)
        score = self.fc(features)
        pred = F.log_softmax(score, dim=1) #batch * num_classes
        return {"pred":pred, "feature":features, "score": score}

    def l2_loss(self):
        W = self.fc.weight
        b = self.fc.bias
        l2_loss = torch.sum(W*W) + torch.sum(b*b)
        l2_loss = self.l2_reg_lambda * l2_loss
        return l2_loss
