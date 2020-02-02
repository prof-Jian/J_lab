from scipy.stats import truncnorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

#A truncated distribution has its domain (the x-values) restricted to a certain
#range of values. For example, you might restrict your x-values to between 0 and 100,
#written in math terminology as {0 > x > 100}. There are several types of truncated distributions:
def truncated_normal(shape, lower=-0.2, upper=0.2):
'''
shape(size:[batch_size,goal_out_size])

output:
    w_truncated(shape:[batch_size,goal_out_size])

我感觉作者是不知道pytorch中有类似产生变量的方法才用这个函数

'''
    size = 1
    for dim in shape:
        size *= dim # so the size = batch_size*goal_out_size,the type is int??
    w_truncated = truncnorm.rvs(lower, upper, size=size)
    w_truncated = torch.from_numpy(w_truncated).float()
    w_truncated = w_truncated.view(shape)
    return w_truncated

class Manager(nn.Module):
    def __init__(self, batch_size, hidden_dim, goal_out_size):
        super(Manager, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.goal_out_size = goal_out_size
        self.recurrent_unit = nn.LSTMCell(
            self.goal_out_size, #input size,   so the input's shape should be [batch_size,input_size]
            self.hidden_dim #hidden size , so h_0,c_0,h_1,c_1's shape should be [batch_size,hidden_size]
        )
        self.fc = nn.Linear(
            self.hidden_dim, #in_features
            self.goal_out_size #out_features
        )
        self.goal_init = nn.Parameter(torch.zeros(self.batch_size, self.goal_out_size))
        self._init_params() 

    def _init_params(self):
        for param in self.parameters():
            nn.init.normal_(param, std=0.1)
        self.goal_init.data = truncated_normal(
            self.goal_init.data.shape
        )
    def forward(self, f_t, h_m_t, c_m_t):
        """
        f_t = feature of CNN from discriminator leaked at time t, it is input into LSTM
        h_m_t = ouput of previous LSTMCell
        c_m_t = previous cell state
        """
        #print("H_M size: {}".format(h_m_t.size()))
        #print("C_M size: {}".format(c_m_t.size()))
        #print("F_t size: {}".format(f_t.size()))
        h_m_tp1, c_m_tp1 = self.recurrent_unit(f_t, (h_m_t, c_m_t))
        sub_goal = self.fc(h_m_tp1)
        #torch.renorm(...):Returns a tensor where each sub-tensor of input along dimension 0 dim 
        #is normalized such that the p-norm of the sub-tensor is lower than the value maxnorm
        sub_goal = torch.renorm(sub_goal, 2, 0, 1.0) #p = 2, dim = 0, maxnorm = 1
        return sub_goal, h_m_tp1, c_m_tp1 # sub_goal's shape should be [batch_size,goal_out_size]


class Worker(nn.Module):
    def __init__(self, batch_size, vocab_size, embed_dim, hidden_dim, 
                    goal_out_size, goal_size):
        super(Worker, self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.goal_out_size = goal_out_size
        self.goal_size = goal_size

        self.emb = nn.Embedding(self.vocab_size, self.embed_dim)
        self.recurrent_unit = nn.LSTMCell(self.embed_dim, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.goal_size*self.vocab_size)
        self.goal_change = nn.Parameter(torch.zeros(self.goal_out_size, self.goal_size))
        self._init_params()
        
    def _init_params(self):
        for param in self.parameters():
            nn.init.normal_(param, std=0.1)
    # 这里相对Manager少一行

    def forward(self, x_t, h_w_t, c_w_t):
        """
            x_t = last word
            h_w_t = last output of LSTM in Worker
            c_w_t = last cell state of LSTM in Worker
        """
        x_t_emb = self.emb(x_t)
        h_w_tp1, c_w_tp1 = self.recurrent_unit(x_t_emb, (h_w_t, c_w_t))
        output_tp1 = self.fc(h_w_tp1) # [batch_size,goal_size*vocab_size]
        output_tp1 = output_tp1.view(self.batch_size, self.vocab_size, self.goal_size)
        return output_tp1, h_w_tp1, c_w_tp1


class Generator(nn.Module):
    def __init__(self, worker_params, manager_params, step_size):
        super(Generator, self).__init__()
        self.step_size = step_size #
        self.worker = Worker(**worker_params)
        self.manager = Manager(**manager_params)

    def init_hidden(self):
        h = Variable(torch.zeros(self.worker.batch_size, self.worker.hidden_dim))
        c = Variable(torch.zeros(self.worker.batch_size, self.worker.hidden_dim))
        return h, c

    def forward(self, x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal, t, temperature):
        '''
        args:
            x_t([]):the current word
            f_t([]):the feature vector come from the discriminator

            real_goal([]):
            temperature([]):as said in paper,this parameter is to control the generation entropy.
        
        output:
            last_goal_temp([]):输出作为last_goal
        '''
        sub_goal, h_m_tp1, c_m_tp1 = self.manager(f_t, h_m_t, c_m_t) #sub_goal = [batch_size,goal_out_size]
        output, h_w_tp1, c_w_tp1 = self.worker(x_t, h_w_t, c_w_t) # output = [batch_size, vocab_size, goal_size] (as the O_t in the paper)
        last_goal_temp = last_goal + sub_goal # 就单用上一个last_goal加上用f_t产生的sub_goal？？
        w_t = torch.matmul( #矩阵乘法 []
            real_goal, self.worker.goal_change # worker.goal_change[goal_out_size, goal_size]
        )
        # 主要问题就在上面，last_goal_temp后续没作用，意味着Manager这次的信息根本没到后续的w_t中，也就没到logits中
        w_t = torch.renorm(w_t, 2, 0, 1.0) # []
        w_t = torch.unsqueeze(w_t, -1)
        logits = torch.squeeze(torch.matmul(output, w_t)) #logits for words
        probs = F.softmax(temperature * logits, dim=1) 
        x_tp1 = Categorical(probs).sample()
        return x_tp1, h_m_tp1, c_m_tp1, h_w_tp1, c_w_tp1,\
                last_goal_temp, real_goal, sub_goal, probs, t + 1
