import torch
import torch.nn as nn
from torch.nn import init
import torchvision.models as models
from torch.autograd import Variable
import pickle 
import math

import pdb

import climate
import logging
logging = climate.get_logger(__name__)
climate.enable_default_logging()

class SkeletonAction_VA(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, num_layers = 3, use_bias = True, dropout = 0):
        super(SkeletonAction_VA, self).__init__()
        self.input_size = input_size
        self.hidden_isze = hidden_size
        self.num_class = num_class
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.dropout = dropout
        
        self.lstm_rotate = nn.LSTM(input_size, hidden_size, 1, batch_first = True)
        self.fc_r = nn.Linear(hidden_size, 3)
        self.lstm_tran = nn.LSTM(input_size, hidden_size, 1, batch_first = True)
        self.fc_t = nn.Linear(hidden_size, 3)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout = dropout)
        self.linear = nn.Linear(hidden_size, num_class)
        self.dropout_layer = nn.Dropout(dropout)
    def init_weights(self):
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.fc_t.weight.data.uniform_(-0.1, 0.1)
        self.fc_t.bias.data.fill_(0)
        self.fc_r.weight.data.uniform_(-0.1, 0.1)
        self.fc_r.bias.data.fill_(0)
    
    def forward(self, x):
        '''
        x: batch x seq_len x input_size
        '''
        # Now, run the va lstm first.
        batch_size = x.size(0)
        seq_len = x.size(1)
        angles,_ = self.lstm_rotate(x)
        angles = angles.contiguous()
        angles = angles.view((-1, angles.size(-1)))
        angles = self.fc_r(angles)
        angles = angles.view((batch_size, seq_len, angles.size(-1)))
         
        # compute the rotation matrix.
        zero = torch.zeros(batch_size, seq_len, angles.size(-1))
        one = torch.ones(batch_size, seq_len, angles.size(-1))
        if torch.cuda.is_available():
            zero = zero.cuda()
            one = one.cuda()
        zero = Variable(zero) 
        one = Variable(one)
        R = []
        R.append(torch.stack([ torch.cos(angles), torch.sin(angles), zero], dim = 3))
        R.append(torch.stack([ -torch.sin(angles), torch.cos(angles), zero], dim = 3))
        R.append(torch.stack([ zero, zero, one], dim = 3))
        R = torch.stack(R, dim = 3)
        R_ = torch.prod(R, dim = 2) # b x seq x 3 x 3
        R_ = R_.transpose(2,3)
        R_ = R_.unsqueeze(2) # b x seq x 1 x 3 x 3
        trans,_ = self.lstm_tran(x) 
        trans = trans.contiguous()
        trans = trans.view((-1, trans.size(-1)))
        trans = self.fc_t(trans)
        trans = trans.view((batch_size, seq_len, trans.size(-1), 1))
        trans = trans.repeat(1,1,1, x.size(-1) // 3)
        trans = trans.view((batch_size, seq_len, x.size(-1)))
        x = x - trans 
        x = x.view(batch_size, seq_len, x.size(-1) // 3, 1, 3)
        x = (R_ * x).sum(dim = -1)
        x = x.view(batch_size, seq_len, self.input_size)
        x,_ = self.lstm(x)
        x = x.contiguous()
        x = x.view((-1, x.size(-1)))
        x = self.dropout_layer(x)
        x = self.linear(x)
        x = x.view(batch_size, seq_len, x.size(1))
        return x

class SkeletonAction_FT(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, num_layers = 1, use_bias = True, dropout = 0):
        super(SkeletonAction_FT, self).__init__()
        self.input_size = input_size
        self.hidden_isze = hidden_size
        self.num_class = num_class
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear_ft = nn.Linear(hidden_size, num_class)
        self.dropout_layer = nn.Dropout(dropout)
    def init_weights(self):
        self.linear_ft.weight.data.uniform_(-0.1, 0.1)
        self.linear_ft.bias.data.fill_(0)
    
    def forward(self, x):
        '''
        x: batch x seq_len x input_size
        '''
        batch_size = x.size(0)
        seq_len = x.size(1)
        x,_ = self.lstm(x)
        x = x.contiguous()
        x = x.view((-1, x.size(-1)))
        x = self.dropout_layer(x)
        x = self.linear_ft(x)
        x = x.view(batch_size, seq_len, x.size(1))
        return x
