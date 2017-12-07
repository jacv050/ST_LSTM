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

class SkeletonAction(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, num_layers = 1, use_bias = True, dropout = 0):
        super(SkeletonAction, self).__init__()
        self.input_size = input_size
        self.hidden_isze = hidden_size
        self.num_class = num_class
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_class)
        self.dropout_layer = nn.Dropout(dropout)
    def init_weights(self):
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
    
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
        x = self.linear(x)
        x = x.view(batch_size, seq_len, x.size(1))
        return x

class SkeletonAction_AVG_H(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, num_layers = 3, use_bias = True, dropout = 0):
        super(SkeletonAction_AVG_H, self).__init__()
        self.input_size = input_size
        self.hidden_isze = hidden_size
        self.num_class = num_class
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout = dropout)
        self.linear = nn.Linear(hidden_size, num_class)
        self.dropout_layer = nn.Dropout(dropout)
    def init_weights(self):
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
    
    def forward(self, x, mask):
        '''
        x: batch x seq_len x input_size
        '''
        batch_size = x.size(0)
        seq_len = x.size(1)
        x,_ = self.lstm(x)
        x = x.contiguous()
        #x = x.view((-1, x.size(-1)))
        #x = self.dropout_layer(x)
        x = (x * mask).sum(dim = 1) / mask.sum(dim = 1)
        x = self.linear(x)
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
