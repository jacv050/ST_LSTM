import torch
import torch.nn as nn
from torch.nn import init
import torchvision.models as models
from torch.autograd import Variable
import numpy as np

import pickle 
import math

import pdb

import climate
import logging
logging = climate.get_logger(__name__)
climate.enable_default_logging()

class ValueNetwork(nn.Module):
    #def __init__(self, input_size, hidden_size):
    def __init__(self, input_size, hidden_size, num_class, num_layers = 3, use_bias = True, dropout = 0):
        super(ValueNetwork, self).__init__()
        self.input_size = input_size
        self.num_class = num_class
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.model = nn.Sequential( nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, num_class), nn.Softmax())
        self.dropout_layer = nn.Dropout(dropout)
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x, _ = self.lstm(x)
        x = x.contiguous()
        x = x.mean(dim = 1)
        x = self.dropout_layer(x)
        x = self.model(x)
        #x = x.view(batch_size, seq_len, x.size(1))
        return x

class PolicyNetwork(nn.Module):
    #def __init__(self, hidden_size, num_actions):
    def __init__(self, input_size, hidden_size, num_actions, num_layers = 3, use_bias = True, dropout = 0):
        super(PolicyNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.model = nn.Sequential( nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, num_actions), nn.Softmax(dim = 1) )
        self.dropout_layer = nn.Dropout(dropout)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout_layer(x)
        x = x[:,-1,:]
        return self.model(x)

class CoreClassification(nn.Module):
    def __init__(self, hidden_size, num_class):
        super(CoreClassification, self).__init__()
        self.hidden_size = hidden_size
        self.model = nn.Sequential( nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, num_class) )
    def forward(self, state):
        return self.model(state) 

class SkeletonAction(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, num_actions = 16, num_layers = 3, use_bias = True, dropout = 0):
        super(SkeletonAction, self).__init__()
        self.input_size = input_size
        self.hidden_isze = hidden_size
        self.num_class = num_class
        self.num_layers = num_layers
        self.num_actions = num_actions
        self.use_bias = use_bias
        self.dropout = dropout
        rnn_cells_lst = []
        ipt_size = input_size
        for i in range(num_layers):
            rnn_cells_lst.append(nn.LSTMCell(ipt_size, hidden_size))
            ipt_size = hidden_size
        self.rnn_cells = nn.ModuleList(rnn_cells_lst)

        self.linear = nn.Linear(hidden_size, num_class)
        self.dropout_layer = nn.Dropout(dropout)

    def init_weights(self):
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.fc_t.weight.data.uniform_(-0.1, 0.1)
        self.fc_t.bias.data.fill_(0)
        self.fc_r.weight.data.uniform_(-0.1, 0.1)
        self.fc_r.bias.data.fill_(0)
    
    def forward(self, x, action, h_t_1, c_t_1):
        '''
        x: batch x input_size
        '''
        # Now, run the va lstm first.

        batch_size = x.size(0)
        
        angles = action / self.num_actions * np.pi * 2
    
        # compute the rotation matrix.
        zero = torch.zeros(batch_size)
        one = torch.ones(batch_size)
        if torch.cuda.is_available():
            zero = zero.cuda()
            one = one.cuda()
        zero = Variable(zero) 
        one = Variable(one)

        R_z = []
        R_z.append(torch.stack([ torch.cos(angles), -torch.sin(angles), zero], dim = 1))
        R_z.append(torch.stack([ torch.sin(angles), torch.cos(angles), zero], dim = 1))
        R_z.append(torch.stack([ zero, zero, one], dim = 1))
        R_z = torch.stack(R_z, dim = 1).unsqueeze(1)
        x = x.contiguous()
        x = x.view(batch_size, x.size(-1) // 3, 1, 3)
        x = (R_z * x).sum(dim = -1)
        x = x.view(batch_size, self.input_size)
        x_t = x
        ht,ct = [],[]
        for idx, cell in enumerate(self.rnn_cells):
           hx, cx = cell(x, (h_t_1[idx], c_t_1[idx]))
           x = hx
           ht.append(hx)
           ct.append(cx)
        return ht,ct, x_t

class SkeletonAction_VA_AVG_H(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, num_layers = 3, use_bias = True, dropout = 0):
        super(SkeletonAction_VA_AVG_H, self).__init__()
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
    
    def forward(self, x, mask):
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
        zero = torch.zeros(batch_size, seq_len)
        one = torch.ones(batch_size, seq_len)
        if torch.cuda.is_available():
            zero = zero.cuda()
            one = one.cuda()
        zero = Variable(zero) 
        one = Variable(one)
        R = []
        R_x, R_y, R_z = [], [], []
        R_x.append(torch.stack([ one, zero, zero], dim = 2))
        R_x.append(torch.stack([ zero, torch.cos(angles[:,:,0]), -torch.sin(angles[:,:,0])], dim = 2))
        R_x.append(torch.stack([ zero, torch.sin(angles[:,:,0]), torch.cos(angles[:,:,0])], dim = 2))
        R_x = torch.stack(R_x, dim = 2).unsqueeze(2)

        R_y.append(torch.stack([ torch.cos(angles[:,:,1]), zero, torch.sin(angles[:,:,1])], dim = 2))
        R_y.append(torch.stack([ zero, one, zero], dim = 2))
        R_y.append(torch.stack([ -torch.sin(angles[:,:,1]), zero, torch.cos(angles[:,:,1])], dim = 2))
        R_y = torch.stack(R_y, dim = 2).unsqueeze(2)

        R_z.append(torch.stack([ torch.cos(angles[:,:,2]), -torch.sin(angles[:,:,2]), zero], dim = 2))
        R_z.append(torch.stack([ torch.sin(angles[:,:,2]), torch.cos(angles[:,:,2]), zero], dim = 2))
        R_z.append(torch.stack([ zero, zero, one], dim = 2))
        R_z = torch.stack(R_z, dim = 2).unsqueeze(2)
        
        trans,_ = self.lstm_tran(x) 
        trans = trans.contiguous()
        trans = trans.view((-1, trans.size(-1)))
        trans = self.fc_t(trans)
        trans = trans.view((batch_size, seq_len, 1, trans.size(-1)))
        trans = trans.repeat(1,1,x.size(-1) // 3, 1)
        trans = trans.view((batch_size, seq_len, x.size(-1)))
        x = x - trans 
        x = x.view(batch_size, seq_len, x.size(-1) // 3, 1, 3)
        x = (R_x * x).sum(dim = -1)
        x = x.unsqueeze(3)
        x = (R_y * x).sum(dim = -1)
        x = x.unsqueeze(3)
        x = (R_z * x).sum(dim = -1)
        x = x.view(batch_size, seq_len, self.input_size)
        x,_ = self.lstm(x)
        x = (x * mask).sum(dim = 1) / mask.sum(dim = 1)
        #x = torch.mean(x, dim = 1)
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
