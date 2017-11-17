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

class ST_LSTMCell(nn.Module):

    """A basic LSTM cell."""
    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most codes are stolen from torch.nn.LSTMCell.
        """
        super(ST_LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 5 * hidden_size))
        # This is the temporal dimension.
        self.weight_hh_t = nn.Parameter(
            torch.FloatTensor(hidden_size, 5 * hidden_size))
        # This is the spatial dimension.
        self.weight_hh_s = nn.Parameter(
            torch.FloatTensor(hidden_size, 5 * hidden_size))

        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(5 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        #self.weight_ih.data.set_(
        #    init.orthogonal(torch.FloatTensor(*self.weight_ih.size())))
        #self.weight_ih.data.uniform_(-0.1, 0.1)
        #weight_hh_data = torch.eye(self.hidden_size)
        #weight_hh_data = weight_hh_data.repeat(1, 5)
        #self.weight_hh_t.data.set_(weight_hh_data)
        #self.weight_hh_s.data.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        if self.use_bias:
            self.bias.data.fill_(0)

    def forward(self, x, h_t_1, h_s_1):
        """
        Args:
            x: A (batch, input_size) tensor containing input
                features.
            h_t_1: A tuple (h_t_1, c_t_1), in temporal dimension
            h_s_1: A tuple (h_0, c_0), in spatial dimension
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        batch_size = x.size(0)
        if self.use_bias:
            bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        
            wh_t_b = torch.addmm(bias_batch, h_t_1[0], self.weight_hh_t)
        else:
            wh_t_b = torch.mm(h_t_1[0], self.weight_hh_t)

        wh_s = torch.mm(h_s_1[0], self.weight_hh_s)
        wi = torch.mm(x, self.weight_ih)
        f_t, f_s, i, o, g = torch.split(wh_t_b + wh_s + wi,
                                 split_size=self.hidden_size, dim=1)
        c = torch.sigmoid(i)*torch.tanh(g) + torch.sigmoid(f_t)*h_t_1[1] + \
                torch.sigmoid(f_s)*h_s_1[1]
        h = torch.sigmoid(o)*torch.tanh(c)
        return h,c


class ST_LSTMCellv2(nn.Module):

    """A basic LSTM cell."""
    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most codes are stolen from torch.nn.LSTMCell.
        """
        super(ST_LSTMCellv2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Linear( input_size, 5 * hidden_size, bias = self.use_bias)
        # This is the temporal dimension.
        self.weight_hh_t = nn.Linear(hidden_size, 5 * hidden_size, bias = False)
        self.weight_hh_s = nn.Linear(hidden_size, 5 * hidden_size, bias = False)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        #self.weight_ih.data.set_(
        #    init.orthogonal(torch.FloatTensor(*self.weight_ih.size())))
        #self.weight_ih.data.uniform_(-0.1, 0.1)
        #weight_hh_data = torch.eye(self.hidden_size)
        #weight_hh_data = weight_hh_data.repeat(1, 5)
        #self.weight_hh_t.data.set_(weight_hh_data)
        #self.weight_hh_s.data.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        if self.use_bias:
            self.bias.data.fill_(0)

    def forward(self, x, h_t_1, h_s_1):
        """
        Args:
            x: A (batch, input_size) tensor containing input
                features.
            h_t_1: A tuple (h_t_1, c_t_1), in temporal dimension
            h_s_1: A tuple (h_0, c_0), in spatial dimension
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        batch_size = x.size(0)
        wh_t_b = self.weight_hh_t(h_t_1[0])
        wh_s = self.weight_hh_s(h_s_1[0])
        wi = self.weight_ih(x)
       
        f_t, f_s, i, o, g = torch.split(wh_t_b + wh_s + wi,
                                 split_size=self.hidden_size, dim=1)
        c = torch.sigmoid(i)*torch.tanh(g) + torch.sigmoid(f_t)*h_t_1[1] + \
                torch.sigmoid(f_s)*h_s_1[1]
        h = torch.sigmoid(o)*torch.tanh(c)
        return h,c

class ST_LSTMCell_Trust(nn.Module):

    """A basic LSTM cell."""
    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most codes are stolen from torch.nn.LSTMCell.
        """
        super(ST_LSTMCell_Trust, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 5 * hidden_size))
        # This is the temporal dimension.
        self.weight_hh_t = nn.Parameter(
            torch.FloatTensor(hidden_size, 5 * hidden_size))
        # This is the spatial dimension.
        self.weight_hh_s = nn.Parameter(
            torch.FloatTensor(hidden_size, 5 * hidden_size))

        self.linear = nn.Parameter( torch.FloatTensor(hidden_size * 2, hidden_size))
        self.linear_i = nn.Parameter( torch.FloatTensor(input_size, hidden_size))
        self.lam = 0.5

        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(5 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        #self.weight_ih.data.set_(
        #    init.orthogonal(torch.FloatTensor(*self.weight_ih.size())))
        #self.weight_ih.data.uniform_(-0.1, 0.1)
        #weight_hh_data = torch.eye(self.hidden_size)
        #weight_hh_data = weight_hh_data.repeat(1, 5)
        #self.weight_hh_t.data.set_(weight_hh_data)
        #self.weight_hh_s.data.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        if self.use_bias:
            self.bias.data.fill_(0)

    def forward(self, x, h_t_1, h_s_1):
        """
        Args:
            x: A (batch, input_size) tensor containing input
                features.
            h_t_1: A tuple (h_t_1, c_t_1), in temporal dimension
            h_s_1: A tuple (h_0, c_0), in spatial dimension
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        batch_size = x.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        wh_t_b = torch.addmm(bias_batch, h_t_1[0], self.weight_hh_t)
        wh_s = torch.mm(h_s_1[0], self.weight_hh_s)
        wi = torch.mm(x, self.weight_ih)
        f_t, f_s, i, o, g = torch.split(wh_t_b + wh_s + wi,
                                 split_size=self.hidden_size, dim=1)
        p = torch.tanh( torch.mm(torch.cat((h_t_1[0], h_s_1[0]), dim = 1), self.linear))
        x_ = torch.tanh(torch.mm(x, self.linear_i))
        tao = x_ - p
        tao = torch.exp( - self.lam * tao * tao)
        c = tao *  torch.sigmoid(i)*torch.tanh(g) + ( 1 - tao) *  torch.sigmoid(f_t)*h_t_1[1] + \
            ( 1 - tao) *  torch.sigmoid(f_s)*h_s_1[1]
        h = torch.sigmoid(o)*torch.tanh(c)
        return h,c

class ST_LSTM(nn.Module):

    """A module that runs multiple steps of ST LSTM."""

    def __init__(self, input_size, hidden_size, num_layers=1,
                 use_bias=True, dropout=0):
        super(ST_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = ST_LSTMCellv2(input_size=layer_input_size,
                              hidden_size=hidden_size, use_bias = use_bias)
            #cell = ST_LSTMCell_Trust(input_size=layer_input_size,
            #                  hidden_size=hidden_size, use_bias = use_bias)
            self.cells.append(cell)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for cell in self.cells:
            cell.reset_parameters()

    def forward(self, input_):
        #input_: lenghts x joints x batch_size x input_size
        max_time, joints,  batch_size, _ = input_.size()
        # The initialized state of the h and c
        hx = Variable(input_.data.new(self.num_layers, batch_size, self.hidden_size).zero_())
        hx = (hx, hx)
        layer_output = None
        for layer in range(self.num_layers):
            hx_t = (hx[0][layer,:], hx[1][layer,:])
            layer_output = []
            h_t_prev = []
            h_s_prev = hx_t
            for time in range(max_time):
                h_opts = []
                h_ts_tmp = []
                for joint in range(joints):
                    if time == 0:
                        h_next, c_next = self.cells[layer](input_[time][joint], hx_t, h_s_prev)
                    else:
                        h_next, c_next = self.cells[layer](input_[time][joint], h_t_prev[joint], h_s_prev)
                    h_s_prev = (h_next, c_next)
                    h_ts_tmp.append(h_s_prev)
                    h_opts.append(h_next)
                h_t_prev = h_ts_tmp
                layer_output.append(torch.stack(h_opts, 0))
            layer_output = torch.stack(layer_output, 0)
            input_ = layer_output
        output = layer_output
        return output

class SkeletonAction(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, num_layers = 1, use_bias = True, dropout = 0):
        super(SkeletonAction, self).__init__()
        self.input_size = input_size
        self.hidden_isze = hidden_size
        self.num_class = num_class
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.lstm = ST_LSTM(input_size, hidden_size, num_layers, use_bias, dropout)
        self.linear = nn.Linear(hidden_size, num_class)
    def init_weights(self):
        self.linear.weight.data.uniform_(-0.01, 0.01)
        self.linear.bias.data.fill_(0)
    
    def forward(self, x):
        '''
        x: batch x lengths x joints x input_size
        '''
        batch_size = x.size(0)
        length = x.size(1)
        joints = x.size(2)
        x = x.permute(1,2,0,3)
        x = self.lstm(x)
        x = x.view((-1, x.size(-1)))
        x = self.linear(x)
        x = x.view(length, joints, batch_size, x.size(1))
        x = x.permute(2,0,1,3)
        return x
