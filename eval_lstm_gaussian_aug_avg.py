import argparse
import torch
import pickle
import torch.nn as nn
import numpy as np
import os
import glob
import pdb

#from data_loader_lstm import get_loader 
from data_loader_lstm_gaussian_aug import get_loader 
from model_lstm import SkeletonAction_AVG_H as SkeletonAction
from torch.autograd import Variable 
import torch.nn.functional as F

import climate
import logging
logging = climate.get_logger(__name__)
climate.enable_default_logging()


def main(args):
    # Build data loader

    eval_data_loader,_ = get_loader(args.data_dir_test, args.batch_size,
                             shuffle=False, num_workers=args.num_workers, ds = args.ds)
 
    model = SkeletonAction(args.input_size, args.hidden_size, args.num_class, args.num_layers, args.use_bias, args.dropout)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
        criterion = criterion.cuda()
    m_fn = args.model_fn
    model.load_state_dict(torch.load(m_fn))

    # Train the Models
    model.eval()
    total_num = 0
    correct_num = 0
    for k_step, (lbl, data, length) in enumerate(eval_data_loader):
        lbl = Variable(lbl.squeeze())
        data = Variable(data)
        mask = torch.zeros(data.size(0), data.size(1))
        for i,m in zip(length, mask):
            m[0:i[0]] = 1
        if torch.cuda.is_available():
            lbl = lbl.cuda()
            data = data.cuda()
            mask = mask.cuda()
    
        mask = Variable(mask)
        mask = mask.unsqueeze(2)
        model.zero_grad()
        opt = model(data, mask)
        pred_lbl = opt.max(dim = -1)[1].data.cpu()
        total_num += data.size(0)
        correct_num += (pred_lbl.squeeze() == lbl.data.cpu().squeeze()).sum()
        loss = criterion(opt, lbl)
        logging.info('[%d/%d]', k_step, len(eval_data_loader))
    accuracy = correct_num * 1.0 / total_num

    logging.info('Validating, Loss: %.4f, accuracy: %.4f',
                    loss.data[0], accuracy)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models_lstm/' ,
                        help='path for saving trained models')
    # Model parameters
    parser.add_argument('--data_dir_test', type = str, default = './data/NTURGBD/skeletons_3d')
    parser.add_argument('--input_size', type=int , default=75,
                        help='dimension of input skeleton size(default 3d)')
    parser.add_argument('--ds', type = str, default = 'NTU')
    parser.add_argument('--hidden_size', type=int , default=128,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=2,
                        help='number of layers in lstm')
    parser.add_argument('--num_class', type = int, default = 10, help = 'number of action classes')
    parser.add_argument('--use_bias', action='store_true', help = 'use the bias or not in lstm.')
    parser.add_argument('--dropout', type = float, default = 0.5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_fn', type=str)
    args = parser.parse_args()
    print(args)
    main(args)
