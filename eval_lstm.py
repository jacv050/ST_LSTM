import argparse
import torch
import cPickle as pickle
import torch.nn as nn
import numpy as np
import os
import glob
import pdb

from data_loader_lstm import get_loader 
from model_lstm import SkeletonAction
from torch.autograd import Variable 
import torch.nn.functional as F

import climate
import logging
logging = climate.get_logger(__name__)
climate.enable_default_logging()


def main(args):
    # Build data loader

    # Build eval data loader
    data_loader = get_loader(args.data_dir_test, args.seq_len, args.batch_size,
                             shuffle=True, num_workers=args.num_workers, ds = args.ds) 
    
    model = SkeletonAction(args.input_size, args.hidden_size, args.num_class, args.num_layers, args.use_bias, args.dropout)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model.cuda()
        criterion = criterion.cuda()
    
    model_dict = model.state_dict()
    for k,v in model_dict.items():
        print k
    sys.exit()
    model.load_state_dict(torch.load(args.model_fn))
    # Train the Models
    total_step = len(data_loader)
    total_train = 0
    total_correct = 0

    model.eval()
    total_num = 0
    correct_num = 0
    correct_num2 = 0
    for k_step, (lbl, data, length) in enumerate(data_loader):
        lbl = Variable(lbl)
        data = Variable(data)
        mask = torch.zeros(data.size(0), data.size(1))
        for i,m in zip(length, mask):
            m[0:i[0]] = 1
        if torch.cuda.is_available():
            lbl = lbl.cuda()
            data = data.cuda()
            mask = mask.cuda()
    
        mask = Variable(mask)
        model.zero_grad()
        opt = model(data)
        pred_lbl = opt.max(dim = -1)[1].data.cpu()
        gt_lbl = lbl.data.cpu()
        cnt = torch.LongTensor(lbl.size(0), args.num_class).zero_()
        for i in range(pred_lbl.size(0)):
            for j in range(length[i][0]):
                cnt[i][pred_lbl[i,j]] += 1
        cnt = cnt.max(dim = -1)[1]
        total_num += data.size(0)
        correct_num += (cnt.squeeze() == gt_lbl.squeeze()).sum()
        prob = F.softmax(opt.view(opt.size(0) * opt.size(1), opt.size(2)))
        prob = prob.view(opt.size(0), opt.size(1), opt.size(2))
        prob = prob.sum(dim = 1)
        pred_lbl = prob.max(dim = -1)[1].data.cpu()
        correct_num2 += (pred_lbl.squeeze() == gt_lbl.squeeze()).sum()

        lbl = lbl.squeeze().unsqueeze(1)
        lbl = lbl.repeat(1, opt.size(1)).contiguous()
        lbl = lbl.view(lbl.size(0) * lbl.size(1)) 
        opt = opt.contiguous()
        opt = opt.view(opt.size(0) * opt.size(1), opt.size(2))
        #loss = criterion(opt, lbl)
        log_p = F.log_softmax(opt)
        loss = - (mask.squeeze() * log_p[torch.LongTensor(range(opt.size(0))).cuda(), lbl.squeeze().data]).sum() / mask.sum()
    accuracy = correct_num * 1.0 / total_num
    accuracy2 = correct_num2 * 1.0 / total_num
    logging.info('Loss: %.4f, accuracy: %.4f, accuracy2: %.4f',
                    loss.data[0], accuracy, accuracy2) 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_fn', type=str,
                        help='path for saving trained models')
    # Model parameters
    parser.add_argument('--data_dir_test', type = str, default = './data/utkinect/joints_processed_shrink_rm_center_test/')
    parser.add_argument('--input_size', type=int , default=60,
                        help='dimension of input skeleton size(default 3d)')
    parser.add_argument('--ds', type = str, default = 'NTU')
    parser.add_argument('--seq_len', type=int , default=10, help = 'default length of the sequence for training.')
    parser.add_argument('--hidden_size', type=int , default=128,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1,
                        help='number of layers in lstm')
    parser.add_argument('--num_class', type = int, default = 10, help = 'number of action classes')
    parser.add_argument('--use_bias', action='store_true', help = 'use the bias or not in lstm.')
    parser.add_argument('--dropout', type = float, default = 0.5)
    parser.add_argument('--batch_size', type=int, default=56)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    print(args)
    main(args)
