from __future__ import division

import argparse
import torch
import pickle
import torch.nn as nn
import numpy as np
import os
import glob
import pdb

from data_loader import get_loader 
from model import SkeletonAction
from torch.autograd import Variable 
import torch.nn.functional as F

import climate
import logging
logging = climate.get_logger(__name__)
climate.enable_default_logging()


def main(args):
    # Build eval data loader
    eval_data_loader = get_loader(args.data_dir_test, args.seq_len, args.batch_size,
                             shuffle=False, num_workers=args.num_workers, ds = args.ds, is_val = False) 
    
    model = SkeletonAction(args.input_size, args.hidden_size, args.num_class, args.num_layers, args.use_bias, args.dropout)


    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model.cuda()
        criterion = criterion.cuda()

    params = model.parameters()
    model.load_state_dict(torch.load(args.model_fn))
    model.eval()
    total_num = 0
    correct_num = 0
    total_correct_2 = 0
    for k_step, (lbl, data, length) in enumerate(eval_data_loader):
        lbl = Variable(lbl)
        data = Variable(data)
        mask = torch.zeros(data.size(0), data.size(1), data.size(2))
        for i,m in zip(length, mask):
            m[0:i[0],:] = 1
        if torch.cuda.is_available():
            lbl = lbl.cuda()
            data = data.cuda()
            mask = mask.cuda()

        mask = Variable(mask)
        opt = model(data)
        pred_lbl = opt.max(dim = -1)[1].data.cpu()
        gt_lbl = lbl.data.cpu()
        cnt = torch.LongTensor(lbl.size(0), args.num_class).zero_()
        for i in range(pred_lbl.size(0)):
            for j in range(length[i][0]):
            #for j in range(pred_lbl.size(1)):
                for k in range(pred_lbl.size(2)):
                    cnt[i][pred_lbl[i,j,k]] += 1
        cnt = cnt.max(dim = -1)[1]
        #print cnt
        #accuracy = (cnt.squeeze() == gt_lbl.squeeze()).sum() * 1.0 / cnt.size(0)
        total_num += data.size(0)
        correct_num += (cnt.squeeze() == gt_lbl.squeeze()).sum()
        
        lbl = lbl.squeeze().unsqueeze(1).unsqueeze(2)
        lbl = lbl.repeat(1, opt.size(1), opt.size(2)).contiguous()
        lbl = lbl.view(lbl.size(0) * lbl.size(1) * lbl.size(2))
        opt = opt.contiguous()
        prob = F.softmax(opt.view(opt.size(0) * opt.size(1) * opt.size(2), opt.size(3)))
        prob = prob.view(opt.size(0), opt.size(1), opt.size(2), opt.size(3))
        prob = prob.sum(dim = 2)
        prob_sum = torch.zeros(prob.size(0), prob.size(2))
        for i in range(prob.size(0)):
            for k in range(prob_sum.size(1)):
                for j in range(length[i][0]):
                    prob_sum[i,k] += prob.data[i,j,k]
        pred_lbl = prob_sum.max(dim = -1)[1].cpu()
        l_correct_2 = (pred_lbl.squeeze() == gt_lbl.squeeze()).sum()
        total_correct_2 += l_correct_2
        
        opt = opt.view(opt.size(0) * opt.size(1) * opt.size(2), opt.size(3))
        #loss = criterion(opt, lbl)
        log_p = F.log_softmax(opt)
        mask = mask.view(opt.size(0), 1)
        loss = - (mask.squeeze() * log_p[torch.LongTensor(range(opt.size(0))).cuda(), lbl.squeeze().data]).sum() / mask.sum()
        logging.info('[%d/%d] processed', k_step, len(eval_data_loader))
    accuracy = correct_num * 1.0 / total_num
    accuracy2 = total_correct_2 * 1.0 / total_num
    logging.info('Accuracy %.4f', accuracy)
    logging.info('Accuracy2 %.4f', accuracy2)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_fn', type=str,
                        help='path for saving trained models')
    # Model parameters
    parser.add_argument('--data_dir_test', type = str, default = './data/utkinect/joints_processed_shrink_rm_center_test/')
    parser.add_argument('--input_size', type=int , default=3,
                        help='dimension of input skeleton size(default 3d)')
    parser.add_argument('--ds', type = str, default = 'UTKinect')
    parser.add_argument('--seq_len', type=int , default=10, help = 'default length of the sequence for training.')
    parser.add_argument('--hidden_size', type=int , default=128,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1,
                        help='number of layers in lstm')
    parser.add_argument('--num_class', type = int, default = 10, help = 'number of action classes')
    parser.add_argument('--use_bias', action='store_true', help = 'use the bias or not in lstm.')
    parser.add_argument('--dropout', type = float, default = 0)
    parser.add_argument('--batch_size', type=int, default=56)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    print(args)
    main(args)
