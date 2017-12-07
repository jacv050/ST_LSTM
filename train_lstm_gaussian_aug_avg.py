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
    if not os.path.isdir(args.model_path):
        os.makedirs(args.model_path)

    data_loader,ds_class = get_loader(args.data_dir, args.batch_size,
                             shuffle=True, num_workers=args.num_workers, ds = args.ds) 

    # Build eval data loader
    if hasattr(ds_class, 'lbl2id'):
        eval_data_loader,_ = get_loader(args.data_dir_test, args.batch_size,
                             shuffle=True, num_workers=args.num_workers, ds = args.ds, lbl2id = ds_class.lbl2id) 
    else:
        eval_data_loader,_ = get_loader(args.data_dir_test, args.batch_size,
                             shuffle=True, num_workers=args.num_workers, ds = args.ds)
 
    model = SkeletonAction(args.input_size, args.hidden_size, args.num_class, args.num_layers, args.use_bias, args.dropout)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model.cuda()
        criterion = criterion.cuda()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Load the trained model parameters
    # Now, we try to find the latest encoder and decoder model.
    if os.path.isdir(args.model_path) and os.listdir(args.model_path):
        m_fn = max(glob.glob(os.path.join(args.model_path, 'model*')), key = os.path.getctime)
        if m_fn:
            model.load_state_dict(torch.load(m_fn))

    # Train the Models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        total_train = 0
        total_correct = 0
        total_train_2 = 0
        total_correct_2 = 0
        for i_step, (lbl, data, length) in enumerate(data_loader):
            # Set mini-batch dataset
            lbl = Variable(lbl.squeeze())
            data = Variable(data)
            mask = torch.zeros(data.size(0), data.size(1))
            for i,m in zip(length, mask):
                m[0:i[0]] = 1
            mask = Variable(mask)
            if torch.cuda.is_available():
                lbl = lbl.cuda()
                data = data.cuda()
                mask = mask.cuda()
            model.zero_grad()
            mask = mask.unsqueeze(2)
            opt = model(data, mask)
            # compute accuracy.        
            pred_lbl = opt.max(dim = -1)[1].data.cpu()
            total_train += data.size(0)
            total_correct += (pred_lbl.squeeze() == lbl.data.cpu().squeeze()).sum()
            loss = criterion(opt, lbl)
            loss.backward()
            optimizer.step()
            # Eval the trained model
            if i_step % args.eval_step == 0:
                model.eval()
                total_num = 0
                correct_num = 0
                correct_num2 = 0
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
                accuracy = correct_num * 1.0 / total_num

                logging.info('Validating [%d], Loss: %.4f, accuracy: %.4f', epoch,
                                loss.data[0], accuracy)
         
                model.train()

        accuracy = 1.0 * total_correct / total_train 
        logging.info('Epoch [%d/%d], Loss: %.4f, accuracy: %5.4f'
            ,epoch, args.num_epochs, 
            loss.data[0], accuracy)
                # Save the models
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 
                os.path.join(args.model_path, 
                        'model-%d.pkl' %(epoch+1)))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models_lstm/' ,
                        help='path for saving trained models')
    # Model parameters
    parser.add_argument('--data_dir', type = str, default = './data/NTURGBD/skeletons_3d_train')
    parser.add_argument('--data_dir_test', type = str, default = './data/NTURGBD/skeletons_3d_train_val')
    parser.add_argument('--input_size', type=int , default=45,
                        help='dimension of input skeleton size(default 3d)')
    parser.add_argument('--ds', type = str, default = 'UTKinect')
    parser.add_argument('--hidden_size', type=int , default=128,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=2,
                        help='number of layers in lstm')
    parser.add_argument('--num_class', type = int, default = 10, help = 'number of action classes')
    parser.add_argument('--use_bias', action='store_true', help = 'use the bias or not in lstm.')
    parser.add_argument('--dropout', type = float, default = 0.5)
    parser.add_argument('--log_step', type = int, default = 1)
    parser.add_argument('--eval_step', type = int, default = 1)
    parser.add_argument('--save_step', type = int, default = 100)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=56)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
