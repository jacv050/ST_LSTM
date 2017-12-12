import argparse
import torch
import pickle
import torch.nn as nn
import numpy as np
import os
import glob
import pdb

from data_loader_lstm_gaussian_aug import get_loader 

from model_lstm_rl import SkeletonAction, ValueNetwork, PolicyNetwork, CoreClassification
from torch.autograd import Variable 
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.distributions import Categorical

import climate
import logging
logging = climate.get_logger(__name__)
climate.enable_default_logging()

def main(args):
    # Build data loader
    if not os.path.isdir(args.model_path):
        os.makedirs(args.model_path)

    data_loader,ds_class = get_loader(args.data_dir,args.batch_size,
                             shuffle=True, num_workers=args.num_workers, ds = args.ds) 

    # Build eval data loader
    if hasattr(ds_class, 'lbl2id'):
        eval_data_loader,_ = get_loader(args.data_dir_test, args.batch_size,
                             shuffle=True, num_workers=args.num_workers, ds = args.ds, lbl2id = ds_class.lbl2id) 
    else:
        eval_data_loader,_ = get_loader(args.data_dir_test, args.batch_size,
                             shuffle=True, num_workers=args.num_workers, ds = args.ds)

    # Loss and Optimizer
    model_base = SkeletonAction(args.input_size, args.hidden_size, args.num_class, args.num_action, args.num_layers, dropout = args.dropout)
    model_value = ValueNetwork( args.hidden_size )
    model_policy = PolicyNetwork( args.hidden_size, args.num_action )
    model_c = CoreClassification( args.hidden_size, args.num_class )
    criterion = nn.CrossEntropyLoss()
    criterion_value = nn.SmoothL1Loss()

    if torch.cuda.is_available():
        model_base.cuda()
        model_value.cuda()
        model_policy.cuda()
        model_c.cuda()
        criterion = criterion.cuda()
        criterion_value = criterion_value.cuda()

    params = list(model_base.parameters()) + list(model_c.parameters()) + list(model_value.parameters()) \
             + list(model_policy.parameters())
    opt = torch.optim.Adam(params, lr=args.learning_rate)
    #opt_value = torch.optim.Adam(model_value.parameters(), lr = args.learning_rate)
    #opt_policy = torch.optim.Adam(model_policy.parameters(), lr = args.learning_rate)
    #opt_c = torch.optim.Adam(model_c.parameters(), lr = args.learning_rate)

    # Load the trained model parameters
    # Now, we try to find the latest encoder and decoder model.
    if os.path.isdir(args.model_path) and os.listdir(args.model_path):
        m_fn = max(glob.glob(os.path.join(args.model_path, 'model*')), key = os.path.getctime)
        if m_fn:
            logging.info("Loading model from %s", m_fn)
            model.load_state_dict(torch.load(m_fn))

    # Train the Models
    total_step = len(data_loader)
    # Initialize some variables.
    h_tensor = torch.zeros(args.batch_size, args.hidden_size)
    if torch.cuda.is_available():
        h_tensor = h_tensor.cuda()
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

            h_tensor.resize_(data.size(0), data.size(1)) 
            init_h = Variable(h_tensor)
            init_hs = [ init_h for i in range( args.num_layers ) ]
            init_cs = init_hs

            zero = torch.zeros(data.size(0),)
            zero = Variable(zero)
            if torch.cuda.is_available():
                zero = zero.cuda()
 
            hs = []
            action_probs = []
            actions = []
            ht, ct = model_base( data[:,0,:], zero, init_hs, init_cs)
            hs.append(ht[-1])

            action_prob = model_policy(ht[-1])
            action_probs.append(action_prob)
            action = Categorical(action_prob)
            action = action.sample()
            actions.append(action)
            
            for j_step in range(1, data.shape[1]):
                ht, ct = model_base( data[:,j_step,:], actions[j_step-1].float(), ht, ct)
                hs.append(ht[-1])
                action_prob = model_policy(ht[-1])
                # We need to smooth the probability
                action = Categorical((action_prob + action_probs[j_step-1]) / 2)
                action = action.sample()
                actions.append(action)
                action_probs.append(action_prob)
            # now, we have finished all the actions.
            # need to bp.
            # the award only returns at the end of the episode.
            hs_t = torch.stack(hs, dim = 1)
            hs_t = (hs_t * mask.unsqueeze(2) ).sum(dim = 1) / mask.sum(dim = 1).unsqueeze(1)
            logits = model_c(hs_t) 
            #log_p = F.log_softmax(logits, dim = 1)
            loss_ent = criterion(logits, lbl)
            #loss = - (mask.squeeze() * log_p[long_idx, lbl.squeeze().data]).sum() / mask.sum()

            pred_lbl = logits.max(dim = 1)[1]
            reward = Variable((pred_lbl.data == lbl.data).float())
            reward = reward.view(data.size(0), 1)
            reward = reward.repeat(1, data.size(1))
            loss_value = []
            loss_policy = []

            actions = torch.stack(actions, dim = 1)
            action_probs = torch.stack(action_probs, dim = 1)
            hs = torch.stack(hs, dim = 1)
            hs = hs.view(-1, hs.size(-1))
            exp_reward = model_value(hs)
            exp_reward = exp_reward.view(data.size(0), data.size(1))
            loss_value =( exp_reward - reward ) ** 2
            loss_value = (loss_value * mask).sum() / mask.sum()
            advantage = reward - Variable(exp_reward.data)
            idx = torch.LongTensor(range(data.size(0)))
            idx = idx.view(data.size(0), 1)
            idx = idx.repeat(1, data.size(1))
            idx = idx.view(data.size(0) * data.size(1))
            if torch.cuda.is_available():
                idx = idx.cuda()
            action_probs = action_probs.view(action_probs.size(0) * action_probs.size(1),action_probs.size(-1))
            actions = actions.view(actions.size(0) * actions.size(1))
            log_prob = action_probs[idx, actions]
            log_prob = log_prob.view(mask.size(0), mask.size(1))
            loss_policy = -torch.log(log_prob + 1e-7) * mask * advantage
            loss_policy = loss_policy.sum() / mask.sum()
            loss = loss_ent + loss_policy + loss_value
            
            # Now we update the value network
            #for j_step, (h, action, action_prob) in enumerate(zip(hs, actions, action_probs)):
            #    # total reward.
            #    target = reward * discount ** (data.size(0) - j_step)
            #    exp_reward = model_value(h)
            #    logging.info('exp_reward: %.4f, target: %.4f', exp_reward.mean().data[0], target.mean().data[0])
            #    l_value = criterion_value(exp_reward, target)
            #    loss_value.append( l_value )
            #    advantage = target - exp_reward
            #    c = Categorical(action_prob)
            #    l_policy = -c.log_prob(action) * advantage
            #    loss_policy.append( l_policy.mean() )
            #loss_value = torch.stack(loss_value).mean()
            #loss_policy = torch.stack(loss_policy).mean()
            #loss += loss_value + loss_policy
           

            opt.zero_grad()
            loss.backward()
            old_norm = clip_grad_norm(params, args.grad_clip)
            opt.step()
            total_train += data.size(0)
            total_correct += (pred_lbl.data.cpu().squeeze() == lbl.data.cpu().squeeze()).sum()
            # Use grad clip.
            # Eval the trained model
            #logging.info('Epoch [%d/%d], Loss: %.4f, reward: %5.4f, loss_value: %5.4f, loss_policy: %5.4f', 
            #                        epoch, args.num_epochs, 
            #                        loss_ent.data[0], reward.mean().data[0], loss_value.data[0], loss_policy.data[0])
            if i_step % args.log_step == 0:
                accuracy = total_correct * 1.0 / total_train
                logging.info('Epoch [%d/%d], Loss: %.4f, reward: %5.4f, loss_value: %5.4f, loss_policy: %5.4f, accuracy: %5.4f', 
                                    epoch, args.num_epochs, 
                                    loss_ent.data[0], reward.mean().data[0], loss_value.data[0], loss_policy.data[0], accuracy)
                #logging.info('Epoch [%d/%d], Loss: %.4f, accuracy: %5.4f, reward: %5.4f'
                #                  ,epoch, args.num_epochs, 
                #                    loss_ent.data[0], accuracy, reward.mean().data[0])

            if i_step % args.eval_step == 0:
                model_base.eval()
                model_c.eval()
                model_policy.eval()
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

                    h_tensor.resize_(data.size(0), data.size(1)) 
                    init_h = Variable(h_tensor)
                    init_hs = [ init_h for i in range( args.num_layers ) ]
                    init_cs = init_hs


                    zero = torch.zeros(data.size(0),)
                    zero = Variable(zero)
                    if torch.cuda.is_available():
                        zero = zero.cuda()
 
                    hs = []
                    action_probs = []
                    actions = []
                    ht, ct = model_base( data[:,0,:], zero, init_hs, init_cs)
                    hs.append(ht[-1])

                    action_prob = model_policy(ht[-1])
                    action_probs.append(action_prob)
                    action = Categorical(action_prob)
                    action = action.sample()
                    actions.append(action)
                    
                    for j_step in range(1, data.shape[1]):
                        ht, ct = model_base( data[:,j_step,:], action.float(), ht, ct)
                        hs.append(ht[-1])
                        action_prob = model_policy(ht[-1])
                        action = Categorical(action_prob)
                        action = action.sample()
                        actions.append(action)
                    # now, we have finished all the actions.
                    # need to bp.
                    # the award only returns at the end of the episode.
                    hs_t = torch.stack(hs, dim = 1)
                    hs_t = (hs_t * mask.unsqueeze(2) ).sum(dim = 1) / mask.sum(dim = 1).unsqueeze(1)
                    logits = model_c(hs_t) 
                    log_p = F.log_softmax(logits, dim = 1)

                    pred_lbl = logits.max(dim = -1)[1].data.cpu()
                    total_num += data.size(0)
                    correct_num += (pred_lbl.squeeze() == lbl.data.cpu().squeeze()).sum()
                    loss = criterion(logits, lbl)
                accuracy = correct_num * 1.0 / total_num
                logging.info('Validating [%d], Loss: %.4f, accuracy: %.4f' ,epoch,
                                loss.data[0], accuracy)
         
                model_base.train()
                model_c.train()
                model_policy.eval()
                

        accuracy = total_correct * 1.0 / total_train
        logging.info('Epoch [%d/%d], Loss: %.4f, accuracy: %5.4f, reward: %5.4f'
                          ,epoch, args.num_epochs, 
                            loss_ent.data[0], accuracy, reward.mean().data[0])
                # Save the models
        #if epoch % 10 == 0:
        #    torch.save(model.state_dict(), 
        #        os.path.join(args.model_path, 
        #                'model-%d.pkl' %(epoch+1)))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models_lstm/' ,
                        help='path for saving trained models')
    # Model parameters
    parser.add_argument('--data_dir', type = str, default = './data/NTURGBD/skeletons_3d_train')
    parser.add_argument('--data_dir_test', type = str, default = './data/NTURGBD/skeletons_3d_train_val')
    parser.add_argument('--input_size', type=int , default=75,
                        help='dimension of input skeleton size(default 3d)')
    parser.add_argument('--ds', type = str, default = 'NTU')
    parser.add_argument('--hidden_size', type=int , default=100,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=3,
                        help='number of layers in lstm')
    parser.add_argument('--num_class', type = int, default = 10, help = 'number of action classes')
    parser.add_argument('--num_action', type = int, default = 16, help = 'number of action')
    parser.add_argument('--use_bias', action='store_true', help = 'use the bias or not in lstm.')
    parser.add_argument('--dropout', type = float, default = 0.5)
    parser.add_argument('--grad_clip', type = float, default = 1.0)
    parser.add_argument('--log_step', type = int, default = 1)
    parser.add_argument('--eval_step', type = int, default = 1)
    parser.add_argument('--save_step', type = int, default = 100)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
