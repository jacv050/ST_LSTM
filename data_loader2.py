import torch
import torch.utils.data as data
import os
import pickle
import random
import pdb
import numpy as np

def __loaddata(root):
    dict_lbl2id = {}
    data = []
    for r, subdirs, fns in os.walk(root):
        for fn in fns:
            ffn = os.path.join(r, fn)
            lbl = fn.split('_')[0]
            if lbl not in dict_lbl2id:
                dict_lbl2id[lbl] = len(dict_lbl2id)
            if ffn.endswith('npy'):
                idata = np.load(ffn)
            else:
                idata = np.loadtxt(ffn)
                idata = np.stack(idata,0)
                idata = np.reshape(idata, (idata.shape[0], 3, idata.shape[1]/3))
                idata = np.swapaxes(idata, 1,2)
            data.append((dict_lbl2id[lbl], idata))
    return dict_lbl2id, data

class UTKinect(data.Dataset):
    """Load the preprocessed UTKInect Dataset."""
    def __init__(self, root, seq_len,lbl2id, data):
        self.root = root
        self.seq_len = seq_len
        self.lbl2id, self.data = lbl2id,data
    
    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        idata = self.data[index]
        lbl = torch.Tensor([idata[0]]).long()
        data = idata[1]
        length = torch.LongTensor(1)
        length[0] = self.seq_len
        if data.shape[0] < self.seq_len:
            # No need to sample
            ndata = np.zeros((self.seq_len, data.shape[1], data.shape[2]))
            ndata[0:data.shape[0],:,:] = data
            data = torch.from_numpy(ndata)
        elif data.shape[0] < 2 * self.seq_len:
            # less than 2 in each frame, just randomly select one.
            idx = range(data.shape[0])
            random.shuffle(idx)
            idx = idx[0:self.seq_len]
            sorted(idx)
            ndata = data[idx, :]
            data = torch.from_numpy(ndata)
        else:
            start_idx = range(0, data.shape[0], data.shape[0] / self.seq_len)
            if len(start_idx) > self.seq_len:
                start_idx = start_idx[0:self.seq_len] # the last one has more data.
            start_idx.append(data.shape[0])
            idx = []
            for i in range(self.seq_len):
                idx.append(random.randrange(start_idx[i], start_idx[i+1]))
            ndata = data[idx,:]
            data = torch.from_numpy(ndata)
        return lbl, data, length

    def __len__(self):
        return len(self.data)

def collate_fn(data):
    lbl, dta, length= zip(*data)
    lbl = torch.stack(lbl, 0)
    dta = torch.stack(dta, 0).float()
    length = torch.stack(length, 0)
    return lbl, dta, length


def get_loader(lbl2id, data, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    uk_data = UTKinect(lbl2id, data)
    data_loader = torch.utils.data.DataLoader(dataset=uk_data, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
