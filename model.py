import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import h5py

KSZ_CONSTANT = 2.7255 * 1e6

'''
A custom dataset loader specifically for our LaPlanteSim files. This can be changed later if/when we use new sims.

Loads all 21cm data into a tensor of size (limit_len,512,512,30). Loads labels into tensor of size (limit_len,3)

Currently, I'm just loading all images simultaneously, for speed. 
This uses a ton of memory but it shouldn't be a problem considering Oscar's resources? 
'''
class EORImageDataset_LaPlante(Dataset):
    
    '''
    Load data at initialization. Override from Dataset superclass
    '''
    def __init__(self, path="../data/shared/LaPlanteSims/v10/t21_snapshots_wedge_transposed.hdf5", limit_len=None, fourier=False):
        #load data
        if not fourier:
            with h5py.File(path,"r") as h5f:
                #self.ksz = torch.tensor(h5f["Data/ksz_snapshots"][:limit_len]) * KSZ_CONSTANT
                self._21cm = torch.tensor(h5f['Data']['t21_snapshots_transposed'][:limit_len])
                self.labels = torch.tensor(h5f["Data/snapshot_labels"][:limit_len,:3]) #First three labels: dur, mdpt, meanz
                self.redshifts = torch.tensor(h5f['Data']['layer_redshifts'])
        else:
            #TODO
            with h5py.File(path,"r") as h5f:
                #self.ksz = torch.tensor(h5f["Data/ksz_snapshots"][:limit_len]) * KSZ_CONSTANT
                self._21cm = torch.tensor(h5f['Data']['t21_snapshots_transposed'][:limit_len])
                self.labels = torch.tensor(h5f["Data/snapshot_labels"][:limit_len,:3]) #First three labels: dur, mdpt, meanz
                self.redshifts = torch.tensor(h5f['Data']['layer_redshifts'])
        
        #Confirm shape
        assert(len(self.labels) == len(self._21cm))
        print("Data Tensor shape: {}".format(list(self._21cm.shape)))
        print("Labels Tensor shape: {}".format(list(self.labels.shape)))

    '''
    Override from Dataset
    '''
    def __len__(self):
        return len(self.labels)

    '''
    Override from Dataset
    '''
    def __getitem__(self, idx):
        return self._21cm[idx], self.labels[idx]

data = EORImageDataset_LaPlante(limit_len=10)
for i in range(10):
    print(data.__getitem__(i)[0].shape)
    print(data.__getitem__(i)[1].shape)