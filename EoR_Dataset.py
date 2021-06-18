import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import h5py

KSZ_CONSTANT = 2.7255 * 1e6

'''
A custom dataset loader specifically for our LaPlanteSim files. This can be changed later if/when we use new sims.

Initializing loads all 21cm data from an h5py file into a tensor of size (limit_len,512,512,30) and loads labels into tensor of size (limit_len,3). Also stores the redshift range as a tensor.

Currently, I'm loading all images into memory simultaneously, for speed. 
This uses a ton of memory but shouldn't be a problem considering Oscar's resources? I can request jobs up to 190G, 
and a dataset like this needs approx. 150G in memory. If our datasets get bigger I will implement some sort of cache.
'''
class EORImageDataset_LaPlante(Dataset):
    
    '''
    Load data at initialization. Override from Dataset superclass
    '''
    def __init__(self, path="../data/shared/LaPlanteSims/v10/t21_snapshots_wedge_transposed.hdf5", \
                 train=True, train_percent=0.8, limit_len=None, fourier=False):
        #load data
        if not fourier:
            with h5py.File(path,"r") as h5f:
                size = len(h5f["Data/snapshot_labels"][:limit_len,:3])
                train_size = int(train_percent * size)

                #First <train_percent> fraction of samples = training set
                #Remaining samples = testing set
                begin, end = 0, 0
                if train:
                    begin = 0
                    end = train_size
                if not train:
                    begin = train_size
                    end = size
                
                self._21cm = torch.tensor(h5f['Data/t21_snapshots_transposed'][begin:end])
                self.labels = torch.tensor(h5f["Data/snapshot_labels"][begin:end,:3]) #First three labels: dur, mdpt, meanz
                '''
                self.ksz = torch.tensor(h5f["Data/ksz_snapshots"][begin:end]) * KSZ_CONSTANT
                self.redshifts = torch.tensor(h5f['Data/layer_redshifts'])
                '''
                
        #else: LOAD FOURIER IMAGES
        
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

    
    
if __name__ == "__main__":
    print("Training set:")
    data = EORImageDataset_LaPlante(limit_len=10)
    print("Length: {}".format(data.__len__()))
    for i in range(data.__len__()):
        print(data.__getitem__(i)[0].shape)
        print(data.__getitem__(i)[1].shape)
        
    print("Testing set")
    data = EORImageDataset_LaPlante(limit_len=10, train=False)
    print("Length: {}".format(data.__len__()))
    for i in range(data.__len__()):
        print(data.__getitem__(i)[0].shape)
        print(data.__getitem__(i)[1].shape)