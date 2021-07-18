import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from hyperparams import Model_Hyperparameters as hp

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
    def __init__(self, train, fourier=False, path=hp.DATA_PATH, train_percent=hp.TRAIN_PERCENT, limit_len=None):
        #load data
        if not fourier:
            with h5py.File(path,"r") as h5f:
                print("Loading data from {}...".format(path))
                size = len(h5f["Data/snapshot_labels"][:limit_len,0])
                train_size = int(train_percent * size)

                # train_percent fraction of samples = training set.
                begin, end = 0, train_size
                # remaining samples = testing set
                if not train:
                    begin, end = train_size, size

                self._21cm = torch.tensor(h5f['Data/t21_snapshots'][begin:end])
                self.labels = torch.tensor(h5f["Data/snapshot_labels"][begin:end,:hp.N_PARAMS]) #First three labels: dur, mdpt, meanz
                '''
                self.ksz = torch.tensor(h5f["Data/ksz_snapshots"][begin:end]) * KSZ_CONSTANT
                self.redshifts = torch.tensor(h5f['Data/layer_redshifts'])
                '''

        #else: LOAD FOURIER IMAGES TODO

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
    print("___ Training set: ___")
    data = EORImageDataset_LaPlante(limit_len=10, train=True)
    print("Length: {}".format(data.__len__()))
    #for i in range(data.__len__()):
    #    print(data.__getitem__(i)[0].shape)
    #    print(data.__getitem__(i)[1].shape)

    print("___ Testing set: ___")
    data = EORImageDataset_LaPlante(limit_len=10, train=False)
    print("Length: {}".format(data.__len__()))
    #for i in range(data.__len__()):
    #    print(data.__getitem__(i)[0].shape)
    #    print(data.__getitem__(i)[1].shape)

