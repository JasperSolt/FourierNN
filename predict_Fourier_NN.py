import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from EoR_Dataset import EORImageDataset_LaPlante
from model import Fourier_NN, predict, load
from hyperparams import Model_Hyperparameters as hp

test_data = EORImageDataset_LaPlante(train=False, limit_len=hp.N_SAMPLES)
test_dataloader = DataLoader(test_data, batch_size=hp.BATCHSIZE, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = Fourier_NN().to(device)
load(model)

predict(test_dataloader, model, device)
