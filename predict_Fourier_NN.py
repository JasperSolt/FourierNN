import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from EoR_Dataset import EORImageDataset_LaPlante
from model import Fourier_NN, predict, load
from hyperparams import Model_Hyperparameters as hp
from accelerate import Accelerator

accelerator = Accelerator()

test_data = EORImageDataset_LaPlante(train=False, limit_len=hp.N_SAMPLES)
test_dataloader = DataLoader(test_data, batch_size=hp.BATCHSIZE, shuffle=True)

model = Fourier_NN()
load(model, accelerator)

model, test_dataloader = accelerator.prepare(model, test_dataloader)

predict(test_dataloader, model, accelerator)
