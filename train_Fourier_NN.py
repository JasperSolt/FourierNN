import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from EoR_Dataset import EORImageDataset_LaPlante
from model import Fourier_NN, train, test, save
from hyperparams import Model_Hyperparameters as hp

# training & testing datasets
train_data = EORImageDataset_LaPlante(train=True)
test_data = EORImageDataset_LaPlante(train=False)

# training & testing dataloaders
train_dataloader = DataLoader(train_data, batch_size=hp.BATCHSIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=hp.BATCHSIZE, shuffle=True)

# find our device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# train loop
model = Fourier_NN(hp.INPUT_CHANNELS).to(device)
for t in range(hp.EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model)
    test(test_dataloader, model)
print("Done!")

save(model)

hp.save_hyparam_summary()
