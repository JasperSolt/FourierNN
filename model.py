import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from hyperparams import Model_Hyperparameters as hp

class Fourier_NN(nn.Module):
    def __init__(self, input_channels=hp.INPUT_CHANNELS):
        super(Fourier_NN, self).__init__()
        
        self.layer_dict = hp.LAYER_DICT
        for name, layer in self.layer_dict.items():
            self.add_module(name, layer)
        self.input_channels = input_channels

    # Forward propagation of some batch x. 
    def forward(self, x):
        layer_output = x
        for name, layer in self.layer_dict.items():
            layer_output = layer(layer_output)
        return layer_output

#train func
def train(dataloader, model, device, loss_fn=hp.LOSS_FN):
    #set optimizer
    optimizer = hp.optimizer(model)
    
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)

        predflat = torch.flatten(pred)
        yflat = torch.flatten(y)
        loss = loss_fn(predflat, yflat)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#test func
def test(dataloader, model, device, loss_fn=hp.LOSS_FN):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, total_error = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            total_error += torch.sum(torch.abs(pred - y))
    print("Average loss: {}".format(test_loss / size))
    print("Average error: {}".format(total_error / size))
    
def predict(dataloader, model, device, loss_fn=hp.LOSS_FN, pred_save_path=hp.MODEL_PATH, pred_save_name=hp.MODEL_NAME):
    size = len(dataloader.dataset)
    model.eval()
    pred, labels = torch.tensor([]).to(device), torch.tensor([]).to(device)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = torch.cat((pred, model(X)),dim=1)
            labels = torch.cat((labels, y),dim=1)
            print(pred.shape)
    f = pred_save_path + "/pred_" + pred_save_name
    np.savez('{}.npz'.format(f),targets=labels.cpu().numpy(),predictions=pred.cpu().numpy())

    
def save(model, model_save_path=hp.MODEL_PATH, model_save_name=hp.MODEL_FILENAME):
    # save trained model
    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)
    f = model_save_path + "/" + model_save_name
    print("Saving PyTorch Model State to {}...".format(f))
    torch.save(model.state_dict(), f)
    print("Model Saved.")

def load(model, model_load_path=hp.MODEL_PATH, model_load_name=hp.MODEL_FILENAME):
    f = model_load_path + "/" + model_load_name
    if os.path.isfile(f):
        print("Loading model state from {}".format(f))
        model.load_state_dict(torch.load(f))
        print("Model loaded.")
    else:
        print("Cannot find model path!")

