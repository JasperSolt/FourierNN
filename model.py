import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from accelerate import Accelerator
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

def train(dataloader, model, optimizer, accelerator):
    tot_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        #feed batch through model
        #X, y = X.to(device), y.to(device)
        pred = model(X)
        
        # Compute prediction error
        loss = hp.loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        accelerator.backward(loss) #loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        
    #return the average epoch training loss
    avg_loss = tot_loss / len(dataloader)
    accelerator.print("Average train loss: {}".format(avg_loss))
    return avg_loss

def test(dataloader, model, accelerator):
    model.eval()
    tot_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            #X, y = X.to(device), y.to(device)
            pred = model(X)
            tot_loss += hp.loss_fn(pred, y).item()
    avg_loss = tot_loss / len(dataloader)
    accelerator.print("Average test loss: {}".format(avg_loss))
    return avg_loss
    
def predict(dataloader, model, device, pred_save_dir=hp.MODEL_DIR, pred_save_name=hp.MODEL_NAME):
    model.eval()
    shape = (dataloader.dataset.__len__(), hp.N_PARAMS)
    predictions, labels = np.zeros(shape), np.zeros(shape)
    
    #predict
    print("Predicting on {} samples...".format(len(dataloader.dataset)))
    with torch.no_grad():
        i = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            batch_size = len(X)
            predictions[i : (i + batch_size)] = model(X).cpu()
            labels[i : (i + batch_size)] = y.cpu()
            i += batch_size
    print(predictions)
    print(labels)
    #save prediction
    f = pred_save_dir + "/pred_" + pred_save_name
    print("Saving prediction to {}.npz...".format(f))
    np.savez('{}.npz'.format(f), targets=labels, predictions=predictions)
    print("Prediction saved.")
    
def save(model, accelerator, loss=None, model_save_dir=hp.MODEL_DIR, model_save_name=hp.MODEL_NAME):
    # save trained model
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)
    f = model_save_dir + "/" + model_save_name + ".pth"
    accelerator.print("Saving PyTorch Model State to {}.pth...".format(f))
    #model = accelerator.unwrap_model(model)
    accelerator.save(model.state_dict(), f)
    print("Model Saved.")
    
    #save loss to npz
    if loss:
        fl = model_save_dir + "/loss_" + model_save_name + ".npz"
        print("Saving loss data to {}...".format(fl))
        np.savez(fl, train=loss["train"], test=loss["test"])
        print("Loss data saved.")

def load(model, model_load_dir=hp.MODEL_DIR, model_load_name=hp.MODEL_NAME):
    f = model_load_dir + "/" + model_load_name + ".pth"
    if os.path.isfile(f):
        print("Loading model state from {}".format(f))
        model.load_state_dict(torch.load(f))
        print("Model loaded.")
    else:
        print("Cannot find model path!")
        


