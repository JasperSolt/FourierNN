import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from accelerate import Accelerator
from hyperparams import Model_Hyperparameters as hp

'''
model class
'''
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

'''
Training function
'''
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

'''
Validation function
'''
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
    
'''
Prediction + save prediction 
'''
def predict(dataloader, model, accelerator, pred_save_dir=hp.MODEL_DIR, pred_save_name=hp.MODEL_NAME):
    model.eval()
    shape = (dataloader.dataset.__len__(), hp.N_PARAMS)
    predictions, labels = np.zeros(shape), np.zeros(shape)
    #predictions, labels = np.array((0,2)), np.array((0,2))
    i = 0
    
    #predict
    accelerator.print("Predicting on {} samples...".format(len(dataloader.dataset)))
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to("cuda"), y.to("cuda")
            batch_pred = model(X)
            batch_size = len(batch_pred)
            
            #gather threads
            #all_batch_pred = accelerator.gather(batch_pred)
            #all_batch_labels = accelerator.gather(y)
            #batch_size, _ = all_batch_pred.shape

            #store
            if accelerator.is_main_process:
                #predictions = np.concatenate((predictions, all_batch_pred.cpu().numpy()), axis=0)
                #labels = np.concatenate((labels, all_batch_labels.cpu().numpy()), axis=0)
                predictions[i : i + batch_size] = batch_pred.cpu()
                labels[i : i + batch_size] = y.cpu()
                i += batch_size
    accelerator.print(predictions.shape)

    #save prediction
    if accelerator.is_main_process:
        f = pred_save_dir + "/pred_" + pred_save_name
        print("Saving prediction to {}.npz...".format(f))
        np.savez('{}.npz'.format(f), targets=labels, predictions=predictions)
        print("Prediction saved.")
    
'''
Save model
'''
def save(model, accelerator, model_save_dir=hp.MODEL_DIR, model_save_name=hp.MODEL_NAME):
    # save trained model
    if accelerator.is_main_process and not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)
    
    f = model_save_dir + "/" + model_save_name + ".pth"
    accelerator.print("Saving PyTorch Model State to {}.pth...".format(f))
    
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), f)
    
    accelerator.print("Model Saved.")

'''
Save loss
'''
def save_loss(loss, model_save_dir=hp.MODEL_DIR, model_save_name=hp.MODEL_NAME):
    fl = model_save_dir + "/loss_" + model_save_name + ".npz"
    print("Saving loss data to {}...".format(fl))
    np.savez(fl, train=loss["train"], test=loss["test"])
    print("Loss data saved.")
    
'''
load model
'''
def load(model, accelerator, model_load_dir=hp.MODEL_DIR, model_load_name=hp.MODEL_NAME):
    f = model_load_dir + "/" + model_load_name + ".pth"
    if os.path.isfile(f):
        accelerator.print("Loading model state from {}".format(f))
        model.load_state_dict(torch.load(f))
        accelerator.print("Model loaded.")
    else:
        accelerator.print("Cannot find model path!")
        


