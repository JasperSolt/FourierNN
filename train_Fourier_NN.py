import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from accelerate import Accelerator
from datetime import datetime
from EoR_Dataset import EORImageDataset_LaPlante
from model import Fourier_NN, train, test, save, predict, save_loss
from hyperparams import Model_Hyperparameters as hp
from plot_model_results import plot_loss, plot_model_predictions

accelerator = Accelerator()
accelerator.print("Model: {}".format(hp.MODEL_NAME))

#make sure we aren't overwriting
if os.path.isdir(hp.MODEL_DIR):
    accelerator.print("Attempting to overwrite existing model. Please rename current model or delete old model directory.")
else:
    start_time = None
    if accelerator.is_main_process:
        start_time = datetime.now()

    accelerator.print("Loading data...")
    # training & testing datasets
    train_data = EORImageDataset_LaPlante(train=True, limit_len=hp.N_SAMPLES)
    test_data = EORImageDataset_LaPlante(train=False, limit_len=hp.N_SAMPLES)

    # training & testing dataloaders
    train_dataloader = DataLoader(train_data, batch_size=hp.BATCHSIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=hp.BATCHSIZE, shuffle=True)

    # initialize model, optimizer
    model = Fourier_NN() #model = Fourier_NN().to(device)
    optim = hp.optimizer(model)
    
    #set up multithreading
    model, optim, train_dataloader, test_dataloader = accelerator.prepare(model, optim, train_dataloader, test_dataloader)
    
    #initialize scheduler--MUST go after accelerator.prepare
    scheduler = hp.scheduler(optim)

    #train / test loop
    loss = { "train" : [], "test" : [] }
    for t in range(hp.EPOCHS):
        accelerator.print(f"Epoch {t+1}\n-------------------------------")
        loss["train"].append(train(train_dataloader, model, optim, accelerator))
        loss["test"].append(test(test_dataloader, model, accelerator))
        if hp.LR_DECAY:
            scheduler.step()
            accelerator.print("Learning Rate: {}".format(optim.param_groups[0]['lr']))

    save(model, accelerator)

    #none of these save multithreaded components
    if accelerator.is_main_process:
        hp.save_hyparam_summary()
        hp.save_time(start_time)
        save_loss(loss)
        plot_loss(loss)


