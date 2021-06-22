import torch
from torch import nn

'''
Hyperparameters for the model. You should only have to edit this file between runs.
'''
class Model_Hyperparameters():
    # model metadata
    MODEL_NAME = "test"
    DATA_PATH = "../data/shared/LaPlanteSims/v10/t21_snapshots_wedge_transposed.hdf5"
    DESC = "I am a model description"
    
    # training hyperparameters
    BATCHSIZE = 64
    EPOCHS = 5
    TRAIN_PERCENT = 0.8 #fraction of dataset used in training

    # Loss function & optimizer
    loss_fn = None #nn.CrossEntropyLoss()
    optimizer = None #torch.optim.SGD(model.parameters(), lr=1e-3)
    
    #other random constants
    KSZ_CONSTANT = 2.7255 * 1e6

    @classmethod
    def save_hyparam_summary(cls, report_path="model_hyperparams.txt"):
        print("Generating hyperparameter summary at {}".format(report_path))
        #TODO

