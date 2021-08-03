import os
from collections import OrderedDict
import json
import jsonpickle
from datetime import datetime
import torch
from torch import nn

'''
These should literally never change
'''
class Constant():
    #random constants
    KSZ_CONSTANT = 2.7255 * 1e6

    #indices for each parameter
    PARAM_DICT = {0:"midpoint", 1:"duration", 2:"meanz"}

'''
Hyperparameters for the model. You should only have to edit this class between runs.
'''
class Model_Hyperparameters():
    # model metadata
    MODEL_ID = str(datetime.timestamp(datetime.now())).replace(".","")
    MODEL_NAME = "multithread_test_1_gpus"
    MODEL_DIR = "models/" + MODEL_NAME
    HP_JSON_FILENAME = "hp_" + MODEL_NAME + ".json"
    DATA_PATH = "../data/shared/LaPlanteSims/v10/t21_snapshots_nowedge.hdf5"
    DESC = "huggingface accelerate test 1 gpus"

    #Multithreading
    N_GPU = 1
    
    # training hyperparameters 
    BATCHSIZE = 32
    EPOCHS = 10
    TRAIN_PERCENT = 0.8 #fraction of dataset used in training
    INITIAL_LR = 0.1 #static learning rate if LR_DECAY = False, or initial learning rate if LR_DECAY = True
    LR_DECAY = False
    DECAY_RT = 1

    #from dataset
    INPUT_CHANNELS = 30
    N_PARAMS = 2
    N_SAMPLES = 200

    # Loss function
    loss_fn = torch.nn.MSELoss()

    #model architecture
    LAYER_DICT = OrderedDict([
      # batch_size x input_channels x 512 x 512
      ('conv1', nn.Conv2d(INPUT_CHANNELS, 16, 3, padding='same')),
      ('relu1_1', nn.ReLU()),
      ('batch1', nn.BatchNorm2d(16)),
      ('maxpool1', nn.MaxPool2d(2)),

      # batch_size x 16 x 256 x 256
      ('conv2', nn.Conv2d(16, 32, 3, padding='same')),
      ('relu1_2', nn.ReLU()),
      ('batch2', nn.BatchNorm2d(32)),
      ('maxpool2', nn.MaxPool2d(2)),

      # batch_size x 32 x 128 x 128
      ('conv3', nn.Conv2d(32, 64, 3, padding='same')),
      ('relu1_3', nn.ReLU()),
      ('batch3', nn.BatchNorm2d(64)),
      ('maxpool3', nn.MaxPool2d(2)),

      # batch_size x 64 x 64 x 64
      # pytorch doesn't have global pooling layers, so I made the kernel the
      # same dimensions as the input
      ('global_avgpool', nn.AvgPool2d(64)),
      ('flat1', nn.Flatten()),

      # batch_size x 64 x 1 x 1
      ('drop1', nn.Dropout(0.2)),
      ('dense1', nn.Linear(64, 200)),
      ('relu2_1', nn.ReLU()),

      # batch_size x 200 x 1 x 1
      ('drop2', nn.Dropout(0.2)),
      ('dense2', nn.Linear(200, 100)),
      ('relu2_2', nn.ReLU()),

      # batch_size x 100 x 1 x 1
      ('drop3', nn.Dropout(0.2)),
      ('dense3', nn.Linear(100, 20)),
      ('relu2_3', nn.ReLU()),

      # batch_size x 100 x 1 x 1
      ('output', nn.Linear(20, N_PARAMS))
    ])

    @classmethod
    def save_hyparam_summary(cls, dirr=MODEL_DIR, report_name=HP_JSON_FILENAME):
        if not os.path.isdir(dirr):
            os.mkdir(dirr)
        print("Generating hyperparameter summary at {}...".format(dirr + "/" + report_name))
        with open(dirr + "/" + report_name, 'w') as file:
            json_encode = jsonpickle.encode(cls.__dict__.copy(), unpicklable=False, indent=4, max_depth=2)
            json.dump(json_encode, file)
        print("Hyperparameter summary saved.")
    
    @classmethod
    def optimizer(cls, model):
        return torch.optim.Adam(model.parameters(), lr=cls.INITIAL_LR)
    
    @classmethod
    def scheduler(cls, opt):
        if cls.LR_DECAY:
            #batches_per_epoch = (cls.N_SAMPLES * cls.TRAIN_PERCENT) / cls.BATCHSIZE
            #lam = lambda epoch: 1 / (1 + cls.DECAY_RT * batches_per_epoch * epoch)
            lam = lambda epoch: 1 / (1 + cls.DECAY_RT * epoch)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lam])
            return scheduler
        return None

    @classmethod
    def save_time(cls, start_time, dirr=MODEL_DIR):
        print("\n* * * * * * * *\nPROCESS TIME: {}\n* * * * * * * *".format(datetime.now() - start_time))
        with open(dirr + "/" + "time.txt", 'w') as file:
            file.write("--- %s seconds ---" % (datetime.now() - start_time))
        
if __name__ == "__main__":
    Model_Hyperparameters.save_hyparam_summary()
    
   