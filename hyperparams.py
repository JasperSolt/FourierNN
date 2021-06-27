import torch
from torch import nn
import json
from datetime import datetime

'''
Hyperparameters for the model. You should only have to edit this file between runs.
'''
class Model_Hyperparameters():
    # model metadata
    MODEL_ID = datetime.timestamp(datetime.now())
    MODEL_NAME = "test" + "_" + MODEL_ID
    MODEL_PATH = MODEL_NAME
    MODEL_FILENAME = MODEL_PATH + "/" + MODEL_NAME + ".pth"
    HP_JSON_FILENAME = MODEL_PATH + "/" + "hp_" + MODEL_NAME + ".json"
    DATA_PATH = "../data/shared/LaPlanteSims/v10/t21_snapshots_wedge_transposed.hdf5"
    DESC = "I am a model description"

    # training hyperparameters
    BATCHSIZE = 64
    EPOCHS = 5
    TRAIN_PERCENT = 0.8 #fraction of dataset used in training

    #from dataset
    INPUT_CHANNELS = 30
    N_PARAMS = 3

    # Loss function & optimizer
    LOSS_FN = None #nn.CrossEntropyLoss()
    OPTIMIZER = None #torch.optim.SGD(model.parameters(), lr=1e-3)

    #model architecture
    layer_dict = OrderedDict([
      # batch_size x input_channels x 512 x 512
      ('conv1', nn.Conv2d(INPUT_CHANNELS, 16, 3, padding='same')),
      ('relu1_1', nn.ReLU()),
      ('batch1', nn.BatchNorm2d(16)),
      ('maxpool1', nn.MaxPool2d(2, padding='same')),

      # batch_size x 16 x 256 x 256
      ('conv2', nn.Conv2d(16, 32, 3, padding='same')),
      ('relu1_2', nn.ReLU()),
      ('batch2', nn.BatchNorm2d(32)),
      ('maxpool2', nn.MaxPool2d(2, padding='same')),

      # batch_size x 32 x 128 x 128
      ('conv3', nn.Conv2d(32, 64, 3, padding='same')),
      ('relu1_3', nn.ReLU()),
      ('batch3', nn.BatchNorm2d(64)),
      ('maxpool3', nn.MaxPool2d(2, padding='same')),

      # batch_size x 64 x 64 x 64
      # pytorch doesn't have global pooling layers, so I made the kernel the
      # same dimensions as the input and set padding='valid' (i.e. no padding)
      ('global_avgpool', nn.AvgPool2d(64, padding='valid')),

      # batch_size x 64
      ('drop1', nn.Dropout(0.2)),
      ('dense1', nn.Linear(64, 200)),
      ('relu2_1', nn.ReLU()),

      # batch_size x 200
      ('drop2', nn.Dropout(0.2)),
      ('dense2', nn.Linear(200, 100)),
      ('relu2_2', nn.ReLU()),

      # batch_size x 100
      ('drop3', nn.Dropout(0.2)),
      ('dense3', nn.Linear(100, 20)),
      ('relu2_3', nn.ReLU()),

      # batch_size x 100
      ('output', nn.Linear(20, N_PARAMS))
    ])

    ''' Based on LaPlante 2019:

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())

    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))

    model.add(Dense(Nregressparams))
    print(model.summary())
    return model
    '''
    #other random constants
    KSZ_CONSTANT = 2.7255 * 1e6

    @classmethod
    def save_hyparam_summary(cls, report_path=HP_JSON_FILENAME):
        print("Generating hyperparameter summary at {}".format(report_path))
        with open(report_path, 'w') as file:
            json.dump(Model_Hyperparameters.__dict__.copy(), file)

        '''
        from torchsummary import summary

        summary(<model extending nn.Module>, (3, 224, 224))
        '''
