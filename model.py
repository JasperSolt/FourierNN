import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from hyperparams import Model_Hyperparameters as hp

class Fourier_NN(nn.Module):
    def __init__(self):
        super(Fourier_NN, self).__init__()
        
        self.input_channels = 30
        #TODO
        
        self.layer_dict = OrderedDict([
          ('conv1', nn.Conv2d(self.input_channels, 16, 3)),
          ('relu1', nn.ReLU()),
          ('batch1', nn.BatchNorm1d(16)),
        ]) 
        
        ''' LaPlante 2019:
        
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
        
        #self.stack = nn.Sequential(<list of layers>)
        
    # Forward propagation of some batch x. called by model(x)
    def forward(self, x):
        return self.stack(self.l1(x))
    
    

    
#train func
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#test func
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    
    
'''
from torchsummary import summary

summary(<model extending nn.Module>, (3, 224, 224))
'''