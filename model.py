import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from hyperparams import Model_Hyperparameters as hp

class Fourier_NN(nn.Module):
    def __init__(self, input_channels=hp.INPUT_CHANNELS):
        super(Fourier_NN, self).__init__()
        self.model = nn.Sequential(hp.layer_dict)
        self.input_channels = input_channels

    # Forward propagation of some batch x. called by model(x)
    def forward(self, x):
        return self.model(x)

#train func
def train(dataloader, model, loss_fn=hp.LOSS_FN, optimizer=hp.OPTIMIZER):
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

def save(model, model_save_path=hp.MODEL_FILENAME):
    # save trained model
    print("Saving PyTorch Model State to {}...".format(model_save_path))
    torch.save(model.state_dict(), model_save_path)
    print("Model Saved.")

def load(model, model_load_path=hp.MODEL_FILENAME):
    print("Loading PyTorch Model State from {}".format(model_load_path))
    return model.load_state_dict(torch.load())
