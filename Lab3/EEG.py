import dataloader
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.data import DataLoader
import torch.utils.data as Data
import copy
import os
import matplotlib.pyplot as plt
import pandas as pd

best_weight = {'ReLU':None,'LeakyReLU':None,'ELU':None}
best_acc = {'ReLU':0.0,'LeakyReLU':0.0,'ELU':0.0}


class EEG(nn.Module):
    def __init__(self, activation = nn.ELU(alpha = 1.0)): 
        super(EEG, self).__init__()
        # network layer
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseconv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            # nn.Dropout(p=0.25)
        )
        self.separableconv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            # nn.Dropout(p=0.25)
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )


    def forward(self, x):
        # forward propagation
        x = self.firstconv(x)
        x = self.depthwiseconv(x)
        x = self.separableconv(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        # output = F.log_softmax(x, dim=1)
        return x


    # def train_model(model, criterion, optimizer, dataloaders, num_epochs):
        
def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.long)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss = loss.item()
        train_loss += loss

    correct /= size
    train_loss /= size
    print(f"Train: \n Accuracy: {(100*correct):>0.1f}%, loss: {loss:>7f}")
    return correct

def test_loop(dataloader, model, loss_fn, device, name):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            
            X = X.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.long)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, loss: {test_loss:>8f} \n")
    if correct > best_acc[name]:
        best_acc[name] = correct
        best_weight[name] = copy.deepcopy(model.state_dict())
    return correct


if __name__ == '__main__':
    activations = {'ReLU':nn.ReLU(),'LeakyReLU':nn.LeakyReLU(),'ELU':nn.ELU()}
    # best_weight = {'ReLU':None,'LeakyReLU':None,'ELU':None}
    # best_acc = {'ReLU':0.0,'LeakyReLU':0.0,'ELU':0.0}
    learning_rate = 1e-3
    BATCH_SIZE = 64
    epochs = 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    # print(torch.__version__)
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    
    # model.load_state_dict(torch.load("model.pth"))
    
    x_train = torch.tensor(train_data)
    y_train = torch.tensor(train_label)
    x_test = torch.tensor(test_data)
    y_test = torch.tensor(test_label)

    train_dataset = Data.TensorDataset(x_train,y_train)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    test_dataset = Data.TensorDataset(x_test,y_test)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size = BATCH_SIZE, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    
    # load
    # model.load_state_dict(torch.load(os.path.join('eeg models','ReLU.pt')))
    # acc = test_loop(test_loader, model, loss_fn, device, name)
    
    output_acc = []
    for name,activation in activations.items():
        train_acc = []
        test_acc = []
        print(activation)
        
        model = EEG(activation).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            acc = train_loop(train_loader, model, loss_fn, optimizer, device)
            train_acc.append(acc*100)
            acc = test_loop(test_loader, model, loss_fn, device, name)
            # print(best_acc)
            # best_weight[name] = best_weight
            # buf = best_acc
            test_acc.append(acc*100)
        output_acc.append(train_acc)
        output_acc.append(test_acc)
    print("Done!")

    # for name,model_wts in best_weight.items():
    #     torch.save(model_wts,os.path.join('eeg models',name+'.pth'))
    # print(output_acc)
    # torch.save(model.state_dict(), "model.pth")
    # print("Saved PyTorch Model State to model.pth")

    data = {
        # "epoch": range(1,epochs+1),
        "ReLU_train": output_acc[0],
        "ReLU_test": output_acc[1],
        "LeakyReLU_train": output_acc[2],
        "LeakyReLU_test": output_acc[3],
        "ELU_train": output_acc[4],
        "ELU_test": output_acc[5]
    }
    
    name = ["ReLU_train","ReLU_test","LeakyReLU_train","LeakyReLU_test","ELU_train","ELU_test"]
    for i in range(len(output_acc)):
        value = max(output_acc[i])
        print(f"Max {name[i]} Accuracy: {value}%")

    df = pd.DataFrame(data)
    df.to_csv("result.csv",index = False)
    




    
