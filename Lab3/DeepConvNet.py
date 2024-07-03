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


class DeepConvNet(nn.Module):
    def __init__(self, activation = nn.ELU(alpha = 1.0)): 
        super(DeepConvNet, self).__init__()

        # network layer
        self.layer0 = nn.Conv2d(1,25,kernel_size=(1,5))
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=(2, 1)),
            nn.BatchNorm2d(25),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5)),
            nn.BatchNorm2d(50),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5)),
            nn.BatchNorm2d(100),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5)),
            nn.BatchNorm2d(200),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        
        
        self.classify = nn.Sequential(
            nn.Linear(8600, 2)
        )


    def forward(self, x):
        # forward propagation
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
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
        
        model = DeepConvNet(activation).to(device)
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

    name = ["ReLU_train","ReLU_test","LeakyReLU_train","LeakyReLU_test","ELU_train","ELU_test"]
    data = {
        # "epoch": range(1,epochs+1),
        "ReLU_train": output_acc[0],
        "ReLU_test": output_acc[1],
        "LeakyReLU_train": output_acc[2],
        "LeakyReLU_test": output_acc[3],
        "ELU_train": output_acc[4],
        "ELU_test": output_acc[5]
    }

    for i in range(len(output_acc)):
        value = max(output_acc[i])
        print(f"Max {name[i]} Accuracy: {value}%")

    df = pd.DataFrame(data)
    df.to_csv("result_deep.csv",index = False)
    
    



    
