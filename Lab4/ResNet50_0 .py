import copy
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import dataloader
from torch.utils import data
import pandas as pd
import torch
import torchvision
import numpy as np
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_weight = None
best_acc = 0.0


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """ Resnet50
    """
    model_ft = models.resnet50(pretrained=use_pretrained)

    # if use_pretrained:
    #     set_parameter_requires_grad(model_ft, feature_extract)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    return model_ft


def train(train_loader, test_loader, model, loss_fn, optimizer, epochs, num_class, device, name):
    train_loader_size = len(train_loader.dataset)
    best_weight = None
    best_acc = 0
    print("Train:")
    model.train()

    df = pd.DataFrame()
    # df['epoch'] = range(1, epochs+1)

    acc_train = []
    acc_test = []
    # print(train_loader_size)
    for epoch in range(epochs):
        train_loss, correct = 0, 0
        count = 0
        for images, label in tqdm(train_loader):
            images = images.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            pred = model(images)
            loss = loss_fn(pred, label)
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            train_loss += loss
        correct /= train_loader_size
        correct *= 100
        acc_train.append(correct)
        train_loss /= train_loader_size
        print(f"epoch{epoch:>2d} Accuracy: {(correct):.2f}%, loss: {loss:.4f}")

        "---test---"

        confusion_matrix, test_correct = test(
            test_loader, model, loss_fn, device, num_class)
        acc_test.append(test_correct)

        if test_correct > best_acc:
            best_acc = test_correct
            best_weight = copy.deepcopy(model.state_dict())

    df['Train(w/o pretraining)'] = acc_train
    df['Test(w/o pretraining)'] = acc_test
    torch.save(best_weight, os.path.join('models', f"{name}.pt"))
    model.load_state_dict(best_weight)

    return df


def test(test_loader, model, loss_fn, device, num_class):
    confusion_matrix = np.zeros((num_class, num_class))
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for images, label in test_loader:
            images = images.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            pred = model(images)
            test_loss += loss_fn(pred, label).item()
            predict_class = pred.max(dim=1)[1]
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
            for i in range(len(label)):
                confusion_matrix[int(label[i])][int(predict_class[i])] += 1
    test_loss /= num_batches
    correct /= size
    correct *= 100
    # print(f"Test: \n Accuracy: {(correct):.2f}%, loss: {test_loss:.4f} \n")
    confusion_matrix = confusion_matrix / \
        confusion_matrix.sum(axis=1).reshape(num_class, 1)
    return confusion_matrix, correct


def plot(dataframe1, dataframe2, title):
    fig = plt.figure(figsize=(10, 6))
    for name in dataframe1.columns[1:]:
        plt.plot(range(1, 1+len(dataframe1)), name, data=dataframe1,
                 label=name[4:]+'(w/o pretraining)')
    for name in dataframe2.columns[1:]:
        plt.plot(range(1, 1+len(dataframe2)), name, data=dataframe2,
                 label=name[4:]+'(with pretraining)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy(%)')
    plt.title(title)
    plt.legend()
    return fig


def plot_confusion_matrix(confusion_matrix):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    ax.xaxis.set_label_position('top')
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(i, j, '{:.2f}'.format(
                confusion_matrix[j, i]), va='center', ha='center')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    return fig


if __name__ == '__main__':
    print(torch.cuda.memory_allocated())
    test_dataset = dataloader.RetinopathyLoader("./data/new_test", "test")
    train_dataset = dataloader.RetinopathyLoader('./data/new_train', 'train')
    learning_rate = 1e-3
    # BATCH_SIZE = 4
    epochs = 5
    num_classes = 5
    batch_size = 4
    Momentum = 0.9
    Weight_decay = 5e-4
    loss_fn = nn.CrossEntropyLoss()
    feature_extract = True

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    model_0 = initialize_model(
        "Resnet50", num_classes, feature_extract, use_pretrained=False).to(device)

    params_to_update = model_0.parameters()

    optimizer = optim.SGD(params_to_update, lr=learning_rate,
                          momentum=Momentum, weight_decay=Weight_decay)
    df_0 = train(train_loader, test_loader, model_0, loss_fn,
                 optimizer, epochs, num_classes, device, "model_Resnet50_0")
    df_0.to_csv("ResNet50_0.csv", index=False)
    confusion_matrix, acc = test(
        test_loader, model_0, loss_fn, device, num_classes)
    figure = plot_confusion_matrix(confusion_matrix)
    figure.savefig('ResNet50 (NO Pretrained weights).png')

    # figure = plot(df_0, df_1, 'Result Comparison(ResNet50)')
    # figure.savefig('Result Comparison(ResNet50).png')
