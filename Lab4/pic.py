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
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df_1 = pd.read_csv('ResNet50_0.csv')
    df_0 = pd.read_csv('ResNet50_1.csv')
    result = pd.concat([df_0, df_1], axis=1)
    figure = result.plot(title='Result Comparsion(ResNet50)',
                         xlabel='Epoch', ylabel='Accuracy(%)')
    plt.savefig('Result Comparsion(ResNet50).png')

    plt.show()
