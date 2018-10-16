import sys
import csv
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

F_DIM = 254
NUM_TRAIN = 50
NUM_DEV = 30
NUM_TEST = 0

class LinearRegModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LinearRegModel,self).__init__()
        self.linear = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.linear(x)
    
def train(hyper_param, model, criterion, optimizer, x, y):
    x = Variable(torch.Tensor(x))
    y = Variable(torch.Tensor(y))
    for epoch in range(hyper_param["epochs"]):
        y_hat = model(x)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
def test(model, x, y):
    return 0


def main(args):
    train_x = []
    train_y = []
    dev_x = []
    dev_y = []
    test_x = []
    test_y = []
    count = 0
    with open("labels_features.csv") as file:
        f = csv.reader(file, delimiter = ',')
        next(f)
        for row in f:
            if count < NUM_TRAIN:
                train_y.append(row[0])
                train_x.append(row[1:])
            elif count < (NUM_TRAIN+NUM_DEV):
                dev_y.append(row[0])
                dev_x.append(row[1:])
            else:
                test_y.append(row[0])
                test_x.append(row[1:])
            
    
    hyper_param = {}
    hyper_param["epochs"] = 100
    hyper_param["lr"] = .01
    
    model = LinearRegModel(len(train_x[0]), 2) ##### modify len(x)!!!!!!!!
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), hyper_param["lr"])
    
    train(hyper_param, model, criterion, optimizer, train_x, train_y)
    
    test(model, dev_x, dev_y)
    
    

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))