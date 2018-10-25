import sys
import csv
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np

NUM_TRAIN = 300
NUM_DEV = 100
NUM_TEST = 100

class FeatureData(Dataset):
    def __init__(self, x, y):
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    
        
class LinearRegModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LinearRegModel,self).__init__()
        self.linear = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.linear(x)
    
def train(hyper_param, model, criterion, optimizer, train_loader):
    for epoch in range(hyper_param["epochs"]):
        for i, data in enumerate(train_loader, 0):
            x, y = data
            x, y = Variable(x), Variable(y)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            #print(epoch, i, loss.data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
def test(model,criterion, x, y):
    losses = []
    for i in range(x.shape[0]):
        y_pred = model(Variable(torch.from_numpy(x[i])))
        loss = criterion(y_pred, Variable(torch.from_numpy(y[i])))
        losses.append([loss.data.item()])
        print(loss.data.item())
    return losses
    


def main(args):
    train_x, train_y = None, None
    dev_x, dev_y = None, None
    test_x, test_y = None, None
    xy = np.loadtxt('labels_features.csv', delimiter=',', dtype = np.float32)
    x = xy[:,1:]
    y = xy[:,[0]]
    train_x = x[0:NUM_TRAIN]
    train_y = y[0:NUM_TRAIN]
    dev_x = x[NUM_TRAIN:NUM_TRAIN+NUM_DEV]
    dev_y = y[NUM_TRAIN:NUM_TRAIN+NUM_DEV]
    test_x = x[NUM_TRAIN+NUM_DEV:]
    test_y = y[NUM_TRAIN+NUM_DEV:]
            
    
    hyper_param = {}
    hyper_param["epochs"] = 1500
    hyper_param["lr"] = .001
    hyper_param["batch"] = 16
    
    dataset = FeatureData(train_x, train_y)
    train_loader = DataLoader(dataset=dataset,\
                              batch_size=hyper_param["batch"],\
                              shuffle=True,\
                              num_workers=2)
    
    model = LinearRegModel(x.shape[1], 1) 
    #model = LinearRegModel(1,1) -- for baseline
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), hyper_param["lr"])
    
    train(hyper_param, model, criterion, optimizer, train_loader)
    
#    word_param_weights = []
    p = True
    for param in model.parameters():
        if p:
            print(param.data[0,0:19])
            print(param.data[0,19:])
#            for w in param.data[0][19:]:
#                word_param_weights.append(w.item())
            p = False
    result = test(model,criterion, dev_x, dev_y)
    
#    word_weights = []
#    word_type = []
#    with open("word_feature_weights.csv") as file:
#        f = reader(file, delimiter = ',')
#        for row in f:
#            word_type = row
#            
#    for index in range(len(word_type)):
#        word_weights.append([word_type[index], word_param_weights[index]])
#    
#    with open('word_weights.csv', mode = 'w') as file:
#        w = writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#        for row in word_weights:
#            w.writerow(row)
#    
#    with open('baseline.csv', mode = 'w') as file:
#        w = writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#        w.writerow(["result"])
#        for l in result:
#            w.writerow(l)
    
    

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))