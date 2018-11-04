import sys
import csv
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np

NUM_TRAIN = 2500
NUM_DEV = 315
NUM_TEST = 315
FINAL = False

class FeatureData(Dataset):
    def __init__(self, x, y):
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    
        
class LogRegModel(nn.Module):
    def __init__(self, num_features):
        super(LogRegModel,self).__init__()
        self.linear = nn.Linear(num_features, 1)
    
    def forward(self, x):
        return F.sigmoid(self.linear(x))
    
def train(model, criterion, optimizer, train_loader):
    acc = 0
    losses = 0
    count = 0
    for i, data in enumerate(train_loader, 0):
        x, y = data
        x, y = Variable(x), Variable(y)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        
        count += 1
        losses += loss.data.item()
        
        inner_count = 0
        inner_acc = 0
        for index in range(y.shape[0]):
            inner_count += 1
            if y_hat[index][0].item() >= 0.5 and y[index][0].item() == 1:
                inner_acc += 1
            elif y_hat[index][0].item() < 0.5 and y[index][0].item() == 0:
                inner_acc += 1
        acc += inner_acc/ inner_count
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return acc/count, losses/count
    
        
def test(model,criterion, x, y):
    acc = 0
    count = 0
    l = 0
    for i in range(x.shape[0]):
        count += 1
        y_pred = model(Variable(torch.from_numpy(x[i])))
        loss = criterion(y_pred,Variable(torch.from_numpy(y[i])))
        y_pred = y_pred.item()
        
        l += loss.data.item()
        if y_pred >= 0.5 and y[i][0] == 1:
            acc += 1
#            print("Number " + str(i) + " has y_pred: "+str(y_pred)+\
#                     " but true y should be: " + str(y[i][0]))
        elif y_pred < 0.5 and y[i][0] == 0:
            acc += 1
        
        
    return acc/count, l/count

def final_test(model,criterion, x, y):
    acc = 0
    count = 0
    l = 0
    tt=0
    tf=0
    ft=0
    ff=0
    for i in range(x.shape[0]):
        count += 1
        y_pred = model(Variable(torch.from_numpy(x[i])))
        loss = criterion(y_pred,Variable(torch.from_numpy(y[i])))
        y_pred = y_pred.item()
        
        l += loss.data.item()
        if y_pred >= 0.5 and y[i][0] == 1:
            acc += 1
            tt+=1
#            print("Number " + str(i) + " has y_pred: "+str(y_pred)+\
#                     " but true y should be: " + str(y[i][0]))
        elif y_pred < 0.5 and y[i][0] == 0:
            ff+=1
            acc += 1
        else:
            if y_pred >= 0.5 and y[i][0] == 0:
                tf +=1
            else:
                ft+=1
#                print("Number " + str(i) + " has y_pred: "+str(y_pred)+\
#                     " but true y should be: " + str(y[i][0]))
        
#    print(tt)
#    print(tf)
#    print(ft)
#    print(ff)
    return acc/count, l/count


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
    

#    train_x = np.concatenate((x[0:400], x[500:500]), axis=0)
#    train_y = np.concatenate((y[0:400], y[500:500]), axis=0)
#    dev_x = x[400:500]
#    dev_y = y[400:500]
    
    # Check the balance of train and dev data
#    train_specific = 0
#    train_general = 0
#    for y in train_y:
#        if y[0] == 0:
#            train_general += 1
#        else:
#            train_specific += 1
#    dev_specific = 0
#    dev_general = 0
#    for y in dev_y:
#        if y[0] == 0:
#            dev_general += 1
#        else:
#            dev_specific += 1
#            
#    print(train_specific) --639
#    print(train_general) --861
#    print(dev_specific) --83
#    print(dev_general) --117
            
            
    
    hyper_param = {}
    hyper_param["epochs"] = 30
    hyper_param["lr"] = .001
    hyper_param["batch"] = 16
    
    dataset = FeatureData(train_x, train_y)
    train_loader = DataLoader(dataset=dataset,\
                              batch_size=hyper_param["batch"],\
                              shuffle=True,\
                              num_workers=2)
    
    model = LogRegModel(x.shape[1]) 
    #model = LogRegModel(1) #-- for baseline
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), hyper_param["lr"])
    
    
    for epoch in range(hyper_param["epochs"]):
        print("Epoch: " + str(epoch) + "---------------------")
        train_acc, train_loss = train( model, criterion, optimizer, train_loader)
        print("Training Accuracy: " + str(train_acc))
        print("Training Loss: " + str(train_loss))
        test_acc, test_loss= test(model,criterion, dev_x, dev_y)
        print("Testing Accuracy: " + str(test_acc)) 
        print("Testing Loss: " + str(test_loss))
    
    # print out result for analysis
#    test_acc, test_loss= final_test(model,criterion, dev_x, dev_y)
#    print("Testing Accuracy: " + str(test_acc)) 
#    print("Testing Loss: " + str(test_loss))
#    final_test(model, criterion, train_x, train_y)
    
#    word_param_weights = []
#    p = True
#    for param in model.parameters():
#        if p:
#            print(param.data[0,0:19]) #print out parameters
#            print(param.data[0,19:])
#            for w in param.data[0][19:]:
#                word_param_weights.append(w.item())
#            p = False
    
    
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