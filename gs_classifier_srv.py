import sys
import csv
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.svm import SVR
import pickle

NUM_TRAIN = 2350
NUM_DEV = 290
NUM_TEST = 290

#NUM_TRAIN = 2500
#NUM_DEV = 315
#NUM_TEST = 315
FINAL = False

MODEL_FILE = "classifier.sav"

class FeatureData(Dataset):
    def __init__(self, x, y):
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len


def accuracy(y,y_hat):
    count = 0
    score = 0
    for i,train in enumerate(y,0):
        count += 1
        if y_hat[i][0] == 0 and  train<0.5:
            score += 1
        elif y_hat[i][0] == 1 and train>= 0.5:
            score += 1
    return score/count

def main(args):
    train_x, train_y = None, None
    dev_x, dev_y = None, None
    test_x, test_y = None, None
    xy = np.loadtxt('labels_features_train.csv', delimiter=',', dtype = np.float32)
    train_x = xy[:,1:]
    train_y = xy[:,[0]]
    xy = np.loadtxt('labels_features_dev.csv', delimiter=',', dtype = np.float32)
    dev_x = xy[:,1:]
    dev_y = xy[:,[0]]


#    dev_x = x[0:NUM_DEV]
#    dev_y = y[0:NUM_DEV]
#    train_x = np.concatenate((x[0:0*NUM_DEV],x[0*NUM_DEV+NUM_DEV:NUM_DEV+NUM_TRAIN]))
#    train_y = np.concatenate((y[0:0*NUM_DEV],y[0*NUM_DEV+NUM_DEV:NUM_DEV+NUM_TRAIN]))
    
#    test_x = x[NUM_TRAIN+NUM_DEV:]
#    test_y = y[NUM_TRAIN+NUM_DEV:]
#    hyper_param = {}
#    hyper_param["epochs"] = 30
#    hyper_param["lr"] = .001
#    hyper_param["batch"] = 16
#    
#    dataset = FeatureData(train_x, train_y)
#    train_loader = DataLoader(dataset=dataset,\
#                              batch_size=hyper_param["batch"],\
#                              shuffle=True,\
#                              num_workers=2)
#    
    model = SVR( C=15, epsilon=0.01)
    
    model = model.fit(train_x, train_y)
    
    #save model
    pickle.dump(model, open(MODEL_FILE, 'wb'))
    # load and use model as following:
    #loaded_model = pickle.load(open(filename, 'rb'))
    # y = loaded_model.predict(test_x)
    
    test_acc = accuracy(model.predict(dev_x),dev_y)
    print("Train acc: " + str(accuracy(model.predict(train_x), train_y)))
    print("Testing acc: " + str(test_acc))
    
#    overall = 0
#    for group_num in range(9):
#        dev_x = x[group_num*NUM_DEV:group_num*NUM_DEV+NUM_DEV]
#        dev_y = y[group_num*NUM_DEV:group_num*NUM_DEV+NUM_DEV]
#        train_x = np.concatenate((x[0:group_num*NUM_DEV],x[group_num*NUM_DEV+NUM_DEV:NUM_DEV+NUM_TRAIN]))
#        train_y = np.concatenate((y[0:group_num*NUM_DEV],y[group_num*NUM_DEV+NUM_DEV:NUM_DEV+NUM_TRAIN]))
#                
##        hyper_param = {}
##        hyper_param["epochs"] = 30
##        hyper_param["lr"] = .001
##        hyper_param["batch"] = 16
#        
##        dataset = FeatureData(train_x, train_y)
##        train_loader = DataLoader(dataset=dataset,\
##                                  batch_size=hyper_param["batch"],\
##                                  shuffle=True,\
##                                  num_workers=2)
#        
#        model = SVR( C=15, epsilon=0.03)
#        
#        model = model.fit(train_x, train_y)
#        test_acc = accuracy(model.predict(dev_x),dev_y)
#        overall += test_acc
#        print("Group " + str(group_num) + "-------------------------")
#        print("Train acc: " + str(accuracy(model.predict(train_x), train_y)))
#        print("Testing acc: " + str(test_acc))
#        
#    print("Overall acc: " + str(overall/9))
    
# Cross Validation
#    Group0= 0.6828571428571428
#    Group1= 0.7114285714285714
#    Group2= 0.7942857142857143
#    Group3= 0.7457142857142857
#    Group4= 0.7914285714285715
#    Group5= 0.5971428571428572
#    Group6= 0.7485714285714286
#    Group7= 0.7828571428571428
#    Group8= 0.7714285714285715
#    overall = (Group0+Group1+Group2+Group3+Group4+Group5+Group6+Group7+Group8)/9
#    print(overall) 0.7361904761904762
    
    

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))