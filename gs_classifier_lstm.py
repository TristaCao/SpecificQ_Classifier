import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import sys
import io
import re
import math
import csv
from csv import reader, writer
from random import shuffle
import nltk
from nltk import word_tokenize
from collections import Counter
from nltk.corpus import wordnet as wn
import string
import collections

NUM_TRAIN = 3000
NUM_DEV = 315
NUM_TEST = 315

torch.manual_seed(1)

class Question_Dataset(Dataset):
    """
    Pytorch data class for question classfication data
    """

    def __init__(self, question_label):
        self.questions = question_label[0]
        self.labels = question_label[1]
    
    def __getitem__(self, index):
        return (self.questions[index], self.labels[index])
    
    def __len__(self):
        return len(self.questions)


def batchify(batch):
    """
    Gather a batch of individual examples into one batch, 
    which includes the question text, question length and labels 

    Keyword arguments:
    batch: list of outputs from vectorize function
    """

    question_len = list()
    label_list = list()
    for ex in batch:
        question_len.append(len(ex[0]))
        label_list.append(ex[1])
    target_labels = torch.LongTensor(label_list)
    x1 = torch.LongTensor(len(question_len), max(question_len), len(ex[0][0])).zero_()
    for i in range(len(question_len)):
        question_text = batch[i][0]
        vec = torch.LongTensor(question_text)
        x1[i, :len(question_text)].copy_(vec)
    q_batch = {'text': x1, 'len': torch.FloatTensor(question_len), 'labels': target_labels}
    return q_batch


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_classes, boe):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.embedding = boe

        self.hidden2tag = nn.Linear(hidden_dim, num_classes)
        self.linear = nn.Linear(num_classes,1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),torch.zeros(1, 1, self.hidden_dim))
    
    def sent_embed(self, sent):
        embed = []
        for token in sent:
            if token not in self.embedding:
                print("here")
                continue
            embed.append(self.embedding[token])
        print("----------------------")
        print(torch.tensor(embed).size())
        return torch.tensor(embed)
        
    def forward(self, sentence):
        embed = Variable(self.sent_embed(sentence))
        print(len(sentence))
        print(sentence)
        lstm_out, self.hidden = self.lstm(
            embed.view(embed.size(0), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out[-1].view(-1))
        tag_scores = F.sigmoid(self.linear(tag_space))
        return tag_scores

def main(args):
    hyper_param = {}
    hyper_param["epochs"] = 60
    hyper_param["lr"] = .005
    hyper_param["batch"] = 32
    hyper_param["num_classes"] = 2
    hyper_param["hidden_dim"] = 50
    
    # import the pre-trained word embeddings
    embedding_vectors = open("vectors.txt")
    boe = {}
    for v in embedding_vectors:
        l = v.split()
        lst = l[1:]
        boe[l[0]] = [float(i) for i in lst]
        hyper_param["embedding_dim"] = len(boe[l[0]])    
    
    
    # read in data and transfer to embeddings
    data = []
    with open("processed_data.csv") as file:
        f = reader(file, delimiter = ',')
        next(f)
        for row in f:
            q = row[1]
            q_no_punc = q.translate(str.maketrans('','',string.punctuation)).lower()
            q_tokens = word_tokenize(q_no_punc) 
            if len(q_tokens) == 0:
                continue
            data.append((q_tokens, row[3]))
            
    
    train_data = data[:NUM_TRAIN]
#    print(train_data[0])
#    train_dataset = Question_Dataset(train_data)
#    train_loader = torch.utils.data.DataLoader(train_dataset, \
#                                             batch_size=hyper_param["batch"],\
#                                             shuffle=True, num_workers=0,\
#                                             collate_fn=batchify)
#    dev_data = data[NUM_TRAIN: NUM_TRAIN+NUM_DEV]
#    dev_dataset = Question_Dataset(dev_data)
#    dev_loader = torch.utils.data.DataLoader(dev_dataset, \
#                                             batch_size=hyper_param["batch"],\
#                                             shuffle=True, num_workers=0,\
#                                             collate_fn=batchify)
#    test_data = data[NUM_TRAIN+NUM_DEV:]
#    test_dataset = Question_Dataset(test_data)
#    test_loader = torch.utils.data.DataLoader(test_dataset, \
#                                             batch_size=hyper_param["batch"],\
#                                             shuffle=True, num_workers=0,\
#                                             collate_fn=batchify)
        
    model = LSTMClassifier(hyper_param["embedding_dim"],\
                           hyper_param["hidden_dim"], \
                           hyper_param["num_classes"], boe)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), hyper_param["lr"])
    
    for epoch in range(hyper_param["epochs"]): 
        for sentence, label in train_data:
            label = Variable(torch.FloatTensor([int(label)]))
            model.zero_grad()
    
            model.hidden = model.init_hidden()
    
#            sentence_in = prepare_sequence(sentence, word_to_ix)
#            targets = prepare_sequence(tags, tag_to_ix)
            tag_scores = model(sentence)
    
            loss = loss_function(tag_scores, label)
            loss.backward()
            optimizer.step()
        

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))