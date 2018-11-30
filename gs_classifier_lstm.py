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
PAD = "<PAD>"

torch.manual_seed(1)


class FeatureData(Dataset):
    def __init__(self, x, y, length):
        self.len = len(x)
        self.x_data = x
        self.length = length
        self.y_data = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.length[index]
    
    def __len__(self):
        return self.len


# take in a list of data and pad them
# return the list of padded data, labels, and a list of len
def pad(data):
    lengths = [len(sentence) for sentence, label in data]
    longest = max(lengths)
    paddeds = []
    labels = []
    for i, y_len in enumerate(lengths):
        sequence, label = data[i]
        labels.append(int(label))
        padded = [PAD]*longest
        padded[0:y_len] = sequence[:y_len]
        paddeds.append(padded)
    
    return paddeds, labels, lengths


def sents_embed(sents, boe):
    embeds = []
    for sent in sents:
        embed = []
        for token in sent:
            if token not in boe:
                embed.append(boe[PAD])
                continue
            embed.append(boe[token])
        embeds.append(embed)
    #embeds = np.asarray(embeds)
    # embeds = torch.tensor(embeds)
    return embeds


def iterate_minibatches(sents, labels, lengths, batch_size, shuffle=True):
    if shuffle:
        indices = np.arange(len(sents))
        np.random.shuffle(indices)
    for start_idx in range(0, len(sents) - batch_size + 1, batch_size):
        if shuffle:
            ex = indices[start_idx:start_idx + batch_size]
        else:
            ex = slice(start_idx, start_idx + batch_size)
        yield np.array(sents)[ex], np.array(labels)[ex], np.array(lengths)[ex]


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, batch):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.batch_size = batch
        self.hidden2tag = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim),
                torch.zeros(1, self.batch_size, self.hidden_dim))

    def forward(self, sentences):
        output, (hidden, cell) = self.lstm(sentences)
        mask = sentences.ge(0.)
        mask = mask.type(torch.FloatTensor).cuda()
        output = output * mask[:, :, :self.hidden_dim]
        output = torch.sum(output, dim=1)
        X = self.hidden2tag(output)
        tag_score = F.sigmoid(X)

        return tag_score


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum()/len(correct)
    return acc


def main(args):
    hyper_param = {}
    hyper_param["epochs"] = 20
    hyper_param["lr"] = .0001
    hyper_param["batch"] = 32
    hyper_param["hidden_dim"] = 100
    
    # import the pre-trained word embeddings
    embedding_vectors = open("vectors.txt")
    boe = {}
    for v in embedding_vectors:
        l = v.split()
        lst = l[1:]
        boe[l[0]] = [float(i) for i in lst]
        hyper_param["embedding_dim"] = len(boe[l[0]])    
    
    boe[PAD] = [0]*200  # "<PAD>" has embedding 0*200   
    
    model = LSTMClassifier(hyper_param["embedding_dim"],
                           hyper_param["hidden_dim"],
                           hyper_param["batch"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), hyper_param["lr"])
        
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
    train_p_sents, train_labels, train_lengths = pad(train_data)
    train_sents = sents_embed(train_p_sents, boe)

    dev_data = data[NUM_TRAIN:NUM_TRAIN+NUM_DEV]
    dev_p_sents, dev_labels, dev_lengths = pad(dev_data)
    dev_sents = sents_embed(dev_p_sents, boe)

    for epoch in range(hyper_param["epochs"]):
        train_avg_loss = 0.
        train_avg_acc = 0.
        count = 0
        for sents, labels, lengths in iterate_minibatches(train_sents, train_labels, train_lengths,
                                                          hyper_param["batch"]):
            labels = torch.tensor(labels).type(torch.FloatTensor)
            sents = torch.tensor(sents).type(torch.FloatTensor).cuda()
            labels = Variable(labels)
            sents = Variable(sents)
            model.zero_grad()
            model.hidden = model.init_hidden()
            preds = model(sents)[:, 0]
            preds = preds.type(torch.FloatTensor)
            loss = loss_function(preds, labels)
            acc = binary_accuracy(preds, labels)
            train_avg_loss += loss
            train_avg_acc += acc
            loss.backward()
            optimizer.step()
            count += 1
        train_avg_loss = train_avg_loss / count
        train_avg_acc = train_avg_acc / count
        print('Epoch: %d, Train Loss: %.2f, Train Acc: %.2f' % (epoch, train_avg_loss, train_avg_acc))
        dev_avg_loss = 0.
        dev_avg_acc = 0.
        count = 0
        for sents, labels, lengths in iterate_minibatches(dev_sents, dev_labels, dev_lengths,
                                                          hyper_param["batch"]):
            labels = torch.tensor(labels).type(torch.FloatTensor)
            sents = torch.tensor(sents).type(torch.FloatTensor).cuda()
            labels = Variable(labels)
            sents = Variable(sents)
            model.hidden = model.init_hidden()
            preds = model(sents)[:, 0]
            preds = preds.type(torch.FloatTensor)
            loss = loss_function(preds, labels)
            acc = binary_accuracy(preds, labels)
            dev_avg_loss += loss
            dev_avg_acc += acc
            count += 1
        dev_avg_loss = dev_avg_loss / count
        dev_avg_acc = dev_avg_acc / count
        print('Epoch: %d, Dev Loss: %.2f, Dev Acc: %.2f' % (epoch, dev_avg_loss, dev_avg_acc))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
