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
        self.len = x.size(0)
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
    embeds = torch.tensor(embeds)
    return embeds

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_classes, batch):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        
        self.batch_size = batch

        self.hidden2tag = nn.Linear(hidden_dim, num_classes)
        self.linear = nn.Linear(num_classes,1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim),\
                torch.zeros(1, self.batch_size, self.hidden_dim))
    
    
        
    def forward(self, sentences, sent_lengths):
        print(sentences.size())
        print(sent_lengths.size())
        X = torch.nn.utils.rnn.pack_padded_sequence(sentences, sent_lengths, batch_first=True)
        lstm_out, self.hidden = self.lstm(
           X.view(sentences.size(0), self.batch_size, -1), self.hidden)
        
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        X = self.hidden_to_tag(X)
        tag_score = F.sigmoid(self.linear(X))
        return tag_score

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
    
    boe[PAD] = [0]*200  # "<PAD>" has embedding 0*200   
    
    model = LSTMClassifier(hyper_param["embedding_dim"],\
                           hyper_param["hidden_dim"], \
                           hyper_param["num_classes"],\
                           hyper_param["batch"])
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
    p_sents, labels, lengths = pad(train_data)
    sents = sents_embed(p_sents, boe)
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    dataset = FeatureData(sents, labels, lengths)
    train_loader = DataLoader(dataset=dataset,\
                              batch_size=hyper_param["batch"],\
                              shuffle=True,\
                              num_workers=2)
    
    
    for epoch in range(hyper_param["epochs"]): 
        for i, data in enumerate(train_loader, 0):
            sents, labels, lengths = data
            labels = Variable(labels)
            sents = Variable(sents)
            lengths = Variable(lengths)
            model.zero_grad()
    
            model.hidden = model.init_hidden()
    
#            sentence_in = prepare_sequence(sentence, word_to_ix)
#            targets = prepare_sequence(tags, tag_to_ix)
            tag_scores = model(sents, lengths)
    
            loss = loss_function(tag_scores, labels)
            loss.backward()
            optimizer.step()
        

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))