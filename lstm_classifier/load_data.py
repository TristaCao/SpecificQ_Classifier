# _*_ coding: utf-8 _*_

import os
import argparse
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe

BATCH_SIZE = 32

def load_dataset(args, test_sen=None):

    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    
    """

    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True,)
                      #fix_length=200)
    # LABEL = data.LabelField(tensor_type=torch.FloatTensor)
    LABEL = data.LabelField()
    train_data = data.TabularDataset(path=args.train_data_tsv_file, format='tsv',
                                     fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
    valid_data = data.TabularDataset(path=args.val_data_tsv_file, format='tsv',
                                     fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
    vectors = Vectors(name='amazon_word_embeddings.txt', cache='~/')
    TEXT.build_vocab(train_data, vectors=vectors)
    # TEXT.build_vocab(train_data, vectors=GloVe('6B', 100))
    # TEXT.build_vocab(train_data, vectors=GloVe('840B', 300))
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))

    train_iter, valid_iter = data.BucketIterator.splits((train_data, valid_data), batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

    '''Alternatively we can also use the default configurations'''
    # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)

    vocab_size = len(TEXT.vocab)

    return TEXT, LABEL, vocab_size, word_embeddings, train_iter, valid_iter
