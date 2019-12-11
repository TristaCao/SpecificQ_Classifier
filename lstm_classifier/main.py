import os
import argparse
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from models.LSTM import LSTMClassifier
from models.LSTM_Attn import AttentionModel
from models.selfAttention import SelfAttention
from torchtext import data
import sys
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

BATCH_SIZE = 32

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, loss_fn, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not BATCH_SIZE):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter),

def eval_model(model, loss_fn, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            if (text.size()[0] is not BATCH_SIZE):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter),


def evaluate(model, TEXT, LABEL, data_tsv_file, epoch):
    test_data = data.TabularDataset(path=data_tsv_file, format='tsv',fields=[('text', TEXT),('label', LABEL)], skip_header=True)
    #test_predictions_file = open(args.val_data_tsv_file + '.preds.epoch' + str(epoch), 'w')
    pred_labels = []
    true_labels = []
    for i in range(len(test_data)):
        test_sent = [[TEXT.vocab.stoi[x] for x in test_data[i].text]]
        test_sent = np.asarray(test_sent)
        test_sent = torch.LongTensor(test_sent)
        test_tensor =  Variable(test_sent, volatile=True).cuda()
        model.eval()
        output = model(test_tensor, 1)
        out = F.softmax(output, 1)
        if (torch.argmax(out[0]) == 1):
            pred_label = 0
        else:
            pred_label = 1
        #test_predictions_file.write('%s\t%s\t%d\n' % (' '.join(test_data[i].text), test_data[i].label, pred_label))
        pred_labels.append(pred_label)
        true_labels.append(int(test_data[i].label))
    print('0: %d, 1: %d' % (pred_labels.count(0), pred_labels.count(1)))
    #print('Weighted F1-score: %.4f Precision: %.4f Recall: %.4f Accuracy: %.4f' % (f1_score(true_labels, pred_labels, average='weighted'), precision_score(true_labels, pred_labels, average='weighted'), recall_score(true_labels, pred_labels, average='weighted'), accuracy_score(true_labels, pred_labels)))
    #print('Macro F1-score: %.4f Precision: %.4f Recall: %.4f Accuracy: %.4f' % (f1_score(true_labels, pred_labels, average='macro'), precision_score(true_labels, pred_labels, average='macro'), recall_score(true_labels, pred_labels, average='macro'), accuracy_score(true_labels, pred_labels)))
    print('Micro F1-score: %.4f Precision: %.4f Recall: %.4f Accuracy: %.4f' % (f1_score(true_labels, pred_labels, average='micro'), precision_score(true_labels, pred_labels, average='micro'), recall_score(true_labels, pred_labels, average='micro'), accuracy_score(true_labels, pred_labels)))


def main(args):

    TEXT, LABEL, vocab_size, word_embeddings, train_iter, valid_iter = load_data.load_dataset(args)

    #learning_rate = 2e-5
    learning_rate = 0.00001
    batch_size = BATCH_SIZE
    output_size = 2
    # hidden_size = 256
    hidden_size = 64
    embedding_length = 200
    
    #model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    model = AttentionModel(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    #model = SelfAttention(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    #loss_fn = F.cross_entropy
    print(LABEL.vocab.stoi)
    print(LABEL.vocab.freqs)
    print(LABEL)
    label_weights = torch.FloatTensor(np.asarray([1.0, 2.0]))
    label_weights_tensor = Variable(label_weights, volatile=True).cuda()
    loss_fn = torch.nn.CrossEntropyLoss(weight=label_weights_tensor)
    #loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        train_loss, train_acc = train_model(model, loss_fn, train_iter, epoch)
        val_loss, val_acc = eval_model(model, loss_fn, valid_iter)

        print('Epoch: %d, Train Loss: %.3f, Train Acc: %.2f, Val Loss: %.3f, Val Acc: %.2f' % (epoch+1,train_loss, train_acc, val_loss, val_acc))
        evaluate(model, TEXT, LABEL, args.train_data_tsv_file, epoch)
        evaluate(model, TEXT, LABEL, args.val_data_tsv_file, epoch)
        #torch.save(model.state_dict(), args.save_model_file+'.epoch'+str(epoch+1))
    
    test_loss, test_acc = eval_model(model, loss_fn, valid_iter)
    print('Test Loss: %.3f, Test Acc: %.2f' % (test_loss, test_acc))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--train_data_tsv_file", type = str)
    argparser.add_argument("--val_data_tsv_file", type = str)
    argparser.add_argument("--save_model_file", type = str)
    args = argparser.parse_args()
    print (args)
    print ("")
    main(args)
