# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

'''
use nn.LSTM and nn.Linear in the LSTM model

nn.LSTM

input of shape (seq_len, batch, input_size)
output of shape (seq_len, batch, hidden_size * num_directions)
so we need to permute the input data


nn.Linear

Input: (N,∗,in_features)(N,∗,in_features) 
Output: (N,∗,out_features)

torch.mean(tensor, dim)
dim: the dim to reduce
'''

class LSTMclf(nn.Module):
    def __init__(self, opt):
        self.opt = opt
        super(LSTMclf, self).__init__()

        self.hidden_dim = opt.hidden_dim
        self.batch_size = opt.batch_size
        self.use_cuda = opt.use_cuda
        # layer
        if opt.one_hot is False:
            # use pre_trained word embedding
            self.embedding = nn.Embedding(opt.vocab_size, opt.embedding_dim)
            self.embedding.weight = nn.Parameter(opt.embeddings, requires_grad=opt.embedding_training)
        else:
            # use one-hot
            self.embedding = nn.Embedding(opt.vocab_size, opt.embedding_dim)
            self.embedding.weight.data = torch.eye(opt.vocab_size)
            self.embedding_linear = nn.Linear(opt.vocab_size, opt.embedding_dim)

        self.lstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim)
        self.label = nn.Linear(opt.hidden_dim, opt.label_size)
        self.hidden = self.init_hidden()

        self.mean =opt.__dict__.get('lstm_mean', True)

    def init_hidden(self, batch_size=None):
        # use special initail strategy
        if batch_size is None:
            batch_size = self.batch_size
        # set for cuda variable setting
        if self.use_cuda:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, sentence): # (batch_size, seq_len)
        # print(sentence.size())
        if self.opt.one_hot is False:
            # use pretrained word embedding
            embeds = self.embedding(sentence) # (batch_size, seq_len, embedding_dim)
        else:
            embeds = self.embedding(sentence) # (batch_size, seq_len, vocab_size)
            embeds = self.embedding_linear(embeds) # (batch_size, seq_len, embedding_dim)
        # tensor.permute, swap the tensor dim
        x = embeds.permute(1, 0, 2) # (seq_len, batch_size, embedding_dim)
        self.hidden = self.init_hidden(sentence.size()[0]) # send batch_size
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        if self.mean == "mean":
            out = lstm_out.permute(1, 0, 2) # (seq_len, batch_size, hidden_dim) to (batch_size, seq_len, hidden_dim)
            final = torch.mean(out, 1) # (batch_size, hidden_dim) 使用每个单词的情感向量的均值作为句子的情感向量
        else:
            final = lstm_out[-1] # (batch_size, hidden_dim) 使用最后一个单词的情感向量作为句子的情感向量
        ret = self.label(final) # (batch_size, label_size) 得到每个句子的类别分布
        return ret


class LSTMclf_mixed(nn.Module):
    def __init__(self, opt):
        self.opt = opt
        super(LSTMclf_mixed, self).__init__()

        self.hidden_dim = opt.hidden_dim
        self.batch_size = opt.batch_size
        self.use_cuda = opt.use_cuda
        # layer
        self.embedding = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.embedding.weight = nn.Parameter(opt.embeddings, requires_grad=opt.embedding_training)
        self.lstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim)
        self.label = nn.Linear(opt.hidden_dim, opt.label_size)
        self.hidden = self.init_hidden()

        self.mean =opt.__dict__.get('lstm_mean', True)

    def init_hidden(self, batch_size=None):
        # use special initail strategy
        if batch_size is None:
            batch_size = self.batch_size
        # set for cuda variable setting
        if self.use_cuda:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, sentence): # (batch_size, seq_len, input_dim)
        embeds = self.embedding(sentence) # (batch_size, seq_len, embedding_dim)
        # tensor.permute, swap the tensor dim
        x = embeds.permute(1, 0, 2) # (seq_len, batch_size, embedding_dim)
        self.hidden = self.init_hidden(sentence.size()[0]) # send batch_size
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        if self.mean == "mean":
            out = lstm_out.permute(1, 0, 2) # (seq_len, batch_size, hidden_dim) to (batch_size, seq_len, hidden_dim)
            final = torch.mean(out, 1) # (batch_size, hidden_dim) 使用每个单词的情感向量的均值作为句子的情感向量
        else:
            final = lstm_out[-1] # (batch_size, hidden_dim) 使用最后一个单词的情感向量作为句子的情感向量
        ret = self.label(final) # (batch_size, label_size) 得到每个句子的lebel分布
        return ret, final