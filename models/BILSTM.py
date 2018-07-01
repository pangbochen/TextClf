# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BILSTM(nn.Module):
    def __init__(self, opt):
        # laod options
        super(BILSTM, self).__init__()
        self.opt = opt
        self.hidden_dim = opt.hidden_dim
        self.batch_size = opt.batch_size
        self.use_cuda = opt.use_cuda
        # torch.Embedding(num_embeddings, embedding_dim)
        # size of dcitionary
        # size of each embedding vector
        # weight:  the learnable weights of the module of shape (num_embeddings, embedding_dim)
        self.embedding = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.embedding.weight = nn.Parameter(opt.embeddings, requires_grad=opt.embedding_training)

        self.num_layers = 2

        self.dropout = opt.keep_dropout
        # nn.LSTM
        # input_size – The number of expected features in the input x
        # hidden_size – The number of features in the hidden state h
        # num_layers – Number of recurrent layers, for bilstm, 2
        self.bilstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim // 2, num_layers=self.num_layers, dropout=self.dropout, bidirectional=True)
        self.linear = nn.Linear(opt.hidden_dim, opt.label_size)
        self.hidden = self.init_hidden()
        self.mean = opt.__dict__.get('lstm_mean', True)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.use_cuda:
            h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2).cuda())
            c0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2).cuda())
        else:
            h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
            c0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
        return (h0, c0)

    def forward(self, sentence):
        # sentence : batch_size x len_words x embedding_dim
        # for example, 128  x 200 x 300
        self.hidden = self.init_hidden(sentence.size()[0]) # load batch_size

        x = self.embedding(sentence)
        x = x.permute(1, 0, 2) #
        lstm_out, self.hidden = self.bilstm(x, self.hidden)
        if self.mean == 'mean':
            out = lstm_out.permute(1, 0, 2)
            final = torch.mean(out, 1) # dim : dim to reduce
        else:
            final = lstm_out[-1]
        y = self.linear(final)
        return y

