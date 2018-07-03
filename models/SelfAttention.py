# -*- coding: UTF-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class SelfAttention(nn.Module):
    def __init__(self, opt):
        '''Parameters
        opt include
        hidden_dim
        embedding_dim
        embedding_training
        batch_size
        keep_dropout
        '''
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.hidden_dim = opt.hidden_dim
        self.batch_size = opt.batch_size
        self.use_cuda = opt.use_cuda
        # Embedding layer and load weight
        self.embedding = nn.Embedding(num_embeddings=opt.vocab_size, embedding_dim=opt.embedding_dim)
        self.embedding.weight = nn.Parameter(opt.embeddings, requires_grad=opt.embedding_training)
        # layer init
        self.num_layers = 1
        self.dropout = opt.keep_dropout
        self.bilstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim//2 , num_layers=self.num_layers, dropout=self.dropout, bidirectional=True)
        self.linear = nn.Linear(opt.hidden_dim, opt.label_size) # linear layer: hidden_dim -> label_dim
        self.hidden = self.init_hidden()
        self.self_attention = nn.Sequential(
            nn.Linear(opt.hidden_dim, 24),
            nn.ReLU(True), # inplace
            nn.Linear(24, 1)
        )

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if self.use_gpu:
            h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2).cuda())
            c0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2).cuda())
        else:
            h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
            c0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
        return (h0, c0)

    def forward(self, sentence):
        # (batch_size, sentence_num, embedding_dim)
        embeds = self.word_embeddings(sentence)
        x = embeds.permute(1,0,2)
        self.hidden = self.init_hidden(sentence.size()[0])
        lstm_out, self.hidden = self.bilstm(x, self.hidden)
        final = lstm_out.permute(1,0,2)
        attn_ene = self.self_attention(final)
        attns = F.softmax(attn_ene.view(self.batch_size, -1))
        feats = (final * attns).sum(dim=1)
        y = self.linear(feats)
        return y