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
        self.use_gpu = opt.use_cuda
        # Embedding layer and load weight
        self.embedding = nn.Embedding(num_embeddings=opt.vocab_size, embedding_dim=opt.embedding_dim)
        self.embedding.weight = nn.Parameter(opt.embeddings, requires_grad=opt.embedding_training)
        # layer init
        self.num_layers = 2
        self.dropout = opt.keep_dropout
        self.bilstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim//2 , num_layers=self.num_layers, dropout=self.dropout, bidirectional=True)
        self.linear = nn.Linear(opt.hidden_dim, opt.label_size) # linear layer: hidden_dim -> label_dim
        self.hidden = self.init_hidden()
        self.self_attention = nn.Sequential( # (batch_size, seq_len, hidden_dim)
            nn.Linear(opt.hidden_dim, 24),   # (batch_size, seq_len, 24)
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

        #print(sentence.size())
        embeds = self.embedding(sentence) # (batch_size, seq_len, embedding_dim)
        # print(embeds.size())
        x = embeds.permute(1,0,2) # (seq_len, batch_size, embedding_dim)
        self.hidden = self.init_hidden(sentence.size()[0])
        lstm_out, self.hidden = self.bilstm(x, self.hidden) # lstm_out (seq_len, batch_size, hidden_dim)
        # print(lstm_out.size())
        final = lstm_out.permute(1,0,2)                     # (batch_size, seq_len, hidden_dim)
        attn_ene = self.self_attention(final)               # (batch_size, seq_len, 1)
        # print(attn_ene.size())
        attns = F.softmax(attn_ene.view(sentence.size()[0], -1), dim=1).unsqueeze(2) # (batch_size, seq_len, 1) batch size will change in the last training batch
        #print(attns.size())
        #print(final.size())
        feats = (final * attns).sum(dim=1)                  # (batch_size, hidden_dim)
        #print(feats.size())
        y = self.linear(feats)                              # (batch_size, label_size)
        #print(y.size())
        #exit()
        return y

    '''
    for self-attention
        
        The first question is, what does the lstm model do?
            embedding the sentence or we called sequence into the feature vector.
    
        in the previous lstm based model
        the embedding feature of the whole sequence is the hidden state of last token or the mean of all token's hidden states
        while it is not a good way to asign the specific rule to generate the sequence feature.
        
        self-attention provides one way:
            self-attention will generate the (batch_size, seq_len, 1) 
            that the weight for each token in the sequence to generate the final sequence embedding
    '''