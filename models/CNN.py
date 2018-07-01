# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, opt):
        super(CNN, self).__init__()

        self.batch_size = opt.batch_size
        self.max_sent_len = opt.max_seq_len
        self.embedding_dim = opt.embedding_dim
        self.vocab_size = opt.vocab_size
        self.label_size = opt.label_size
        self.keep_dropout = opt.keep_dropout
        # embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.weight = nn.Parameter(opt.embeddings, requires_grad=opt.embedding_training)
        # for conv layers
        self.kernel_num = 256
        self.kernel_size = 3

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=opt.embedding_dim,
                out_channels=self.kernel_num,
                kernel_size=self.kernel_size,
            ),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=(opt.max_seq_len - self.kernel_size + 1)
            )
        )

        self.fc = nn.Linear(self.kernel_num, opt.label_size)

    def forward(self, sentence):
        # (batch_size, seq_len, input_dim)
        embeds = self.embedding(sentence) # (batch_size, seq_len, embedding_dim)
        cnn_out = self.conv(embeds.permute(0, 2, 1)) # (batch_size, kernel_num, 1)
        cnn_out = cnn_out.view(cnn_out.size(0), -1) # sequuze (batch_size, kernel_num)
        pred = self.fc(cnn_out)
        return pred