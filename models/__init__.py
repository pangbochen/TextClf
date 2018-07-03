# -*- coding: UTF-8 -*-

from .LSTM import LSTMclf, LSTMclf_mixed
from .CNN import CNN
from .BILSTM import BILSTM
from .SelfAttention import SelfAttention

def init(opt):
    if opt.model == 'lstm':
        model = LSTMclf(opt)
    elif opt.model == 'lstm_mixed':
        model = LSTMclf_mixed(opt)
    elif opt.model == 'cnn':
        model = CNN(opt)
    elif opt.model == 'bilstm':
        model = BILSTM(opt)
    elif opt.model == 'attention':
        model = SelfAttention(opt)
    else:
        raise NotImplementedError

    return model