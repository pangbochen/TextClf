# -*- coding: UTF-8 -*-

import torch
import torch.nn.functional as F
from torchtext import data, datasets
from torchtext.vocab import GloVe, FastText
import numpy as np
from sklearn.metrics import accuracy_score
from torchnet.meter import AUCMeter

def clip_gradient(optimizer, grad_clip):
    # https://pytorch.org/docs/stable/torch.html#torch.clamp
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None and param.requires_grad:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def evaluate(model, eval_iter, opt):
    model.eval()
    accuracy = []
    threshold = 0.5

    AUC_list = [AUCMeter() for _ in range(opt.label_size)]

    for index, batch in enumerate(eval_iter):
        text = batch.comment_text.data
        label = torch.stack([
            batch.toxic, batch.severe_toxic, batch.obscene,
            batch.threat, batch.insult, batch.identity_hate
        ], dim=1)

        label = label.float()

        # for label (batch_size, classes_size)
        # for text (batch_size, max_seq_len)

        pred = model(text)

        is_class = pred > threshold  # is it is greater than the threshold
        is_class = is_class.float()  # (batch_size, classes_size)

        # for AUC_meter
        pred = torch.nn.functional.sigmoid(pred)
        # print(pred)
        # print(label)

        for i in range(opt.label_size):
            if opt.use_cuda:
                AUC_list[i].add(
                    output=pred.data.cpu().numpy()[:, i],
                    target=label.data.cpu().numpy()[:, i]
                )
            else:
                AUC_list[i].add(
                    output=pred.data.numpy()[:, i],
                    target=label.data.numpy()[:, i]
                )

        percision = is_class == label # (batch_size, classes_size)
        percision = percision.float()
        percision = percision.mean(dim=0) # (classes_size)

        if opt.use_cuda:
            accuracy.append(percision.data.cpu().numpy())
        else:
            accuracy.append(percision.data.numpy())
    # accuracy ()
    model.train()
    # return (classes_size)
    # AUC
    AUC_scores = [AUC_list[i].value()[0] for i in range(opt.label_size)]
    return np.mean(accuracy, axis=0), AUC_scores # return the mean data, the accuracy for all six classes

def evaluate_mixed(model, clf, test_iter, opt):
    '''
    :param model: nn model extract text feature
    :param clf: svm classifier
    :param test_iter: dataset iter
    :param opt: option
    :return: accuracy
    '''
    model.eval()
    label_fetures = torch.LongTensor()
    global cum_tensor
    cum_tensor = torch.Tensor()
    for index, batch in enumerate(test_iter):
        if index > 2:  # for test use
            break
        text = batch.text[0] # for torchtext
        model(text)
        if opt.mix_model:
            label_fetures = torch.cat((label_fetures, batch.label))
    print(cum_tensor.size())
    if opt.use_cuda:
        test_X = cum_tensor.data.cpu().numpy()
        test_y = label_fetures.data.cpu().numpy()

    else:
        test_X = cum_tensor.data.numpy()
        test_y = label_fetures.data.numpy()
    print('Start prediction')
    print(test_X)
    print(test_X.size())
    pred_y = clf.predict(test_X)
    accuracy = accuracy_score(pred_y, test_y)

    model.train()
    return accuracy # return the mean data

def evaluate_mixed_f1(model, clf, test_iter, opt):
    '''
    :param model: nn model extract text feature
    :param clf: svm classifier
    :param test_iter: dataset iter
    :param opt: option
    :return: accuracy
    '''
    model.eval()
    label_fetures = torch.LongTensor()
    cum_tensor = torch.Tensor()

    for index, batch in enumerate(test_iter):
        text = batch.text[0] # for torchtext
        pred, festure_tensor = model(text) # festure_tensor (batch_size, festure_dim)
        cum_tensor = torch.cat((cum_tensor, festure_tensor))


    if opt.use_cuda:
        test_X = cum_tensor.data.cpu().numpy()
        test_y = label_fetures.data.cpu().numpy()
    else:
        test_X = cum_tensor.data.numpy()
        test_y = label_fetures.data.numpy()
    pred_y = clf.predict(test_X)
    accuracy = accuracy_score((pred_y, test_y))

    model.train()
    return accuracy # return the mean data

def validate(model, val_iter, criterion, opt):
    model.eval()
    loss_list = []
    for index, batch in enumerate(val_iter):
        text = batch.text[0] # for torchtext
        pred = model(text)

        loss = criterion(pred, batch.label)
        if opt.use_cuda:
            loss_data = loss.cpu().data
        else:
            loss_data = loss.data
        loss_list.append(loss_data)
    model.train()
    return np.mean(loss_list)
