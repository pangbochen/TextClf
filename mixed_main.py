# -*- coding: UTF-8 -*-

import torch
import torch.nn.functional as F
import torchtext
import time
import opts
from utils import load_data, clip_gradient, evaluate, validate
import models
from visualize import Visualizer
from tqdm import tqdm
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score

# 使用方法3），使用hook得到网络中间层

def evaluate_mixed(model, clf, test_iter, opt):
    '''
    :param model: nn model extract text feature
    :param clf: svm classifier
    :param test_iter: dataset iter
    :param opt: option
    :return: accuracy
    '''
    model.eval()
    global cum_tensor
    if opt.use_cuda:
        cum_tensor = torch.Tensor().cuda()
        label_fetures = torch.LongTensor().cuda()
    else:
        cum_tensor = torch.Tensor()
        label_fetures = torch.LongTensor()
    for index, batch in enumerate(test_iter):
        # if index > 2:  # for test use
        #     break
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
    print(test_X.size)
    pred_y = clf.predict(test_X)
    accuracy = accuracy_score(pred_y, test_y)

    model.train()
    return accuracy # return the mean data

def for_hook_v1(modeule, input, output):
    # print(modeule)
    # print(output[0].shape)
    global cum_tensor
    cum_tensor = torch.cat((cum_tensor, output[0].permute(1, 0, 2)))
    return None

def for_hook_v2(modeule, input, output):
    # cat the input
    global cum_tensor
    # print(input[0].size())
    # print(cum_tensor.size())
    cum_tensor = torch.cat((cum_tensor, input[0]))
    return None

if __name__ == '__main__':
    # get option
    opt = opts.parse_opt()

    # opt.use_cuda = torch.cuda.is_available()
    opt.use_cuda = True
    # 是否适用混合模型
    opt.mix_model = True

    feature_clf = svm.SVC()

    # select model
    opt.model = 'lstm'

    opt.env = 'mixed_'+opt.model

    # setting for mixed_model
    global cum_tensor
    if opt.use_cuda:
        cum_tensor = torch.Tensor().cuda()
    else:
        cum_tensor = torch.Tensor()


    # visdom
    vis = Visualizer(opt.env)

    # vis log output
    vis.log('user config:')
    for k, v in opt.__dict__.items():
        if not k.startswith('__'):
            vis.log('{} {}'.format(k, getattr(opt, k)))

    # load data
    # use torchtext to load
    train_iter, test_iter = load_data(opt)

    model = models.init(opt)
    # register hook for get
    # model.lstm.register_forward_hook(for_hook_v1)
    handle = model.label.register_forward_hook(for_hook_v2)

    print(type(model))

    # cuda
    if opt.use_cuda:
        model.cuda()

    # start trainning
    model.train()
    # set optimizer
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    optim.zero_grad()
    # use cross_entropy loss for classification
    criterion = F.cross_entropy
    # save best model use accuracy evaluation metrics
    best_accuaracy = 0

    for i in range(opt.max_epoch):
        if opt.mix_model:
            if opt.use_cuda:
                cum_tensor = torch.Tensor().cuda()
                label_fetures = torch.LongTensor().cuda()
            else:
                cum_tensor = torch.Tensor()
                label_fetures = torch.LongTensor()
        print("Start training Epoch {} ".format(i))
        for train_epoch, batch in enumerate(train_iter):
            # for test use
            # if train_epoch > 3:
            #     break

            start_epoch = time.time()
            # print(batch.label.size())
            # print(opt.batch_size)
            # # if batch.label.size()[0] != opt.batch_size:
            # #     continue
            # for torchtext
            # update label for mixed model
            if opt.mix_model:
                label_fetures = torch.cat((label_fetures, batch.label))

            text = batch.text[0]

            pred = model(text)

            loss = criterion(pred, batch.label)

            loss.backward()

            # trainint trick : clip_gradient
            # https://blog.csdn.net/u010814042/article/details/76154391
            # solve Gradient explosion problem
            clip_gradient(optimizer=optim, grad_clip=opt.grad_clip)

            # step optimizer
            optim.step()

            # plot for loss and accuracy
            if train_epoch % 50 == 0:
                if opt.use_cuda:
                    loss_data = loss.cpu().data
                else:
                    loss_data = loss.data
                print("{} EPOCH {} batch: train loss {}".format(i, train_epoch, loss_data))

                # vis loss
                vis.plot('loss', loss_data)

        # handle mix classification model problem
        if opt.mix_model:
            # train model
            print(cum_tensor.size())
            if opt.use_cuda:
                train_X = cum_tensor.data.cpu().numpy()
                train_y = label_fetures.data.cpu().numpy()
            else:
                train_X = cum_tensor.data.numpy()
                train_y = label_fetures.data.numpy()
            print(train_X.size)
            print(train_y.size)
            feature_clf.fit(train_X, train_y)
            # finish model
            print("Finish model for epoch {}".format(i))
            # evaluate model with custom classification model
            # erase cum_features
            if opt.use_cuda:
                cum_tensor = torch.Tensor().cuda()
            else:
                cum_tensor = torch.Tensor()
            print(cum_tensor.size())
            mixed_model_accuaracy = evaluate_mixed(model, feature_clf, test_iter, opt)
            vis.log("{} EPOCH, mixed model accuaracy : {}".format(i, mixed_model_accuaracy))
            vis.plot('mixed model accuracy', mixed_model_accuaracy)
            print("{} EPOCH, mixed model accuaracy : {}".format(i, mixed_model_accuaracy))

        # # evaluate on test for this epoch use the fc layer of nn as the classifier
        # accuracy = evaluate(model, test_iter, opt)
        # vis.log("{} EPOCH, accuaracy : {}".format(i, accuracy))
        # vis.plot('accuracy', accuracy)
        #
        # # handel best model, update best model , best_lstm.pth
        # if accuracy > best_accuaracy:
        #     best_accuaracy = accuracy
        #     torch.save(model.state_dict(), './best_{}.pth'.format(opt.model))

    print('best accuracy: {}'.format(best_accuaracy))