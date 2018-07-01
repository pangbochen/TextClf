# -*- coding: UTF-8 -*-

import torch
import torch.nn.functional as F
import torchtext
import time
import opts
from utils import load_data, clip_gradient, evaluate, validate, evaluate_mixed_f1
import models
from visualize import Visualizer
from tqdm import tqdm
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score

# 使用方法1实现模型的混合

feature_clf = svm.SVC()

# get option
opt = opts.parse_opt()

# opt.use_cuda = torch.cuda.is_available()
opt.use_cuda = False

# select model
opt.model = 'lstm_mixed'

opt.env = opt.model

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
    # 和main.py不同使用mixed model
    cum_tensor = torch.Tensor()
    label_fetures = torch.LongTensor()

    for train_epoch, batch in enumerate(train_iter):
        start_epoch = time.time()
        # print(batch.label.size())
        # print(opt.batch_size)
        # # if batch.label.size()[0] != opt.batch_size:
        # #     continue
        # for torchtext
        text = batch.text[0]

        pred, feature_tensor = model(text)

        # update cum_tensor
        cum_tensor = torch.cat((cum_tensor, feature_tensor))
        label_fetures = torch.cat((label_fetures, batch.label))

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
                loss_data = loss.cpu().data[0]
            else:
                loss_data = loss.data[0]
            print("{} EPOCH {} batch: train loss {}".format(i, train_epoch, loss_data))

            # vis loss
            vis.plot('loss', loss_data)


    # evaluate on test for this epoch
    accuracy = evaluate(model, test_iter, opt)
    vis.log("{} EPOCH, accuaracy : {}".format(i, accuracy))
    vis.plot('accuracy', accuracy)

    # handel best model, update best model , best_lstm.pth
    if accuracy > best_accuaracy:
        best_accuaracy = accuracy
        torch.save(model.state_dict(), './best_{}.pth'.format(opt.model))

    # for mixed classification part
    print(cum_tensor.size())
    feature_tensor = cum_tensor.mean(dim=1)
    print(feature_tensor.size())
    if opt.use_cuda:
        train_X = feature_tensor.data.cpu().numpy()
        train_y = label_fetures.data.cpu().numpy()
    else:
        train_X = feature_tensor.data.numpy()
        train_y = label_fetures.data.numpy()
    feature_clf.fit(train_X, train_y)
    # evaluate model with custom classification model
    # erase cum_features
    cum_tensor = torch.Tensor()
    print(cum_tensor.size())
    mixed_model_accuaracy = evaluate_mixed_f1(model, feature_clf, test_iter, opt)
    vis.log("{} EPOCH, mixed model accuaracy : {}".format(i, accuracy))
    vis.plot('mixed model accuracy', accuracy)

print('best accuracy: {}'.format(best_accuaracy))