# -*- coding: UTF-8 -*-

import torch
import torch.nn.functional as F
import torchtext
import time
import opts
from utils import clip_gradient, evaluate, validate
import models
from visualize import Visualizer
from tqdm import tqdm
from tmp import load_data



# get option
opt = opts.parse_opt()

# opt.debug = True
# opt.debug_iterator = True

# opt.debug in this mode, only

opt.debug = False
opt.debug_iterator = False # check for data generator


opt.one_hot = False
# True for use one_hot embedding, False for use pretrainded
# one_hot only for

# opt.use_cuda = torch.cuda.is_available()
opt.use_cuda = True

# process
opt.data_is_preprocessed = False
# random seed
opt.seed = 666
opt.dataset = 'kaggle'
# select model
# opt.model = 'lstm'
# opt.model = 'cnn'

opt.model = 'bilstm'

opt.env = opt.model + '_clf'

if opt.one_hot is True:
    opt.env += '_one_hot'

# visdom
vis = Visualizer(opt.env)

# vis log output
vis.log('user config:')
for k, v in opt.__dict__.items():
    if not k.startswith('__'):
        vis.log('{} {}'.format(k, getattr(opt, k)))

# load data
# use torchtext to load
train_iter, val_iter, test_iter = load_data(opt)

model = models.init(opt)

print(type(model))


# debug for iterator
if opt.debug_iterator is True:
    for data_iter in [train_iter, test_iter, val_iter]:
        for train_epoch, batch in enumerate(data_iter):
            text = batch.comment_text.data
            label = torch.stack([
                batch.toxic, batch.severe_toxic, batch.obscene,
                batch.threat, batch.insult, batch.identity_hate
            ], dim=1)

            label = label.float()

            pred = model(text)

            if train_epoch == 0:
                print(label.size())
                print(text.size())
                print(pred.size())
                print(label)
                print(pred)
    exit()

# cuda
if opt.use_cuda:
    model.cuda()
if opt.debug:
    print(model)
# start trainning
model.train()
# set optimizer
optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
optim.zero_grad()
# use cross_entropy loss for classification
criterion = F.multilabel_soft_margin_loss
# save best model use accuracy evaluation metrics
best_accuaracy = 0

for idx_epoch in range(opt.max_epoch):
    for train_epoch, batch in enumerate(train_iter):
        start_epoch = time.time()
        # print(batch.label.size())
        # print(opt.batch_size)
        # # if batch.label.size()[0] != opt.batch_size:
        # #     continue
        # for torchtext
        # laod text and predict label from batch
        text = batch.comment_text.data
        label = torch.stack([
            batch.toxic, batch.severe_toxic, batch.obscene,
            batch.threat, batch.insult, batch.identity_hate
        ], dim=1)

        label = label.float()


        # print('text size : {}'.format(text.size()))
        # print('label size : {}'.format(label.size()))

        pred = model(text)

        if train_epoch == 0 and idx_epoch == 0:
            print(label.size())
            print(text.size())
            print(pred.size())
            print(label)
            print(pred)

        loss = criterion(pred, label)

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
            print("{} EPOCH {} batch: train loss {}".format(idx_epoch, train_epoch, loss_data))

            # vis loss
            vis.plot('loss', loss_data)


    # evaluate on test for this epoch
    # accuracy np.array (classes_size)
    accuracy, AUC_scores = evaluate(model, test_iter, opt) # TODO update load data in evaluate and validation
    vis.log("{} EPOCH, accuaracy : {}".format(idx_epoch, accuracy.mean()))
    vis.plot('accuracy', accuracy)
    label_list = ['toxic', 'severe_toxic', 'obscene',
            'threat', 'insult', 'identity_hate']
    print(accuracy)
    for i in range(len(label_list)):
        vis.plot(label_list[i], accuracy[i])
    # for AUC
    print(AUC_scores)
    vis.plot('AUC', AUC_scores)
    for i in range(len(label_list)):
        vis.plot('AUC_{}'.format(label_list[i]), AUC_scores[i])

    # handel best model, update best model , best_lstm.pth
    if accuracy.mean() > best_accuaracy:
        best_accuaracy = accuracy.mean()
        torch.save(model.state_dict(), './best_{}.pth'.format(opt.model))

print('best accuracy: {}'.format(best_accuaracy))