# -*- coding: UTF-8 -*-

#coding:utf8
from config import opt

import os
import torch
import models

from Dataset import Flower
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from torchnet import meter
import numpy as np
from utils.visualize import Visualizer
from utils.weight import weight_init
import torchvision as tv


from tqdm import tqdm


def train(**kwargs):
    # load kwargs
    opt.parse(kwargs)
    print(kwargs)

    # visdom
    vis = Visualizer(opt.env)

    # vis log opt
    vis.log('user config:')
    for k, v in opt.__class__.__dict__.items():
        if not k.startswith('__'):
            vis.log('{} {}'.format(k, getattr(opt, k)))

    # config model
    model = getattr(models, opt.model)()

    if opt.use_pretrained_model:
        model = load_pretrained()

    if opt.load_model_path:
        # load exist model
        model.load(opt.load_model_path)
    elif opt.use_weight_init:
        # we need init weight
        #
        model.apply(weight_init)
    # if use GPU
    if opt.use_gpu:
        model.cuda()

    # genearte_data
    train_data = Flower(train=True)
    val_data = Flower(train=False)
    test_data = Flower(test=True)

    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_dataloader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    if 'Dense' in opt.model:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True,
                                    weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)


    # meters
    loss_meter = meter.AverageValueMeter()
    # 17 classes
    confusion_matrix = meter.ConfusionMeter(17)
    previous_loss = 1e100

    #

    best_accuracy = 0

    # start training
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        for bactch_index, (data, label) in tqdm(enumerate(train_dataloader)):

            # train model
            input = Variable(data)
            target = Variable(label)
            # gpu update
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()


            # update meter
            loss_meter.add(loss.data[0])

            # print(score.data, target.data)
            #      [batch_size, 17]  [batch_size]
            confusion_matrix.add(score.data, target.data)

            # plot
            if bactch_index % opt.print_freq == opt.print_freq-1:
                # cross_entropy
                print('loss ', loss_meter.value()[0])
                # visualize loss
                vis.plot('loss', loss_meter.value()[0])

        # save model for this epoch
        if opt.use_pretrained_model is False and epoch % opt.save_freq == 0:
            model.save()

        # validate
        val_cm, val_accuracy = val(model, val_dataloader)

        # test
        test_cm, test_accuracy = val(model, test_dataloader)

        # plot validation accuracy
        print('Epoch {}/{}: val_accuracy  {}'.format(epoch, opt.max_epoch, val_accuracy))

        # plot vis
        vis.plot('val_accuracy', val_accuracy)
        vis.plot('test_accuracy', test_accuracy)
        vis.log('epoch:{epoch}, lr:{lr}, loss:{loss}'.format(
            epoch=epoch,
            loss=loss_meter.value()[0],
            lr=lr)
        )
        # vis.log('epoch:{epoch}, lr:{lr}, loss:{loss}, train_cm:{train_cm}, val_cm:{val_cm}'.format(
        #     epoch=epoch, loss=loss_meter.value()[0], val_cm = str(val_cm.value()),train_cm=str(confusion_matrix.value()),lr=lr)
        # )

        # update best validation model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), './checkpoints/best_{}.pth'.format(opt.model))
            if opt.use_pretrained_model is False:
                model.save('./checkpoints/best_{}.pth'.format(model.model_name))


        # update learning rate for this epoch
        if float(loss_meter.value()[0]) > previous_loss:
            lr = lr * opt.lr_decay

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]
    print('Best model validation accuracy {}'.format(best_accuracy))

def test_for_train(model):
    test_data = Flower(test=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    model.eval()
    confusion_matrix = meter.ConfusionMeter(17)
    # use argmax way: choice the max index as the prediction
    correct_sum = 0
    total_cnt = 0
    pass


def val(model, dataloader):
    # accuracy in validation dataset
    model.eval()
    confusion_matrix = meter.ConfusionMeter(17)
    # use argmax way: choice the max index as the prediction
    correct_sum = 0
    total_cnt = 0

    for bactch_index, (input, label) in tqdm(enumerate(dataloader)):

        val_input = Variable(input, volatile=True)
        # update gpu
        if opt.use_gpu:
            val_input = val_input.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(), label.type(torch.LongTensor))

        # armax way
        pred = score.data.max(1)[1].cpu()
        correct = pred.eq(label).sum()
        correct_sum+=correct
        total_cnt+=input.size(0)


    model.train()
    cm_value = confusion_matrix.value()
    # confusion matrix accuracy
    accuracy = 100. * (sum(np.diag(cm_value))) / (cm_value.sum())
    # another accuracy way
    #accuracy = 100*float(correct_sum)/float(total_cnt)
    #print('{} / {}'.format(correct_sum, total_cnt))
    return confusion_matrix, accuracy


def test(**kwargs):
    opt.parse(kwargs)
    model = getattr(models, opt.model)().eval()
    # load model path
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_pretrained_model:
        model = load_pretrained_for_test()
    # use cuda
    if opt.use_gpu:
        model.cuda()
    # data
    test_data = Flower(test=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []

    model.eval()

    confusion_matrix = meter.ConfusionMeter(17)
    # use argmax way: choice the max index as the prediction
    correct_sum = 0
    total_cnt = 0

    for batch_index, (data, label) in tqdm(enumerate(test_dataloader)):
        input = Variable(data, volatile=True)
        # gpu
        if opt.use_gpu:
            input = input.cuda()
        score = model(input)
        confusion_matrix.add(score.data.squeeze(), label.type(torch.LongTensor))

        # armax way
        pred = score.data.max(1)[1].cpu()
        correct = pred.eq(label).sum()
        correct_sum += correct
        total_cnt += input.size(0)
    accuracy = 100*float(correct_sum)/float(total_cnt)
    print('{} / {}'.format(correct_sum, total_cnt))
    print(accuracy)
    return accuracy

def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)

def load_pretrained():
    pretrained_model = tv.models.resnet34(pretrained=True)
    pretrained_dict = pretrained_model.state_dict()

    model = tv.models.resnet34(num_classes=17)
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k, v in pretrained_dict.items() if 'fc' not in k }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model

def load_pretrained_for_test():
    path = './checkpoints/best_ResNet34_with_pretrain.pth'
    model = tv.models.resnet34(num_classes=17)
    model.load_state_dict(torch.load(path))
    return model



if __name__ == '__main__':
    import fire
    fire.Fire()
    #train()