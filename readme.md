文本分类

对Kaggle数据集进行文本的分类任务

Kaggle网页链接

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

## environment

python 3.6

pytorch & torchtext & visdom

## dataset analysis

对数据集进行文本分析

具体的部分见data_view.ipynb文件中

## data process

1. 对数据进行预处理
2. 生成dataset和data iterator for pytorch

## experiment
就以下几点进行尝试
- embedding view
    - one-hot
    - normal embedding：word2vec，glove
    - fasttext
    - nn-based function：allen nlp ，etc
- nn model
    - lstm
    - bilstm
    - cnn
- mixed mdel : use nn_model as factor extraction
    - nn_model + svm
    - nn_model + lr or other classic classification model
- curent state of art methods

## training related

for it is six-label binary classification task

loss

accuracy

    calcalate the accuracy for six binary-classification tasks

    while accuracy may not be a perfect quota

    for the 0/1 is not balanced

    AUC is more meaningful

AUC

    later I calculate the AUC for each classification task

## vis

- log
- loss
- accuracy : the mean accuracy
- accuracy for 6 different labels, take each label task as the bianary classification task
    - toxic
    - severe_toxic
    - obscene
    - threat
    - insult
    - identity_hate
-


## Result:
- cnn : best accuracy on test dataset: 0.9652185440063477
- lstm : best accuracy: 0.9680070877075195
    - time : 22min
    - epoch : 40
    - embedding :

