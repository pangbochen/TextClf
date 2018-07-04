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

一个重要的数据集性质：两类样本分布不平均，大部分comment都是正常评论，label大多为0


## data process

1. 对数据进行预处理
2. 生成test数据集
3. 生成dataset和data iterator for pytorch
4. 加载glove编码模型
5. 生成vocabulary
6. pytorch中，字符编码通过Embedding的网络层来实现，加载embedding.weight

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
- self-attention : best accuracy: 0.9622352123260498
    - self attention is very usefull for this task
- bilstm : best accuracy: 0.9641895890235901



## loss function

it is a multi-label binary classification task.

- one idea is to treat it as six seperate binary-classification tasks, train six different models
- another idea is to train six tasks together,

I think the latter is more convincing for me, as the six tasks are jointly learned.

I use F.multilabel_soft_margin_loss in pytorch

Detail for this loss is in https://pytorch.org/docs/stable/nn.html?highlight=multilabel_soft_margin_loss#torch.nn.MultiLabelSoftMarginLoss

- input : (N,C), (batch_size, number of classes)
- target : (N, C), same shape as the input
- output : scalar, if reduce is True, then (N)

## some details in the experiment

Embedding:

    in pytorch, one-hot embedding is just nn.Embedding() layer with weight of torch.dim()

    https://lirnli.wordpress.com/2017/09/03/one-hot-encoding-in-pytorch/

    def one_hot_v3(batch,depth):
        emb = nn.Embedding(depth, depth)
        emb.weight.data = torch.eye(depth)
        return emb(batch)

    while in practice, one-hot is not good way

    use pretrainded embedding can enhance model performance

    I do the comparison experiment on LSTM model (one-hot vs glove model)

AUC and Accuracy

    A

## embedding comparation

lstm

one-hot

    it is more slower for the original of one-hot , the embedding_dim is vocab_size, too large
    and it is too slow, around 10min 50 batch

glove



## Mixed model

    I also mixed, use mixed model

    the classification method I used in this program is

    sklearn multilabel algorithm

    Test on lstm model:
        lstm model will
            hidden state : for training the classifier
            lstm classificastion result : for training model

    For classifer in the mixed model
        Input : hidden dim of lstm model
        Output : is_label for six label (num_of_label)
    Example :
        X : ([127657, 128])
        y : ([127657, 6]))

## Allen-nlp

    for

## self-attention

for self-attention

    The first question is, what does the lstm model do?
        embedding the sentence or we called sequence into the feature vector.

    in the previous lstm based model
    the embedding feature of the whole sequence is the hidden state of last token or the mean of all token's hidden states
    while it is not a good way to asign the specific rule to generate the sequence feature.

    self-attention provides one way:
        self-attention will generate the (batch_size, seq_len, 1)
        that the weight for each token in the sequence to generate the final sequence embedding

