# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import re

import spacy
import torch
from torchtext import data
from torchtext.vocab import GloVe
from sklearn.model_selection import KFold

# load and precess data for pytorch use

def concat_test_commetn_label():
    # for actually we have get the label, so we can concat them together
    test_FN = './data/test.csv'
    label_FN = './data/test_labels.csv'
    test_df = pd.read_csv(test_FN)
    test_label_df = pd.read_csv(label_FN)
    # concat together
    labels = test_label_df.columns[1:]
    for col in labels:
        test_df[col] = test_label_df[col]
    # filter record with label
    test_select = test_df['toxic'] != -1
    # save_new_test
    test_df[test_select].to_csv('./data/new_test.csv', index=False)


def handle_comment_text(comment):
    # remove punctuation in comment text
    comment = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ",str(comment))
    comment = re.sub(r"\r", "!", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    comment = re.sub(r"[ ]+", " ", comment)
    return comment

def preprocess_data_file(opt):
    seed = opt.seed
    # begin
    print('begin preprocess data')
    print('random seed is {}'.format(seed))
    val_rate = 0.2
    # split validation and training dataset from train.csv
    # handle
    train_FN = 'data/train.csv'
    test_FN = 'data/new_test.csv' # here use the concated text data file, not the original one
    # load data_file
    train_df = pd.read_csv(train_FN)
    test_df = pd.read_csv(test_FN)
    # in debug mode, only fetch 200 records
    if opt.debug is True:
        train_df = train_df.iloc[:300, :]
        test_df = test_df.iloc[:100, :]
    # handle comment text, prepare for tokenizer
    print('handle comment text')
    train_df.comment_text = train_df.comment_text.apply(handle_comment_text)
    test_df.comment_text = test_df.comment_text.apply(handle_comment_text)
    # split validation and training dataset
    # set random seed
    np.random.seed(seed)
    idx = np.arange(train_df.shape[0])
    np.random.shuffle(idx)
    # split
    val_len = int(val_rate*train_df.shape[0])
    # save result
    print('save result')
    process_dir = './data/processed/'
    train_df.to_csv('{}/train_processed.csv'.format(process_dir), index=False, encoding='utf-8')
    test_df.to_csv('{}/test.csv'.format(process_dir), index=False, encoding='utf-8')
    # for val and train
    train_df.iloc[val_len:, :].to_csv('{}/train.csv'.format(process_dir), index=False, encoding='utf-8')
    train_df.iloc[:val_len, :].to_csv('{}/val.csv'.format(process_dir), index=False, encoding='utf-8')
    # finish
    print('finish')

def load_data(opt):
    device = 0 if opt.use_cuda else -1
    print('preprocess data')
    if opt.data_is_preprocessed is False:
        # we need to preprocessed the data
        preprocess_data_file(opt)
    # define data.Field
    TEXT = data.Field(
        sequential=True,
        lower=True,
        batch_first=True,
        fix_length=opt.max_seq_len,
        # If "spacy", the SpaCy English tokenizer is used.
        tokenize='spacy', # use spacy the famous python NLP tokenizer
    )
    # for there are six label in the dataset, we need to define six field
    '''
    for TabularDataset
    path (str): Path to the data file.
    format (str): The format of the data file. One of "CSV", "TSV", or
        "JSON" (case-insensitive).
    fields (list(tuple(str, Field)) or dict[str: tuple(str, Field)]:
        If using a list, the format must be CSV or TSV, and the values of the list
        should be tuples of (name, field).
        The fields should be in the same order as the columns in the CSV or TSV
        file, while tuples of (name, None) represent columns that will be ignored.
        If using a dict, the keys should be a subset of the JSON keys or CSV/TSV
        columns, and the values should be tuples of (name, field).
        Keys not present in the input dictionary are ignored.
        This allows the user to rename columns from their JSON/CSV/TSV key names
        and also enables selecting a subset of columns to load.
    skip_header (bool): Whether to skip the first line of the input file.
    '''
    # generate dataset for pytorch
    # train, val, test
    process_dir = './data/processed/'
    print('generate dataset')
    train, val, test = data.TabularDataset.splits(
        path=process_dir,
        format='csv',
        skip_header=True,
        train='train.csv',
        validation='val.csv',
        test='test.csv',
        fields=[
            ('id', None),
            ('comment_text', TEXT),
            ('toxic', data.Field(
                use_vocab=False, sequential=False,
                tensor_type=torch.ByteTensor)),
            ('severe_toxic', data.Field(
                use_vocab=False, sequential=False,
                tensor_type=torch.ByteTensor)),
            ('obscene', data.Field(
                use_vocab=False, sequential=False,
                tensor_type=torch.ByteTensor)),
            ('threat', data.Field(
                use_vocab=False, sequential=False,
                tensor_type=torch.ByteTensor)),
            ('insult', data.Field(
                use_vocab=False, sequential=False,
                tensor_type=torch.ByteTensor)),
            ('identity_hate', data.Field(
                use_vocab=False, sequential=False,
                tensor_type=torch.ByteTensor)),
        ]
    )
    # build vocab for TEXT
    '''
    '''
    print('generate vocab')
    TEXT.build_vocab(train, val, test, min_freq=10,
                     vectors=GloVe(name='6B', dim=300))
    # update opt information
    print('update vocab and embedding')
    opt.vocab_size = len(TEXT.vocab)
    opt.label_size = 6
    opt.embedding_dim = TEXT.vocab.vectors.size()[1]
    opt.embeddings = TEXT.vocab.vectors
    print('finish dataset iterator')
    # generate iterator for pytorch
    # train_iter, val_iter,test_iter = data.BucketIterator.splits((train, val, test), batch_size=opt.batch_size, device=device, repeat=False, shuffle=True)

    train_iter = get_iterator(train, batch_size=opt.batch_size, device= device,train=True, shuffle=True, repeat=False)
    test_iter = get_iterator(test, batch_size=opt.batch_size, device= device,train=True, shuffle=True, repeat=False)
    val_iter = get_iterator(val, batch_size=opt.batch_size, device= device,train=True, shuffle=True, repeat=False)
    # finish loading data
    return train_iter, val_iter, test_iter

def get_iterator(dataset, batch_size, train=True, device=0,
    shuffle=True, repeat=False):
    dataset_iter = data.Iterator(
        dataset, batch_size=batch_size, device=device,
        train=train, shuffle=shuffle, repeat=repeat,
        sort=False
    )
    return dataset_iter