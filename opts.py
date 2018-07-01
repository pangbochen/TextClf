# -*- coding: UTF-8 -*-
import argparse
import os

def parse_opt():
    parser = argparse.ArgumentParser()

    # Visdom setting
    parser.add_argument('--env', type=str, default='default',
                        help='visdom environment')
    # Model settings
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='dims of hedden layers for the lstm layer')
    parser.add_argument('--max_seq_len', type=int, default=200,
                        help='max length of the sentence')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--embedding_dim', type=int, default=100,
                        help='for the embedding layer (vocab_size, embedding_size)')
    parser.add_argument('--grad_clip', type=float, default=1e-1,
                        help='grad_clip for clip_gradient training trick')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='learning rate')
    parser.add_argument('--model', type=str, default="lstm",
                        help='model name, lstm, cnn, attention, bilstm, etc')
    parser.add_argument('--dataset', type=str, default="imdb",
                        help='name of dataset, imdb as default')
    parser.add_argument('--keep_dropout', type=float, default=0.8,
                        help='rate of keep dropout')
    parser.add_argument('--max_epoch', type=int, default=40,
                        help='number of training epoch')
    parser.add_argument('--embedding_file', type=str, default="glove.6b.300",
                        help='the pretrained embedding file, for torchtext, glove as default')
    parser.add_argument('--embedding_training', type=str, default="false",
                        help='the weight of embedding layer in pytorch can be trained')
    # CNN network setting for context extracting
    parser.add_argument('--kernel_sizes', type=str, default="1,3,3",
                        help='kernel_sizes')
    parser.add_argument('--kernel_nums', type=str, default="256,256,256",
                        help='kernel_nums')
    parser.add_argument('--lstm_mean', type=str, default="mean",  # last mean
                        help='lstm_mean')
    parser.add_argument('--lstm_layers', type=int, default=1,
                        help='number of lstm_layers')
    parser.add_argument('--embedding_dir', type=str, default=".glove/glove.6B.300d.txt",
                        help='embedding_dir')
    parser.add_argument('--from_torchtext', type=str, default="false",
                        help='from torchtext ')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu id')
    #
    args = parser.parse_args()
    # update for cnn configs
    args.kernel_sizes = [int(i) for i in args.kernel_sizes.split(",")]
    args.kernel_nums = [int(i) for i in args.kernel_nums.split(",")]

    if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # update bool setting
    if args.embedding_training.lower() == "true":
        args.embedding_training = True
    else:
        args.embedding_training = False
    if args.from_torchtext.lower() == "true":
        args.from_torchtext = True
    else:
        args.from_torchtext = False
    return args

