#-*- coding: utf-8 -*-
import os
import sys
import time
import math
import wavio
import argparse
import queue
import shutil
import random
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim

import label_loader
from edit_distance import *
from loader import *
from models import EncoderRNN, DecoderRNN, Seq2seq

import nsml
from nsml import GPU_NUM, DATASET_PATH, DATASET_NAME, HAS_DATASET

from platform import python_version

char2index = dict()
index2char = dict()
SOS_token = 0
EOS_token = 0
PAD_token = 0

#DATASET_PATH = '../../lck_nipa_2019_speech_100000/'
DATASET_PATH = os.path.join(DATASET_PATH, 'train')

def label_to_string(labels):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents

def get_distance(ref_labels, hyp_labels, display=False):
    total_dist = 0
    total_corr = 0
    for i in range(len(ref_labels)):
        ref = label_to_string(ref_labels[i])
        hyp = label_to_string(hyp_labels[i])
        cer, dist, corr  = char_distance_error(ref, hyp)
        total_dist += dist
        total_corr += corr
        if display:
            debug('%d (%0.4f)\n(%s)\n(%s)' % (i, cer, ref, hyp))
    return total_dist, total_corr

def train(model, total_batch_size, queue, criterion, optimizer, device, train_begin, train_loader_count, print_batch=5, teacher_forcing_ratio=1):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_corr = 0
    total_sent_num = 0
    batch = 0

    model.train()

    info('train start')

    begin = epoch_begin = time.time()

    while True:
        #debug('queue size: %d' % (queue.qsize()))
        feats, scripts, feat_lengths, script_lengths = queue.get()

        if feats.shape[0] == 0:
            # empty feats means closing one loader
            train_loader_count -= 1

            debug('left train_loader: %d' % (train_loader_count))

            if train_loader_count == 0:
                break
            else:
                continue

        optimizer.zero_grad()

        feats = feats.to(device)
        scripts = scripts.to(device)

        src_len = scripts.size(1)
        target = scripts[:, 1:]

        model.module.flatten_parameters()
        logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=teacher_forcing_ratio)

        logit = torch.stack(logit, dim=1).to(device)
        y_hat = logit.max(-1)[1]

        loss = criterion(logit.view(-1, logit.size(-1)), target.contiguous().view(-1))
        total_loss += loss.item()
        total_num += sum(feat_lengths)

        dist, corr = get_distance(target, y_hat)
        total_dist += dist
        total_corr += corr

        total_sent_num += target.size(0)

        loss.backward()
        optimizer.step()

        if batch % print_batch == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            info('batch: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'
                .format(batch,
                        #len(dataloader),
                        total_batch_size,
                        total_loss/float(total_num),
                        total_dist/float(total_dist + total_corr),
                        elapsed, epoch_elapsed, train_elapsed))
            begin = time.time()
        batch += 1

    info('train completed')
    return total_loss / float(total_num), total_dist / float(total_dist + total_corr)

def evaluate(model, dataloader, queue, criterion, device):
    info('evaluate start')
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_corr = 0
    total_sent_num = 0
    model.eval()
    with torch.no_grad():
        while True:
            feats, scripts, feat_lengths, script_lengths = queue.get()
            if feats.shape[0] == 0:
                break

            feats = feats.to(device)
            scripts = scripts.to(device)

            src_len = scripts.size(1)
            target = scripts[:, 1:]

            logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=0)
            logit = torch.stack(logit, dim=1).to(device)
            y_hat = logit.max(-1)[1]

            loss = criterion(logit.view(-1, logit.size(-1)), target.contiguous().view(-1))
            total_loss += loss.item()
            total_num += sum(feat_lengths)

            disp = random.randrange(0, 100) == 0
            dist, corr = get_distance(target, y_hat, display=disp)
            total_dist += dist
            total_corr += corr
            total_sent_num += target.size(0)

    info('evaluate completed')
    return total_loss / float(total_num), total_dist / float(total_dist + total_corr)

def bind_model(model, optimizer=None):
    def load(filename, **kwargs):
        state = torch.load(os.path.join(filename, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    def save(filename, **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(filename, 'model.pt'))

    def infer(wav_path):
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input = get_feature_from_librosa(wav_path, 40).unsqueeze(0)
        input = input.to(device)

        logit = model(input_variable=input, input_lengths=None, teacher_forcing_ratio=0)
        logit = torch.stack(logit, dim=1).to(device)

        y_hat = logit.max(-1)[1]
        hyp = label_to_string(y_hat)

        return hyp[0]

    nsml.bind(save=save, load=load, infer=infer) # 'nsml.bind' function must be called at the end.

def split_dataset(config, wav_paths, script_paths, valid_ratio=0.05):
    train_loader_count = config.workers
    records_num = len(wav_paths)
    batch_num = math.ceil(records_num / config.batch_size)

    valid_batch_num = math.ceil(batch_num * valid_ratio)
    train_batch_num = batch_num - valid_batch_num

    batch_num_per_train_loader = math.ceil(train_batch_num / config.workers)

    train_begin = 0
    train_end_raw_id = 0
    train_dataset_list = list()

    for i in range(config.workers):

        train_end = min(train_begin + batch_num_per_train_loader, train_batch_num)

        train_begin_raw_id = train_begin * config.batch_size
        train_end_raw_id = train_end * config.batch_size

        train_dataset_list.append(BaseDataset(
                                        wav_paths[train_begin_raw_id:train_end_raw_id],
                                        script_paths[train_begin_raw_id:train_end_raw_id],
                                        config.feature_size, SOS_token, EOS_token))
        train_begin = train_end 

    valid_dataset = BaseDataset(wav_paths[train_end_raw_id:], script_paths[train_end_raw_id:], config.feature_size, SOS_token, EOS_token)

    return train_batch_num, train_dataset_list, valid_dataset

def main():

    info('python version: %s' % (python_version()))

    global char2index
    global index2char
    global SOS_token
    global EOS_token
    global PAD_token

    parser = argparse.ArgumentParser(description='NIPA Speech Recognition Baseline')
    parser.add_argument('--feature_size', type=int, default=40, help='size of MFCC feature (default: 40)')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size of model (default: 256)')
    parser.add_argument('--layer_size', type=int, default=3, help='number of layers of model (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate in training (default: 0.2)')
    parser.add_argument('--bidirectional', action='store_true', help='use bidirectional RNN for encoder (default: False)')
    parser.add_argument('--use_attention', action='store_true', help='use attention between encoder-decoder (default: False)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training (default: 32)')
    parser.add_argument('--workers', type=int, default=4, help='number of workers in dataset loader (default: 4)')
    parser.add_argument('--max_epochs', type=int, default=100, help='number of max epochs in training (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-04, help='learning rate (default: 0.0001)')
    parser.add_argument('--teacher_forcing', type=float, default=0.5, help='teacher forcing ratio in decoder (default: 0.5)')
    parser.add_argument('--max_len', type=int, default=80, help='maximum characters of sentence (default: 80)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save_name', type=str, default='model', help='the name of model in nsml, best model name is \'best\'')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument("--pause", type=int, default=0)

    args = parser.parse_args()

    char2index, index2char = label_loader.load_label('./script.labels')
    SOS_token = char2index['<s>']
    EOS_token = char2index['</s>']
    PAD_token = char2index['_']

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    enc = EncoderRNN(args.feature_size, args.hidden_size,
                     input_dropout_p=args.dropout, dropout_p=args.dropout,
                     n_layers=args.layer_size, bidirectional=args.bidirectional, rnn_cell='gru', variable_lengths=False)

    dec = DecoderRNN(len(char2index), args.max_len, args.hidden_size * (2 if args.bidirectional else 1),
                     SOS_token, EOS_token,
                     n_layers=args.layer_size, rnn_cell='gru', bidirectional=args.bidirectional,
                     input_dropout_p=args.dropout, dropout_p=args.dropout, use_attention=args.use_attention)

    model = Seq2seq(enc, dec)
    model.flatten_parameters()

    for param in model.parameters():
        param.data.uniform_(-0.08, 0.08)

    model = nn.DataParallel(model).to(device)

    optimizer = optim.Adam(model.module.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)

    bind_model(model, optimizer)

    if args.pause == 1:
        nsml.paused(scope=locals())

    if args.mode != "train":
        return

    data_list = os.path.join(DATASET_PATH, 'train_data', 'data_list.csv')
    wav_paths = list()
    script_paths = list()

    with open(data_list, 'r') as f:
        for line in f:
            # line: "aaa.wav,aaa.label"

            wav_path, script_path = line.strip().split(',')
            wav_paths.append(os.path.join(DATASET_PATH, 'train_data', wav_path))
            script_paths.append(os.path.join(DATASET_PATH, 'train_data', script_path))

    best_loss = 1e10
    begin_epoch = 0

    # load all target scripts for reducing disk i/o
    target_path = os.path.join(DATASET_PATH, 'train_label')
    load_targets(target_path)

    train_batch_num, train_dataset_list, valid_dataset = split_dataset(args, wav_paths, script_paths, valid_ratio=0.05)

    info('start')

    train_begin = time.time()

    for epoch in range(begin_epoch, args.max_epochs):

        train_queue = queue.Queue(args.workers * 2)

        train_loader = MultiLoader(train_dataset_list, train_queue, args.batch_size, args.workers)
        train_loader.start()

        train_loss, train_cer = train(model, train_batch_num, train_queue, criterion, optimizer, device, train_begin, args.workers, 10, args.teacher_forcing)
        info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

        train_loader.join()

        model.module.flatten_parameters()

        valid_queue = queue.Queue(args.workers * 2)
        valid_loader = BaseDataLoader(valid_dataset, valid_queue, args.batch_size, 0)
        valid_loader.start()

        eval_loss, eval_cer = evaluate(model, valid_loader, valid_queue, criterion, device)
        info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, eval_loss, eval_cer))

        valid_loader.join()

        nsml.report(False,
            step=epoch, train_loss=train_loss, train_cer=train_cer,
            eval_loss=eval_loss, eval_cer=eval_cer)

        best_model = (eval_loss < best_loss)
        nsml.save(args.save_name)

        if best_model:
            nsml.save('best')
            best_loss = eval_loss

if __name__ == "__main__":
    main()
