#-*- coding: utf-8 -*-
import os
import sys
import math
import wavio
import time
import torch
import random
import librosa
import threading
from python_speech_features import mfcc
from torch.utils.data import Dataset, DataLoader
from log import *

# from label file
PAD = 0

target_dict = dict()

first = True
sig = list()
smaple_rate = 16000

def load_targets(path):
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            target_dict[key] = target

def get_feature_from_librosa(filepath, feature_size):
    global first
    global sig
    global sample_rate

    sample_rate = 16000
    hop_length = 128

    sig, sample_rate = librosa.core.load(filepath, sample_rate)

    assert sample_rate == 16000,  '%s sample rate must be 16000 but sample-rate is %d' % (filepath, rate)
    assert sig.shape[0] >= 15984, '%s length must be longer than 1 second (frames: %d)' % (filepath, sig.shape[0])

    mfcc_feat = librosa.feature.mfcc(y=sig, sr=sample_rate, hop_length=hop_length, n_mfcc=feature_size, n_fft=512)
    mfcc_feat = torch.FloatTensor(mfcc_feat).transpose(0, 1)

    return mfcc_feat

def get_feature(filepath, feature_size):
    (rate, width, sig) = wavio.readwav(filepath)

    assert rate == 16000,         '%s sample rate must be 16000 but sample-rate is %d' % (filepath, rate)
    assert width == 2,            '%s sample width must be 2 but width is %d' % (filepath, width)
    assert sig.shape[0] >= 15984, '%s length must be longer than 1 second (frames: %d)' % (filepath, sig.shape[0])
    assert sig.shape[1] == 1,     '%s must be mono file, channels: %d' % (filepath, sig.shape[1])

    mfcc_feat = mfcc(sig, samplerate=rate,
                        winlen=0.02,
                        winstep=0.01,
                        numcep=feature_size,
                        nfilt=80,
                        nfft=512,
                        lowfreq=0,
                        highfreq=rate/2,
                        preemph=0.97,
                        ceplifter=feature_size,
                        appendEnergy=False)
    mfcc_feat = torch.FloatTensor(mfcc_feat)
    return mfcc_feat

def get_script(filepath, bos_id, eos_id):
    key = filepath.split('/')[-1].split('.')[0]
    script = target_dict[key]
    tokens = script.split(' ')
    result = list()
    result.append(bos_id)
    for i in range(len(tokens)):
        if len(tokens[i]) > 0:
            result.append(int(tokens[i]))
    result.append(eos_id)
    return result

class BaseDataset(Dataset):
    def __init__(self, wav_paths, script_paths, feature_size, bos_id=1307, eos_id=1308):
        self.wav_paths = wav_paths
        self.script_paths = script_paths
        self.feature_size = feature_size
        self.bos_id, self.eos_id = bos_id, eos_id

    def __len__(self):
        return len(self.wav_paths)

    def count(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        feat = get_feature(self.wav_paths[idx], self.feature_size)
        #feat = get_feature_from_librosa(self.wav_paths[idx], self.feature_size)
        script = get_script(self.script_paths[idx], self.bos_id, self.eos_id)
        return feat, script

    def getitem(self, idx):
        feat = get_feature(self.wav_paths[idx], self.feature_size)
        #feat = get_feature_from_librosa(self.wav_paths[idx], self.feature_size)
        script = get_script(self.script_paths[idx], self.bos_id, self.eos_id)
        return feat, script

def _collate_fn(batch):
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(PAD)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    return seqs, targets, seq_lengths, target_lengths

class BaseDataLoader(threading.Thread):
    def __init__(self, dataset, queue, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.collate_fn = _collate_fn
        self.dataset = dataset
        self.queue = queue
        self.index = 0
        self.batch_size = batch_size
        self.dataset_count = dataset.count()
        self.thread_id = thread_id

    def count(self):
        return math.ceil(self.dataset_count / self.batch_size)

    def create_empty_batch(self):
        seqs = torch.zeros(0, 0, 0)
        targets = torch.zeros(0, 0).to(torch.long)
        seq_lengths = list()
        target_lengths = list()
        return seqs, targets, seq_lengths, target_lengths

    def run(self):
        info('loader %d start' % (self.thread_id))
        info('batch_size: %d' % (self.batch_size))
        info('dataset_count: %d' % (self.dataset_count))
        while True:
            items = list()

            for i in range(self.batch_size): 
                if self.index >= self.dataset_count:
                    break

                items.append(self.dataset.getitem(self.index))
                self.index += 1
                #debug('thread %d, item: %d' % (self.thread_id, len(items)))

            if len(items) == 0:
                batch = self.create_empty_batch()
                self.queue.put(batch)
                break

            random.shuffle(items)

            batch = self.collate_fn(items)
            #debug('loader %d put item (%d)' % (self.thread_id, self.queue.qsize()))

            self.queue.put(batch)
        info('loader %d stop' % (self.thread_id))

class MultiLoader():
    def __init__(self, dataset_list, queue, batch_size, worker_size):
        self.dataset_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.worker_size = worker_size
        self.loader = list()

        for i in range(self.worker_size):
            self.loader.append(BaseDataLoader(self.dataset_list[i], self.queue, self.batch_size, i))

    def start(self):
        for i in range(self.worker_size):
            self.loader[i].start()

    def join(self):
        for i in range(self.worker_size):
            self.loader[i].join()

