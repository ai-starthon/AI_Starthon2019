import json
import torch
import random
import pickle

OOV_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'


class InputInstances:
    def __init__(self, input_ids, input_length, intent, label):
        self.input_ids = input_ids
        self.input_length = input_length
        self.intent = intent
        self.label = label

    def to_tensors(self):
        input_ids = torch.LongTensor(self.input_ids)
        input_length = torch.LongTensor(self.input_length)
        label = torch.LongTensor(self.label)

        return input_ids, input_length, label


class Dataset:
    def __init__(self, vocab):
        self.instances = []
        self.vocab = vocab

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx].to_tensors()

    def load_dataset(self, file_path):
        with open(file_path) as f:
            data = json.load(f)
        return data

    def create_instances(self, file_path, max_seq_length, type='train'):
        examples = self.load_dataset(file_path)
        if type == 'train':
            random.shuffle(examples)

        for example in examples:
            text = example['utterance']

            input_ids = [self.vocab.token2idx[char] if char in self.vocab.token2idx
                         else self.vocab.token2idx[OOV_TOKEN] for char in text]

            if len(input_ids) > max_seq_length:
                input_ids = input_ids[-max_seq_length:]

            input_length = [len(input_ids)]
            intent = example['intent']
            label = [int(example['intent_label'])]

            while len(input_ids) < max_seq_length:
                input_ids.append(self.vocab.token2idx[PAD_TOKEN])

            assert len(input_ids) == max_seq_length
            self.instances.append(
                InputInstances(input_ids=input_ids,
                               input_length=input_length,
                               intent=intent,
                               label=label)
            )

    def create_vocab(self, file_path, type='train'):
        examples = self.load_dataset(file_path)

        for example in examples:
            text = example['utterance']

            if type == 'train' or type == 'val':
                self.vocab.update_vocab(text)

        with open('vocab.pickle', 'wb') as f:
            pickle.dump(self.vocab.token2idx, f, protocol=pickle.HIGHEST_PROTOCOL)


class Vocabulary:
    def __init__(self, vocab_path=None):
        self.oov_token = OOV_TOKEN
        self.padding_token = PAD_TOKEN

        if vocab_path:
            with open('vocab.pickle', 'rb') as f:
                self.token2idx = pickle.load(f)
            self.word_idx = len(self.token2idx)
        else:
            self.token2idx = {self.padding_token: 0, self.oov_token: 1}
            self.word_idx = 2

    def update_vocab(self, text):
        char_text = [char for char in text]

        for char in char_text:
            if char not in self.token2idx:
                self.token2idx[char] = self.word_idx
                self.word_idx += 1

    def index2token(self):
        return {v:k for k,v in self.token2idx.items()}

    def vocab_size(self):
        return len(self.token2idx)