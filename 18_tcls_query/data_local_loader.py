
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nsml

from data_utils import prepro_text
from data_utils import text2ind
from data_utils import PAD_IND


class QuerySimDataset(Dataset):
    def __init__(
            self,
            data_dir,
            file_name,
            label_file_name=None,
            max_sequence_len=None,
    ):
        self.data_file_name = file_name
        self.label_file_name = label_file_name
        self.data_dir = data_dir
        self.data_file_path = os.path.join(data_dir, file_name)
        if label_file_name:
            self.label_file_path = os.path.join(data_dir, label_file_name)
        else:
            self.label_file_path = None

        self.max_sequence_len = max_sequence_len

        self._load_data(self.data_file_path, self.label_file_path)

    def _load_data(self, data_file_path, label_file_path):
        with open(data_file_path) as f:
            data = f.read().splitlines()
            data = [line.split("\t") for line in data]

            _, a_seqs, b_seqs = list(zip(*data))

            # texts
            self.a_seqs = []
            self.b_seqs = []
            print("preprocessing data")
            for a_seq, b_seq in zip(a_seqs, b_seqs):
                self.a_seqs.append(prepro_text(a_seq))
                self.b_seqs.append(prepro_text(b_seq))

            # sequence dictionary
            seqs = sorted(list(set(self.a_seqs + self.b_seqs)))
            self.uid2seq = {uid: seq for uid, seq in enumerate(seqs)}
            self.uid2ind = {seq: uid for uid, seq in enumerate(seqs)}

        if label_file_path:
            with open(label_file_path) as f:
                labels = f.read().splitlines()
                labels = [int(label) for label in labels]

            self.labels = labels
        else:
            self.labels = None

    def __len__(self):
        assert len(self.a_seqs) == len(self.b_seqs)
        return len(self.a_seqs)

    def __getitem__(self, uid):
        a_seq = self.a_seqs[uid]
        b_seq = self.b_seqs[uid]

        a_seqs_idx = text2ind(a_seq, max_len=self.max_sequence_len)
        b_seqs_idx = text2ind(b_seq, max_len=self.max_sequence_len)

        if self.labels:
            label = self.labels[uid]
            return torch.tensor(uid), torch.tensor(a_seqs_idx), torch.tensor(b_seqs_idx), torch.tensor(label)

        return torch.tensor(uid), torch.tensor(a_seqs_idx), torch.tensor(b_seqs_idx)


def collate_fn(inputs):
    _inputs = list(zip(*inputs))

    if len(_inputs) == 4:
        uids, a_seqs, b_seqs, labels = list(zip(*inputs))
    elif len(_inputs) == 3:
        uids, a_seqs, b_seqs = list(zip(*inputs))
        labels = None
    else:
        raise Exception("Invalid inputs")

    len_a_seqs = torch.tensor([len(a_seq) for a_seq in a_seqs])
    len_b_seqs = torch.tensor([len(b_seq) for b_seq in b_seqs])
    seqs = nn.utils.rnn.pad_sequence(a_seqs + b_seqs, batch_first=True, padding_value=PAD_IND)
    a_seqs, b_seqs = torch.split(seqs, len(inputs), dim=0)

    batch = [
        torch.stack(uids, dim=0),
        a_seqs,
        len_a_seqs,
        b_seqs,
        len_b_seqs,
    ]
    if labels:
        batch.append(torch.stack(labels, dim=0).float())

    return batch


class QuerySimDataLoader(DataLoader):
    def __init__(
            self,
            data_dir,
            file_name,
            label_file_name=None,
            batch_size=64,
            max_sequence_len=128,
            is_train=False,
            shuffle=False,
    ):
        super(QuerySimDataLoader, self).__init__(
            QuerySimDataset(
                data_dir,
                file_name,
                label_file_name,
                max_sequence_len=max_sequence_len,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

        self.is_train = is_train


def get_dataloaders(config):
    data_dir = os.path.join(nsml.DATASET_PATH, "train", "train_data")

    train_loader = QuerySimDataLoader(
        data_dir,
        config.train_file_name,
        config.train_label_file_name,
        batch_size=config.batch_size,
        max_sequence_len=config.max_sequence_len,
        is_train=True,
        shuffle=True,
    )
    valid_loader = QuerySimDataLoader(
        data_dir,
        config.valid_file_name,
        config.valid_label_file_name,
        batch_size=config.batch_size,
        max_sequence_len=config.max_sequence_len,
        is_train=False,
        shuffle=False,
    )

    return train_loader, valid_loader
