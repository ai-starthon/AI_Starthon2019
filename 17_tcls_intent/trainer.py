import nsml
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import LSTMClassifier
from evaluation import ic_metric

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sort_batch(batch):
    input_ids, input_lengths, labels = batch
    input_lengths, sorted_idx = input_lengths.sort(0, descending=True)
    sorted_idx = sorted_idx.squeeze(1)

    input_ids = input_ids[sorted_idx]
    input_lengths = input_lengths.squeeze(1)
    labels = labels[sorted_idx].squeeze(1)

    return input_ids, input_lengths, labels


class Trainer:
    def __init__(self, config, n_gpu, vocab, train_loader=None, val_loader=None):
        self.config = config
        self.vocab = vocab
        self.n_gpu = n_gpu
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Build model
        vocab_size = self.vocab.vocab_size()

        self.model = LSTMClassifier(self.config, vocab_size, self.config.n_label)
        self.model.to(device)

        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)

        # Build optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

        # Build criterion
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        best_f1 = 0.0
        best_acc = 0.0
        global_step = 0
        batch_f1 = []
        batch_acc = []
        for epoch in range(self.config.num_epoch):
            batch_loss = []
            for step, batch in enumerate(self.train_loader):
                self.model.train()
                batch = tuple(t.to(device) for t in batch)
                batch = sort_batch(batch)
                input_ids, input_lengths, labels = batch

                outputs = self.model(input_ids, input_lengths)
                loss = self.criterion(outputs['logits'].view(-1, self.config.n_label), labels.view(-1))

                f1, acc = ic_metric(labels.cpu(), outputs['predicted_intents'].cpu())

                if self.n_gpu > 1:
                    loss = loss.mean()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                global_step += 1
                batch_loss.append(loss.float().item())
                batch_f1.append(f1)
                batch_acc.append(acc)

                if (global_step == 1) or (global_step % self.config.log_interval == 0):
                    mean_loss = np.mean(batch_loss)
                    mean_f1 = np.mean(batch_f1)
                    mean_acc = np.mean(batch_acc)
                    batch_loss = []
                    nsml.report(summary=True, scope=locals(), epoch=epoch, train_loss=mean_loss, step=global_step)

                if (global_step > 0) and (global_step % self.config.val_interval == 0):
                    val_loss, val_f1, val_acc = self.evaluation()
                    nsml.report(summary=True, scope=locals(), epoch=epoch, val_loss=val_loss,
                                val_f1=val_f1, val_acc=val_acc, step=global_step)

                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        best_acc = val_acc
                        nsml.save(global_step)

    def evaluation(self):
        self.model.eval()
        total_loss = []
        preds = []
        targets = []
        with torch.no_grad():
            for step, batch in enumerate(self.val_loader):
                batch = tuple(t.to(device) for t in batch)
                batch = sort_batch(batch)
                input_ids, input_lengths, labels = batch

                outputs = self.model(input_ids, input_lengths)
                loss = self.criterion(outputs['logits'].view(-1, self.config.n_label), labels.view(-1))

                pred = outputs['predicted_intents'].squeeze(-1).cpu().numpy().tolist()
                target = labels.cpu().numpy().tolist()

                preds.extend(pred)
                targets.extend(target)
                total_loss.append(loss.float().item())

        mean_loss = np.mean(total_loss)
        mean_f1, mean_acc = ic_metric(targets, preds)
        return mean_loss, mean_f1, mean_acc
