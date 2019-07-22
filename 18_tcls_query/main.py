import os
from timeit import default_timer as timer
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import nsml

import data_utils
import data_local_loader

from args import get_config
from data_local_loader import get_dataloaders
from model import CharCNNScorer


TRAIN_BATCH_IDX = 0


def bind_nsml(model, optimizer, config):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(state, os.path.join(dir_name, "model"))
        print("saved")

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, "model"))
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        print("loaded")

    def infer(dataset_path):
        return _infer(model, config, dataset_path)

    nsml.bind(save=save, load=load, infer=infer)


def _infer(model, config, dataset_path):
    test_loader = data_local_loader.QuerySimDataLoader(
        dataset_path,
        "test_data",
        label_file_name=None,
        batch_size=config.batch_size,
        max_sequence_len=config.max_sequence_len,
        is_train=False,
        shuffle=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_logits = []
    for i, (uid, a_seqs, len_a_seqs, b_seqs, len_b_seqs) in enumerate(test_loader):
        a_seqs, b_seqs = a_seqs.to(device), b_seqs.to(device)
        logits = model(a_seqs, b_seqs, len_a_seqs, len_b_seqs)
        all_logits.append(torch.sigmoid(logits).data.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    return all_logits


def run_epoch(
        epoch_idx,
        data_loader,
        model,
        criterion,
        optimizer,
        device,
        log_steps,
):
    total_loss = 0
    epoch_preds = []
    epoch_targets = []
    epoch_start = timer()

    for i, (uid, a_seqs, len_a_seqs, b_seqs, len_b_seqs, labels) in enumerate(data_loader):
        a_seqs, b_seqs, labels = a_seqs.to(device), b_seqs.to(device), labels.to(device)

        logits = model(a_seqs, b_seqs, len_a_seqs, len_b_seqs)
        loss = criterion(logits, labels)

        batch_loss = loss.data.cpu().item()
        total_loss += batch_loss

        if data_loader.is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global TRAIN_BATCH_IDX

            nsml.report(
                summary=False,
                step=TRAIN_BATCH_IDX,
                scope=locals(),
                **{
                    f"train__batch_loss": batch_loss,
                })

            if i > 0  and i % log_steps == 0:
                print(f"batch {i:5} loss > {loss.item():.4}")

            TRAIN_BATCH_IDX += 1

        epoch_preds.append(torch.sigmoid(logits).data.cpu().numpy())
        epoch_targets.append(labels.int().data.cpu().numpy())

    score = roc_auc_score(
        np.concatenate(epoch_targets, axis=0),
        np.concatenate(epoch_preds, axis=0),
    )

    mode = "train" if data_loader.is_train else "valid"
    print(f"epoch {epoch_idx:02} {mode} score > {score:.4} ({int(timer() - epoch_start)}s)")

    total_loss /= len(data_loader.dataset)
    return score, total_loss


if __name__ == "__main__":
    config = get_config()

    # random seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = CharCNNScorer(
        vocab_size=len(data_utils.vocabs),
        char_embed_size=config.char_embed_size,
        filter_sizes=config.filter_sizes,
        sentence_embed_size=config.sentence_embed_size,
        dropout=config.dropout,
        activation=config.activation,
        pad_ind=data_utils.PAD_IND,
    ).to(device)
    print(str(model))

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
    )

    bind_nsml(model, optimizer, config)
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == "train":
        print("train")

        train_loader, valid_loader = get_dataloaders(config)
        criterion = nn.BCEWithLogitsLoss()

        num_batches = len(train_loader.dataset) // config.batch_size
        num_batches = num_batches + int((len(train_loader.dataset) % config.batch_size) > 0)
        print(f"number of batches per epoch: {num_batches}")

        best_epoch_idx = -1
        best_valid_score = 0
        early_stop_count = 0

        # train
        for epoch_idx in range(1, config.num_epochs + 1):

            def _run_epoch(data_loader):
                return run_epoch(
                    epoch_idx,
                    data_loader,
                    model,
                    criterion,
                    optimizer,
                    device,
                    config.log_steps
                )

            model.train()
            train_score, train_loss = _run_epoch(train_loader)

            # evaluate
            model.eval()
            with torch.no_grad():
                valid_score, valid_loss = _run_epoch(valid_loader)
                if best_valid_score < valid_score:
                    best_valid_score = valid_score
                    best_epoch_idx = epoch_idx
                    print(f"* best valid score {best_valid_score:.4} achieved at epoch {best_epoch_idx:02}")
                    early_stop_count = 0
                else:
                    early_stop_count += 1

                    if early_stop_count >= config.early_stop_threshold:
                        print("early stopping")
                        break

            nsml.report(
                summary=True,
                step=epoch_idx,
                scope=locals(),
                **{
                    "train__epoch_score": float(train_score),
                    "train__epoch_loss": float(train_loss),
                    "valid__epoch_score": float(valid_score),
                    "valid__epoch_loss": float(valid_loss),
                })

            nsml.save(str(epoch_idx))

    print(f"***** best valid score {best_valid_score:.4} achieved at epoch {best_epoch_idx:02}")
