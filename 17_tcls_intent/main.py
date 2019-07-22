import argparse
import os
import numpy as np
import torch
import nsml

from nsml import DATASET_PATH, IS_ON_NSML, GPU_NUM
from torch.utils.data import DataLoader

from data_utils import Vocabulary, Dataset
from trainer import Trainer, sort_batch, device


def inference(path, model, vocab, config, **kwargs):
    model.eval()
    test_dataset = Dataset(vocab)
    test_path = os.path.join(path, 'test_data')
    test_dataset.create_instances(test_path, config.max_seq_length, type='test')
    test_loader = DataLoader(test_dataset, batch_size=1)

    pred_results = []
    for step, batch in enumerate(test_loader):
        batch = tuple(t.to(device) for t in batch)
        batch = sort_batch(batch)
        input_ids, input_lengths, labels = batch

        outputs = model(input_ids, input_lengths)
        top_1_result = outputs['predicted_intents'][0].item()
        pred_results.append([step, top_1_result])

    return pred_results


def bind_model(model, vocab, config, **kwargs):
    def save(path, *args, **kwargs):
        # save the model with 'checkpoint' dictionary.
        checkpoint = {
            'model': model.state_dict(),
        }
        torch.save(checkpoint, os.path.join(path, 'model.pth'))

    def load(path, *args, **kwargs):
        checkpoint = torch.load(os.path.join(path, 'model.pth'))
        model.load_state_dict(checkpoint['model'])

    def infer(path, **kwargs):
        return inference(path, model, vocab, config)

    nsml.bind(save, load, infer)


def main(config, local):
    n_gpu = int(GPU_NUM)
    n_gpu = 1 if n_gpu == 0 else n_gpu
    np.random.seed(config.random_seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.random_seed)

    # Create data instances
    vocab = Vocabulary(config.vocab_path)

    if config.mode == 'train':
        # Prepare train data loader
        train_dataset, val_dataset = Dataset(vocab), Dataset(vocab)
        train_path = os.path.join(config.data_dir, 'train_data/train_data')
        val_path = os.path.join(config.data_dir, 'train_data/val_data')

        train_dataset.create_instances(train_path, config.max_seq_length, type='train')
        val_dataset.create_instances(val_path, config.max_seq_length, type='val')

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size * n_gpu, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size * n_gpu)
    else:
        train_loader, val_loader = None, None

    trainer = Trainer(config, n_gpu, vocab, train_loader, val_loader)

    if nsml.IS_ON_NSML:
        bind_model(trainer.model,
                   vocab,
                   config)

        if config.pause:
            nsml.paused(scope=local)

    if config.mode == 'train':
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Intent Classification Baseline')

    # Data
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--vocab_path', default='vocab.pickle', type=str)
    parser.add_argument('--n_label', type=int, default=2253)

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--max_seq_length', type=int, default=256)

    # Model Hyper-parameters
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)

    # NSML
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--iteration', type=str, default='0')

    # Misc.
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=100)

    config = parser.parse_args()
    config.cuda = not config.no_cuda and torch.cuda.is_available()

    if IS_ON_NSML:
        config.data_dir = os.path.join(DATASET_PATH, 'train')
    else:
        config.data_dir = os.path.join(config.data_dir, 'train')

    main(config, local=locals())