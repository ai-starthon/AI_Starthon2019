""" evaluation.py
Replicated in the NSML leaderboard dataset, KoreanFood.
"""

import os
import argparse

import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def l1_loss(x_hat, x_GT):
    loss = torch.abs(x_hat - x_GT).sum(dim=[1, 2, 3]).mean()
    return loss


def load_image(fname, image_size=128):
    transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor()])
    image = Image.open(fname).convert('RGB')
    image = transform(image)
    return image


def read_prediction_gt(dname, fnames):
    images = []
    for fname in fnames:
        fname = os.path.join(dname, fname)
        image = load_image(fname)
        images.append(image)
    return torch.stack(images, dim=0)


def evaluation_metrics(path_pred, path_GT):
    """
      Args:
        path_pred: str
        path_GT: str
      Returns:
        loss: float L1 loss
    """
    fin = np.load(path_pred)
    fnames = fin['fnames']
    x_hats = fin['x_hats']
    x_hats = torch.Tensor(x_hats)
    fin.close()
    if os.path.isdir(path_GT):
        x_GTs = read_prediction_gt(path_GT, fnames)
    else:
        fin = np.load(path_GT)
        x_GTs = fin['x_GTs']
        x_GTs = torch.Tensor(x_GTs)
        fin.close()
    loss = float(l1_loss(x_hats, x_GTs))
    return loss


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction', type=str, default='pred.txt')
    config = args.parse_args()
    test_label_path = '/data/14_ig5_inpaint/test/test_label'

    try:
        print(evaluation_metrics(config.prediction, test_label_path))
    except Exception:
        print('999999')
