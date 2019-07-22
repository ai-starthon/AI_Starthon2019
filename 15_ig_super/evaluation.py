""" evaluation.py
Replicated in the NSML leaderboard dataset, KoreanFood.
"""

import argparse
import os
import numpy as np
import time, math, glob

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[1:]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def evaluate(preds, gt_labels):
    """
    Args:
      preds: numpy array (float)
      gt_labels: numpy array (float)
    Returns:
      avr_psnr_predicted: float average peak signal-to-noise ratio of the test data.
    """
    avg_psnr_predicted = 0.0
    for query in range(len(gt_labels)):
        gt_label = gt_labels[query]
        pred = preds[query]
        psnr_predicted = PSNR(gt_label, pred)
        avg_psnr_predicted += psnr_predicted

    avg_psnr_predicted = avg_psnr_predicted / float(len(gt_labels))
    return avg_psnr_predicted


def read_prediction_gt(file_path):
    """
      Args:
        file_path: str
      Returns:
        image_list: list(str)
    """
    loaded = np.load(file_path) # npz object
    loaded = loaded[loaded.files[0]] # now numpy array
    return loaded


def evaluation_metrics(prediction_file_path, testset_path):
    """
      Args:
        prediction_file_path: str
        testset_path: str
      Returns:
        avr_psnr_predicted: float average peak signal-to-noise ratio of the test data.
    """
    preds = read_prediction_gt(prediction_file_path)    
    gt_labels = read_prediction_gt(testset_path)
    return evaluate(preds, gt_labels)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction', type=str, default='preds.npz') # output_file that is returned from 'feed_infer' of data_loader.py
    config = args.parse_args()    
    test_label_path  = '/data/15_ig_super/test/test_label' # do not change
    
    try:
        print(evaluation_metrics(config.prediction, test_label_path))
    except:
        print(0.0)