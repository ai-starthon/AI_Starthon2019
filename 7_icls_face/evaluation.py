""" evaluation.py
Replicated in the NSML leaderboard dataset, Face.
"""

import argparse
import numpy as np


def evaluate(prediction_labels, gt_labels):
    """
    Args:
      top1_reference_ids: dict(str: int)
      gt_labels: dict(str: int)
    Returns:
      acc: float top-1 accuracy.
    """
    count = 0.0
    for idx, query in enumerate(gt_labels):

        gt_label = int(gt_labels[query])
        pred_label = int(prediction_labels[query])

        if gt_label == pred_label:
            count += 1.0

    acc = count / float(len(gt_labels))
    return acc


def read_prediction_pt(file_name):
    """
      Args:
        file_name: str
      Returns:
        top1_reference_ids: dict(str: int)
    """
    with open(file_name) as f:
        lines = f.readlines()
    dictionary = dict([l.replace('\n', '').split(' ') for l in lines])
    return dictionary


def read_prediction_gt(file_name):
    """
      Args:
        file_name: str
      Returns:
        top1_reference_ids: dict(str: int)
    """
    with open(file_name) as f:
        lines = f.readlines()
    dictionary = dict([l.replace('\n', '').split(' ') for l in lines])
    return dictionary




def evaluation_metrics(prediction_file, testset_path):

    """
      Args:
        prediction_file: str
        testset_path: str
      Returns:
        acc: float top-1 accuracy.
    """
    prediction_labels = read_prediction_pt(prediction_file)
    gt_labels = read_prediction_gt(testset_path)
    return evaluate(prediction_labels, gt_labels)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction', type=str, default='pred.txt')
    config = args.parse_args()
    testset_path = '/data/7_icls_face/test/test_label'

    print(evaluation_metrics(config.prediction, testset_path))
