""" evaluation.py
Replicated in the NSML leaderboard dataset, KoreanFood.
"""

import argparse
import torch

def compute_iou(box_a, box_b):
    # make [x1,y1,x2,y2] from [x,y,w,h]
    box_a[:, 2:] += box_a[:, :2]
    box_b[:, 2:] += box_b[:, :2]

    max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
    min_xy = torch.max(box_a[:, :2], box_b[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    intersection = inter[:, 0] * inter[:, 1]

    area_a = (box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])
    area_b = (box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])
    union = area_a + area_b - intersection

    return intersection / union


def read_prediction_gt(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    boxes = [l.replace('\n', '').split(',') for l in lines]
    boxes_t = []
    for box in boxes:
        boxes_t.append(torch.Tensor([float(b) for b in box]).unsqueeze(0))
    boxes_t = torch.cat(boxes_t,0)
    return boxes_t

def evaluation_metrics(prediction_file, testset_path):
    predictions = read_prediction_gt(prediction_file)
    gt_labels = read_prediction_gt(testset_path)
    total_iou = compute_iou(predictions, gt_labels)
    mean_iou = total_iou.mean().item()
    return mean_iou

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction', type=str, default='pred.txt')
    config = args.parse_args()
    testset_path = '/data/10_idet5_food/test/test_label'

    try:
        print(evaluation_metrics(config.prediction, testset_path))
    except:
        print('0')
