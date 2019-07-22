import argparse
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from warnings import filterwarnings
filterwarnings('ignore')


def ic_metric(targets, predicts):
    targets = np.asarray(targets)
    predicts = np.asarray(predicts)

    macro_f1_value = f1_score(targets, predicts, average='macro')
    acc_value = accuracy_score(targets, predicts)
    return macro_f1_value, acc_value


def evaluate(pred_file, gt_file):
    preds = read_label_file(pred_file)
    gts = read_label_file(gt_file)

    preds = [int(label) for label in preds]
    gts = [int(label) for label in gts]

    f1, accuracy = ic_metric(gts, preds)
    return f1


def read_label_file(file_name):
    with open(file_name, 'r') as f:
        label = f.read().split('\n')

    return label


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction', type=str, default='prediction.txt')
    config = args.parse_args()
    label_path = '/data/17_tcls_intent/test/test_label'

    try:
        print(evaluate(config.prediction, label_path))
    except:
        # 에러로인한 0점
        print("0")