
import argparse
import numpy as np

from sklearn.metrics import roc_auc_score


def evaluate(pred_path, target_path):
    preds = np.loadtxt(pred_path)
    targets = np.loadtxt(target_path)

    return roc_auc_score(targets, preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=str, default="prediction.txt")
    config = parser.parse_args()

    test_label_file_path = "/data/18_tcls_query/test/test_label"

    try:
        print(evaluate(config.prediction, test_label_file_path))
    except:
        print("0")
