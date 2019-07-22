import numpy as np
import argparse
from sklearn.metrics import accuracy_score 

# evaluation.py
def read_prediction(prediction_file):
    pred_array = np.load(prediction_file)
    return pred_array


def read_ground_truth(ground_truth_file):
    gt_array = np.load(ground_truth_file)
    return gt_array


# recall
def evaluate(prediction, ground_truth):
    custom_acc = np.array(ground_truth == prediction).sum() / len(ground_truth)
    rst = accuracy_score(ground_truth, prediction)
    return rst


# user-defined function for evaluation metrics
def evaluation_metrics(prediction_file: str, ground_truth_file: str):
    prediction = read_prediction(prediction_file)
    ground_truth = read_ground_truth(ground_truth_file)
    return evaluate(prediction, ground_truth)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # --prediction is set by file's name that contains the result of inference. (nsml internally sets)
    # prediction file requires type casting because '\n' character can be contained.
    args.add_argument('--prediction', type=str, default='pred.txt')
    config = args.parse_args()
    
    # When push dataset, if you push with leaderboard option, automatically test_label is existed on test/test_label path, set to proper path.
    # You should type dataset's name that wants to upload on [dataset] section.
    test_label_path = '/data/3_cls_crane2/test/test_label'
    
    # print the evaluation result
    # evaluation prints only int or float value.
    try:
        print(evaluation_metrics(config.prediction, test_label_path))
    except:
        # 에러로인한 0점
        print("0")