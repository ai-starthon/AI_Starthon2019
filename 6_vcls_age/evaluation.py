# this is for reference only. you don't need to use this file.
import argparse
import numpy as np
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

def evaluate(y_true, y_pred):
    """
    Args:
      y_true (numpy array): ground truth class labels
      y_pred (numpy array): predicted class labels
    Returns:
      score (numpy float): F1 score.
    """
    score = f1_score(y_true=y_true, y_pred=y_pred, average='macro') # F1 score
    return score.item()

def evaluation_metrics(prediction_file, groundtruth_file):
    """
      Args:
        prediction_file (str): path to the file that stores the predicted labels
        groundtruth_file (str): path to the file that stores the ground truth labels
      Returns:
        acc: float top-1 accuracy.
    """
    y_pred = np.loadtxt(prediction_file)
    
    # process emotion labels
    """
    emotion2idx = {"happiness": 0, "sadness": 1, "anger": 2, "surprise": 3, 
                   "afraid": 4, "contempt": 5, "disgust": 6, "neutral": 7}

    with open(groundtruth_file, 'r') as f:
        lines = f.readlines()

    y_true = []
    for line in lines[1:]:
        emotion = line.split(',')[0] # emotion
        y_true += [emotion2idx[emotion]]
    y_true = np.array(y_true)
    """
    
    # process age labels
    age2idx = {"10\'s": 0, "20\'s": 1, "30\'s": 2, "40\'s": 3, "50\'s": 4, "60\'s": 5}
    with open(groundtruth_file, 'r') as f:
        lines = f.readlines()
    y_true = []
    for line in lines[1:]:
        age = line.split(',')[1][:-1] # emotion
        y_true += [age2idx[age]]
    y_true = np.array(y_true)
    return evaluate(y_true, y_pred)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction', type=str, default='pred.txt')
    config = args.parse_args()
    testset_path = '/data/6_vcls_age/test/test_label'
    try:
        print(evaluation_metrics(config.prediction, testset_path))
    except:
        print("0")