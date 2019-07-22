import numpy as np
import argparse
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')

# evaluation.py
def read_prediction(prediction_file):
   # NEED TO IMPLEMENT #1
   # function that loads prediction
   pred_array = np.loadtxt(prediction_file, dtype=np.int16)
   return pred_array


def read_ground_truth(ground_truth_file):
   # NEED TO IMPLEMENT #2
   # function that loads test_data
   gt_array = np.loadtxt(ground_truth_file, dtype=np.int16)
   return gt_array


# f1
def evaluate(prediction, ground_truth):
   # NEET TO IMPLEMENT #3
   # Function that actually evaluates
   return f1_score(ground_truth, prediction, average='macro')

# user-defined function for evaluation metrics
def evaluation_metrics(prediction_file: str, ground_truth_file: str):
   # read prediction and ground truth from file
   prediction = read_prediction(prediction_file)  # NOTE: prediction is text
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
   test_label_path = '/data/16_tcls_movie/test/test_label'
   # print the evaluation result
   # evaluation prints only int or float value.

   try:
      print(evaluation_metrics(config.prediction, test_label_path))
   except:
      print(0.0)
