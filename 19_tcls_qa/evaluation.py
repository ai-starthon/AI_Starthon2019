from __future__ import print_function
from collections import Counter
import string
import re
import argparse

'''본 스크립트는 KorQuAD v1.0 평가 스크립트를 바탕으로 작성됨.'''

def normalize_answer(s):
    def remove_(text):
        ''' 불필요한 기호 제거 '''
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub('《', " ", text)
        text = re.sub('》', " ", text)
        text = re.sub('<', " ", text)
        text = re.sub('>', " ", text)
        text = re.sub('〈', " ", text)
        text = re.sub('〉', " ", text)
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)
        return text

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    #F1 by character
    prediction_Char = []
    for tok in prediction_tokens:
        now = [a for a in tok]
        prediction_Char.extend(now)

    ground_truth_Char = []
    for tok in ground_truth_tokens:
        now = [a for a in tok]
        ground_truth_Char.extend(now)

    common = Counter(prediction_Char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_Char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def evaluate(prediction_path, label_path):

    with open(prediction_path, "r", encoding="utf-8") as in_file:
        predictions = in_file.read().split("\n")

    with open(label_path, "r", encoding="utf-8") as in_file:
        labels = in_file.read().split("\n")

    f1 = exact_match = total = 0
    for prediction, label in zip(predictions, labels):
        exact_match += exact_match_score(prediction, label)
        f1 += f1_score(prediction, label)

        total += 1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', type=str, default='prediction.txt')  # File

    config = parser.parse_args()
    label_path = "/data/19_tcls_qa/test/test_label"

    print(evaluate(config.prediction, label_path))
