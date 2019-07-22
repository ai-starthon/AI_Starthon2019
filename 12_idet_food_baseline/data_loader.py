""" data_loader.py
Replicated in the NSML leaderboard dataset, KoreanFoodLocalize.
"""

from nsml import IS_ON_NSML, DATASET_PATH

import os

def feed_infer(output_file, infer_func):
    """"
    infer_func(function): inference 할 유저의 함수
    output_file(str): inference 후 결과값을 저장할 파일의 위치 패스
     (이위치에 결과를 저장해야 evaluation.py 에 올바른 인자로 들어옵니다.)
    """
    if IS_ON_NSML:
        root_path = os.path.join(DATASET_PATH, 'test')
    else:
        root_path = '/home/data/NIPAKoreanFoodLocalize/test'
    results = infer_func(root_path)
    results_str = [','.join(str(v) for v in l) for l in list(results)]
    print('write output')
    with open(output_file, 'w') as file_writer:
        file_writer.write("\n".join(results_str))

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')


def test_data_loader(root_path):
    return root_path
