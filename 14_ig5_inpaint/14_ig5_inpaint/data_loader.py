""" data_loader.py
Replicated in the NSML leaderboard dataset, KoreanFood.
"""

try:
    import nsml
    dir_data_root = nsml.DATASET_PATH
    use_nsml = True
except ImportError:
    dir_data_root = '/home/data/nipa_korean_faces'
    print('dir_data_root:', dir_data_root)
    use_nsml = False

import os
import shutil
import numpy as np


def feed_infer(output_file, infer_func):
    """"
    infer_func(function): inference 할 유저의 함수
    output_file(str): inference 후 결과값을 저장할 파일의 위치 패스
     (이위치에 결과를 저장해야 evaluation.py 에 올바른 인자로 들어옵니다.)
    """
    if use_nsml:
        root_path = os.path.join(dir_data_root, 'test')
    else:
        root_path = '/home/data/14_ig5_inpaint/test'
    fnames, x_hats = infer_func(root_path)
    np.savez_compressed(output_file, fnames=fnames, x_hats=x_hats)
    shutil.move(output_file + '.npz', output_file)


def test_data_loader(root_path):
    return root_path
