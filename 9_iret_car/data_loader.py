import os
from nsml import DATASET_PATH


def feed_infer(output_file, infer_func):
    """"
    infer_func(function): inference 할 유저의 함수
    output_file(str): inference 후 결과값을 저장할 파일의 위치 패스
     (이위치에 결과를 저장해야 evaluation.py 에 올바른 인자로 들어옵니다.)
    """
    root_path = os.path.join(DATASET_PATH, 'test')
    top1_reference_ids = infer_func(root_path)
    top1_reference_ids_str = [' '.join(l) for l in top1_reference_ids]
    print('write output')
    with open(output_file, 'w') as file_writer:
        file_writer.write("\n".join(top1_reference_ids_str))

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')


def test_data_loader(root_path):
    return root_path
