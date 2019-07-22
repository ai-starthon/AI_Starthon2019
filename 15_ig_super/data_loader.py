from nsml import IS_ON_NSML, DATASET_PATH

import os
import shutil
import numpy as np

def feed_infer(output_file, infer_func):
    """"
    infer_func(function): inference 할 유저의 함수 (defined in nsml.bind function)
    output_file(str): inference 후 결과값을 저장할 파일의 위치 패스
     (이위치에 결과를 저장해야 evaluation.py 에 올바른 인자로 들어옵니다.)
    """
    if IS_ON_NSML:
        root_path = os.path.join(DATASET_PATH, 'test')
    else:
        root_path = '/home/data/nipa_faces_sr_tmp2/test' # local datapath

    # preds: numpy array for entire test data
    preds = infer_func(root_path)
    print('write output')    
    # save model output to output_file (filepath)    
    
    np.savez_compressed(output_file, output=preds)    
    shutil.move(output_file+'.npz',output_file)

