from nsml.constants import DATASET_PATH
#DATASET_PATH = './'

import numpy as np
import glob
import os


def test_data_loader(root_path):
    """
    Data loader for test data
    :param root_path: root path of test set.

    :return: data type to use in user's infer() function
    """
    raise ValueError("Do not call this function!")
    return


def feed_infer(output_file, infer_func):
    """
    This is a function that implements a way to write the user's inference result to the output file.
    :param output_file(str): File path to write output (Be sure to write in this location.)
           infer_func(function): The user's infer function bound to 'nsml.bind()'
    """
    
    file_path = os.path.join(DATASET_PATH, 'test', 'test_data')
    test_data = np.load(file_path)
    
    result = np.array(infer_func(test_data))
    
    print('write output')
    with open(output_file, 'wb') as f:    
        np.save(f, result)
