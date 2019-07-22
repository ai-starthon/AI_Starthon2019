import numpy as np
import os
from nsml import DATASET_PATH

def test_data_loader(root_path):
    """
    Data loader for test data

    :param root_path: path to the test dataset. root_path would be enough.
    :return: data type used in infer() function

    location of the test data is root_path/test/*
    """
    return np.load(os.path.join(root_path, 'test', 'test_data'), dtype=np.float32)
	
def feed_infer(output_file, infer_func):
    """
    This is a function that implements a way to write the user's inference result to the output file.
    :param output_file(str): File path to write output (Be sure to write in this location.)
           infer_func(function): The user's infer function bound to 'nsml.bind()'
    """
    result = infer_func(os.path.join(DATASET_PATH, 'test'))
    result = [str(pred[1]) for pred in result]
    print('write output')
    with open(output_file, 'w') as file_writer:
        file_writer.write("\n".join(result))

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')