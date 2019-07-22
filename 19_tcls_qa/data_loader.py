
import os

from nsml.constants import DATASET_PATH


def test_data_loader(root_path):
    """
    Data loader for test data
    :param root_path: root path of test set.

    :return: data type to use in user's infer() function
    """
    test_path = os.path.join(root_path, 'test', 'test_data')
    with open(test_path, "r", encoding="utf-8") as in_file:
        test_data = in_file.read().split("\n")
    return test_data


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
