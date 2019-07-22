from nsml import DATASET_PATH
import os


def test_data_loader(path):
    return path


def feed_infer(output_file, infer_func):
    data_path = os.path.join(DATASET_PATH, 'test')
    result = infer_func(data_path)
    result_output = [str(pred[1]) for pred in result]

    print('write output')
    with open(output_file, 'w') as output_f:
        output_f.write("\n".join(result_output))