
import os
import numpy as np
from nsml import DATASET_PATH


def test_data_loader():
    return None


def feed_infer(output_file, infer_func):
    result = infer_func(os.path.join(DATASET_PATH, "test"))
    np.savetxt(output_file, result)

    if os.stat(output_file).st_size == 0:
        raise AssertionError("Output result of inference is nothing")
