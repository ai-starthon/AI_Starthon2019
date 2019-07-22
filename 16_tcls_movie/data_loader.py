import os
import csv


def test_data_loader(dataset_path, train=False, percentage_to_load=0.1, batch_size=200, ratio_of_validation=0.1, shuffle=True):
    if train:
        return get_dataloaders(dataset_path=dataset_path, type_of_data='train',
                               percentage_to_load=percentage_to_load, batch_size=batch_size,
                               ratio_of_validation=ratio_of_validation, shuffle=shuffle)
    else:
        # data_dict = {'data': read_test_file(dataset_path=dataset_path)}
        data_dict = read_test_file(dataset_path=dataset_path)
        return data_dict


def read_test_file(dataset_path):
    data_path = os.path.join(dataset_path, 'test', 'test_data')  # Test data only
    loaded_data = []
    with open(data_path, 'rt', encoding='utf-8') as f:  # maximum buffer size == 50 MB
        lines = f.readlines()
    return lines
