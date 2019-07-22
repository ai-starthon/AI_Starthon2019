# this is for reference only. you don't need to use this file.
import os


def feed_infer(output_file, infer_func):
    """
    output_file (str): file path to write output (Be sure to write in this location.)
    infer_func (function): user's infer function bound to 'nsml.bind()'
    """
    try: 
        import nsml
        root = os.path.join(nsml.DATASET_PATH)
    except:
        root = '../nipa_video/'

    predicted_labels = infer_func(root, phase='test')                        # [1, 2, 3, 4]
    predicted_labels = ' '.join([str(label) for label in predicted_labels])  # '1 2 3 4'
    
    with open(output_file, 'w') as f:
        f.write(predicted_labels)

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')
