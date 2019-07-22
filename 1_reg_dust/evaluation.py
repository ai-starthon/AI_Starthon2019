import argparse
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error
from warnings import filterwarnings
filterwarnings('ignore')

def weighted_mse(targets, predicts):    
    #sum_val = 0.0
    pred = np.asarray(predicts)
    gt = np.asarray(targets)
    w_mse = 0.3*mean_squared_error(gt[:][0], pred[:][0]) + 0.7*mean_squared_error(gt[:][1], pred[:][1])
    #for idx, v in enumerate(targets):
    #    #try:
    #        sum_val += (0.3*(v[0]-predicts[idx][0])*(v[0]-predicts[idx][0])+0.7*(v[1]-predicts[idx][1])*(v[1]-predicts[idx][1]))
    #    #except IndexError:
    #    #    print(v, predicts[idx])
    #w_mse = sum_val/len(targets)
    return w_mse

def ic_metric(targets, predicts):
    #targets = np.asarray(targets)
    #targets = [val.]
    predicts = np.asarray(predicts)

	#macro_f1_value = f1_score(targets, predicts, average='macro')
	#acc_value = accuracy_score(targets, predicts)
    #mse_score = mean_squared_error(targets, predicts)
    mse_score = weighted_mse(targets, predicts)
    return mse_score
	#return macro_f1_value, acc_value


def evaluate(pred_file, gt_file):
    preds = read_label_file(pred_file)
    #gts = read_label_file(gt_file)
    gts = np.load(gt_file)    
    ppreds = []
    for label in preds:     
        #try:
            labels = label.strip().replace("[","").replace("]","").split(',')
            if len(labels) > 2:
                labels = [l for l in labels if l != " "]                                       
            ppreds.append((np.float32(labels[0].strip()), np.float32(labels[-1].strip())))
        
        #except ValueError:            
        #    print (labels)
        #    exit()            
   # preds = [(np.float32(label[0]), np.float32(label[1])) for label in preds]
    #gts = [np.float32(label) for label in gts]

    mse_score = ic_metric(gts, ppreds)
	#f1, accuracy = ic_metric(gts, preds)
    return mse_score


def read_label_file(file_name):
    #label = np.load(file_name)
    with open(file_name, 'r') as f:
        label = f.read().split('\n')

    return label


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction', type=str, default='prediction.txt')
    config = args.parse_args()
    label_path = '/data/1_reg_dust/test/test_label'

    print(evaluate(config.prediction, label_path))