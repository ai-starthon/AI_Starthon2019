import torch
from torch.autograd import Variable
import numpy as np

from nsml import DATASET_PATH
import nsml
import h5py

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import keras.models
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error

#import glob
import os
import argparse

FEATURE_DIM = 14 #지역(0~9), 연(2016~2019), 월, 일, t-5 ~ t-1의 미세 & 초미세
OUTPUT_DIM = 2 # t-time의 (미세, 초미세)

def bind_model(model):
    def save(path, *args, **kwargs):
        # save the model with 'checkpoint' dictionary.
        
        model.model.save(os.path.join(path, 'model'))    

    def load(path, *args, **kwargs):
        model = keras.models.load_model(os.path.join(path, 'model'))
        return model

    def infer(path, **kwargs):
        return inference(path, model, config)

    nsml.bind(save, load, infer)


def inference(path, model, config, **kwargs):

    #model.eval()

    test_path = DATASET_PATH+'/test/test_data'
    data = convData(np.load(test_path))
    mean10_val = np.mean(data[:,4::2])
    pred_val = model.model.predict(data/mean10_val).tolist()
    pred_results = [[step, val] for step, val in enumerate(pred_val)]
    return pred_results
 
def convData(data_arr):
    v = np.zeros(FEATURE_DIM, dtype=np.float32)
    v[1] = 2016
    new_d = np.asarray([d - v for d in data_arr])
    return new_d

class Model():
    def __init__(self, lr):
        self.model = Sequential([Dense(32, input_shape=(FEATURE_DIM,)), Activation('relu'), Dense(OUTPUT_DIM), Activation('relu')])
        rms_pr = RMSprop(lr=lr)
        self.model.compile(optimizer=rms_pr, loss='mse')
        return    
        
    

if __name__ == '__main__':
    
    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    
    config = args.parse_args()
    
    EP, LR = 10, 0.001    
    # Bind model
    model = Model(LR)
    bind_model(model)
    
       
    # DONOTCHANGE: They are reserved for nsml
    # Warning: Do not load data before the following code!
    if config.pause:
        nsml.paused(scope=locals())
    
    if config.mode == "train":
    
        train_dataset_path = DATASET_PATH + '/train/train_data'
        #train_data_files = sorted(glob.glob(train_dataset_path + '/*.npy')) 
        tr_X = convData(np.load(train_dataset_path))
        mean10_val = np.mean(tr_X[:,4::2])
        train_label_file = DATASET_PATH + '/train/train_label'
        tr_Y = np.load(train_label_file)
    
        model.model.fit(tr_X/mean10_val, tr_Y, epochs=EP, batch_size=1024)
                   
    
    #tr_X = np.load(DATASET_PATH + '/train/train_data.npy')    
    #tr_Y = np.load(DATASET_PATH + '/train/train_label.npy') # numpy array of labels. They are all zeros!!


    # Train
    
        nsml.save(EP)    
        
    # Save
    #epoch = 1
    #nsml.save(epoch) # If you are using neural networks, you may want to use epoch as checkpoints
    
    # Load test (Check if load method works well)
    #nsml.load(epoch)
    
    # Infer test
    #for file in train_data_files[:10]:
    #    data = np.load(file)
    #    print(model.forward(data))    