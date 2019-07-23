from nsml import DATASET_PATH
import nsml

import numpy as np
import glob
import argparse



def bind_model(model):
    def save(dir_name):
        np.save(dir_name + '/params.npy', np.array([model.all_mean, model.all_std]))

    def load(dir_name):
        params = np.load(dir_name + '/params.npy')
        model.all_mean = params[0]
        model.all_std = params[1]

    def infer(data):
        return model.forward(data)

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)

    
class GaussianModel():
    
    def __init__(self):
        self.all_mean = 0
        self.all_std = 1
        
    def train(self, train_file_list):
        data_list = [np.load(file).flatten() for file in train_file_list]
        all_data = np.concatenate(data_list)
        self.all_mean = all_data.mean()
        self.all_std = all_data.std()
    
    def forward(self, data):
        # data : numpy matrix of shape (T * F ) where T is time and F is frequency
        # F = 500, T is not fixed
        m = data.mean()
        s = data.std()
        
        p = self.kl_divergence(m, s, self.all_mean, self.all_std)
        if p > 1:
            p = 1
        return p
        
    def kl_divergence(self, m1, s1, m2, s2):
        return (m1 - m2)**2 + np.log(s2/s1) + (s1**2 + (m1 - m2)**2)/(2*(s2**2)) - 0.5
    

    

if __name__ == '__main__':
    
    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    
    config = args.parse_args()
    
    
    # Bind model
    model = GaussianModel()
    bind_model(model)
    
    
    # DONOTCHANGE: They are reserved for nsml
    # Warning: Do not load data before the following code!
    if config.pause:
        nsml.paused(scope=locals())
    
    
    # Load data
    train_dataset_path = DATASET_PATH + '/train/train_data'
    train_data_files = sorted(glob.glob(train_dataset_path + '/*.npy')) 
    
    train_label_file = DATASET_PATH + '/train/train_label' # All labels are zero in train data.
    train_label = np.load(train_label_file) # numpy array of labels. They are all zeros!!
    
    
    # Train
    model.train(train_data_files)
    print(model.all_mean, model.all_std)
    
    
    # Save
    epoch = 1
    nsml.save(epoch) # If you are using neural networks, you may want to use epoch as checkpoints
    
    # Load test (Check if load method works well)
    nsml.load(epoch)
    
    # Infer test
    for file in train_data_files[:10]:
        data = np.load(file)
        print(model.forward(data))
        
        