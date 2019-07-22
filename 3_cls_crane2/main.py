from nsml import DATASET_PATH
import nsml

import os
import numpy as np
import glob
import argparse


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler

def to_np(t):
    return t.cpu().detach().numpy()

def bind_model(model):
    def save(dir_name, **kwargs):
        save_state_path = os.path.join(dir_name, 'state_dict.pkl')
        state = {
                    'model': model.state_dict(),
                }
        torch.save(state, save_state_path)

    def load(dir_name):
        save_state_path = os.path.join(dir_name, 'state_dict.pkl')
        state = torch.load(save_state_path, map_location={'cuda:0': 'cpu'})
        model = state['model']
        
    def infer(data_x):
        x = torch.tensor(data_x, device=DEVICE, dtype=torch.float32)
        logit = model(x)
        prob = F.softmax(logit, dim=1)
        pred = torch.argmax(prob, dim=1)
        pred_y = to_np(pred)
        return pred_y

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)

    
class SimpleMLP(nn.Module):
    def __init__(self, config):
        super(SimpleMLP, self).__init__()
        self.config = config
        self.linear1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.linear2 = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x):
        h = self.linear1(x)
        out = self.linear2(h)
        return out
    
class CraneDatasetB(Dataset):
    def __init__(self):
        super().__init__()
        # Load data
        train_dataset_path = DATASET_PATH + '/train/train_data'
        train_data_files = sorted(glob.glob(train_dataset_path + '/*.npy')) 

        self.train_data = []
        for name in train_data_files:
            self.train_data.append(np.load(name))
        
        self.train_label = []
        for name in train_data_files:
            label_str = name.split("/")[-1].split("_")[0]
            if label_str == "normal":
                self.train_label.append(0)
            elif label_str == "A":
                self.train_label.append(1)
            elif label_str == "B":
                self.train_label.append(2)
            elif label_str == "C":
                self.train_label.append(3)
            elif label_str == "D":
                self.train_label.append(4)
            elif label_str == "E":
                self.train_label.append(5)
            elif label_str == "F":
                self.train_label.append(6)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        return self.train_data[index], self.train_label[index]
    
def custom_collate_fn(batch):
    x_list = [items[0] for items in batch]
    x = np.concatenate(x_list, axis=0)
    x = torch.tensor(x, device=DEVICE, dtype=torch.float32) 

    y_list = []
    for items in batch:
        y_list.extend([items[1]]*items[0].shape[0])
    
    y = np.array(y_list).flatten()
    y = torch.tensor(y, device=DEVICE, dtype=torch.long)
    return x, y

DEVICE = 'cpu' #'cuda:0' or 'cpu'
if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    
    # Experiments args
    args.add_argument('--input_dim', type=int, default=6150)
    args.add_argument('--hidden_dim', type=int, default=1024)
    args.add_argument('--output_dim', type=int, default=7)
    args.add_argument('--max_epoch', type=int, default=100)
    args.add_argument('--batch_size', type=int, default=8) # number of files(sequences)
    args.add_argument('--initial_lr', type=float, default=0.00003)
    config = args.parse_args()
    
    # Bind model
    model = SimpleMLP(config)
    bind_model(model)
    
    # DONOTCHANGE: They are reserved for nsml
    # Warning: Do not load data before the following code!
    if config.pause:
        nsml.paused(scope=locals())
    
    train_dataset = CraneDatasetB()
    train_sampler = RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=train_sampler, collate_fn=custom_collate_fn, 
                                               batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # Train
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.initial_lr)
    
    for e_i in range(config.max_epoch):
        sum_loss = 0
        accurate_pred = 0
        num_of_instance = 0
        for i, x_y_pair in enumerate(train_loader):
            x, y = x_y_pair
            logit = model(x)
            loss = loss_fn(logit, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            prob = F.softmax(logit, dim=1)
            pred = torch.argmax(prob, dim=1)
            
            accurate_pred += to_np((pred == y).sum())
            num_of_instance += y.shape[0]
            sum_loss += float(to_np(loss))
            b_acc = to_np(  torch.mean(torch.eq(pred, y).float())  )
            print("this batch acc:", b_acc, "total correct answer:", accurate_pred, "total instances:", num_of_instance)
            
        accuracy = (accurate_pred / num_of_instance)
        nsml.report(**{"summary":True, "step":e_i, "scope":locals(), "train__loss:":float(sum_loss), "train__accuracy:":float(accuracy)})
        print(e_i,"'th epoch acc:", accuracy, "correct answer:", accurate_pred, "total instances:", num_of_instance, "loss:", sum_loss)
        nsml.save(e_i)  

    
