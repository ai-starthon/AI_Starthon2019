import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
from PIL import Image
import numpy as np
from utils import alphanumeric_sort
from utils import pad_sequences
import pandas as pd


class MultimodalDataset(Dataset):
    def __init__(self, root, task='emotion', phase='train'):
        """
        Inputs:
        - root: path where the data file stored.
        - task: emotion or age. Required.
        - phase: train, test, or test_submit. Required.
        """
        self.root = root
        self.task = task
        self.phase = phase
        if self.task == 'emotion':
            self.label_list = ["happiness", "sadness", "anger", "surprise", "afraid", "contempt", "disgust", "neutral"]
        else:
            self.label_list = ["10\'s", "20\'s", "30\'s", "40\'s", "50\'s", "60\'s"]
    
        #print('root: ', root)
        #print('task: ', task)
        #print('phase: ', phase)
        #print('path: ', os.path.join(root, self.phase, phase.split('_')[0]+'_data'))
        
        self.shot_list = alphanumeric_sort(os.listdir(os.path.join(root, self.phase, phase.split('_')[0]+'_data')))
        print(len(self.shot_list))
        
        if self.phase == 'train':
            label_tot = pd.read_csv(os.path.join(root, self.phase, phase.split('_')[0]+'_label'))
            self.labels = dict()
            self.labels['emotion'] = list(label_tot.emotion)
            self.labels['age'] = list(label_tot.age)

    def __len__(self):
        return len(self.shot_list)

    def __getitem__(self, idx):
        if self.phase == 'train':
            shot_path = os.path.join(self.root, self.phase, self.phase.split('_')[0]+'_data', self.shot_list[idx])
            img_name_list = os.listdir(shot_path)
            for idx, img_name in enumerate(img_name_list):
                if 'txt' not in img_name:
                    img = Image.open(os.path.join(shot_path, img_name)).resize((320, 180))
                    img = ToTensor()(img)
                    if idx == 0:
                        imgs = img
                    else:
                        imgs = torch.cat((imgs, img), 0)
            seq_len = len(img_name_list)

            label = self.labels[self.task][idx]
            label = self.label_list.index(label)
            return (imgs, seq_len, label)
        else:
            shot_path = os.path.join(self.root, self.phase, self.phase.split('_')[0]+'_data', self.shot_list[idx])
            img_name_list = os.listdir(shot_path)
            for idx, img_name in enumerate(img_name_list):
                if 'txt' not in img_name:
                    img = Image.open(os.path.join(shot_path, img_name)).resize((320, 180))
                    img = ToTensor()(img)
                    if idx == 0:
                        imgs = img
                    else:
                        imgs = torch.cat((imgs, img), 0)
            seq_len = len(img_name_list)
            
            # pseudo label
            label = 0
            return (imgs, seq_len, label)

def collate_fn(batch):
    """
    batch = [dataset[i] for i in N]
    """
    MAX_LEN = 15
    size = len(batch[0])
    assert (size == 3)
    imgs, lens, y = zip(*batch)
    lens = np.array(lens)
    imgs = torch.tensor(pad_sequences(imgs, maxlen=MAX_LEN), dtype=torch.float32)
    return imgs, lens, torch.tensor(y, dtype=torch.long)


def get_loader(root='./nipa_video', task='age', phase='train', batch_size=16):
    dataset = MultimodalDataset(root=root, task=task, phase=phase)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=(phase=='train'), num_workers=8)
    return data_loader

"""
import pdb
for imgs, seq_len, label in loader_train:
    print('here')
    pdb.set_trace()
"""