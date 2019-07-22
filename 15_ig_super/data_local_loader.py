import os
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms

class TrainDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dname_lr = os.path.join(root_dir, 'lr')
        self.dname_hr = os.path.join(root_dir, 'hr')
        self.samples = sorted(os.listdir(self.dname_hr))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (lr, hr)
        """
        fname = self.samples[index]
        fname_lr = os.path.join(self.dname_lr, fname)
        fname_hr = os.path.join(self.dname_hr, fname)
        lr = Image.open(fname_lr).convert('YCbCr')
        lr_y, _, _ = lr.split()   
        hr = Image.open(fname_hr).convert('YCbCr')
        hr_y, _, _ = hr.split()   
        if self.transform is not None:
            lr_y = self.transform(lr_y)
            hr_y = self.transform(hr_y)

        return lr_y, hr_y

    def __len__(self):
        return len(self.samples)


class TestDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir # '/home/data/nipa_faces_sr_tmp/test' if local
        self.transform = transform
        self.dname_lr = os.path.join(root_dir, 'test_data')        
        self.samples = sorted(os.listdir(self.dname_lr))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (lr_test)
        """
        fname = self.samples[index]
        fname_lr = os.path.join(self.dname_lr, fname)        
        lr_test = Image.open(fname_lr).convert('YCbCr')      
        lr_test_y, _, _ = lr_test.split()   
        if self.transform is not None:
            lr_test_y = self.transform(lr_test_y)           

        return lr_test_y

    def __len__(self):
        return len(self.samples)

    
def test_loader(root, batch_size=32, num_workers=1):    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = TestDataset(root, transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)
    return dataloader

def data_loader_with_split(root, train_split=0.9, batch_size=32, num_workers=1):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = TrainDataset(root, transform=transform)
    train_split_size = int(len(dataset) * train_split)
    val_split_size = len(dataset) - train_split_size
    
    if train_split_size > 0:
        train_set, valid_set = data.random_split(dataset, [train_split_size, val_split_size])
        tr_loader = data.DataLoader(dataset=train_set,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)
    else:
        tr_loader = None
        valid_set = dataset
    val_loader = data.DataLoader(dataset=valid_set,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    return tr_loader, val_loader
