import os
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms

from utils import cutout


class CustomDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (fname, masked, mask, GT)
        """
        path, target = self.samples[index]
        GT = self.loader(path)
        if self.transform is not None:
            GT = self.transform(GT)

        masked, mask = cutout(GT)
        fname = os.path.basename(path)

        return fname, masked, mask, GT


class TestDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dname_masked = os.path.join(root_dir, 'masked')
        self.dname_mask = os.path.join(root_dir, 'mask')
        self.samples = sorted(os.listdir(self.dname_masked))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (masked, mask)
        """
        fname = self.samples[index]
        fname_masked = os.path.join(self.dname_masked, fname)
        fname_mask = os.path.join(self.dname_mask, fname)
        masked = Image.open(fname_masked).convert('RGB')
        mask = Image.open(fname_mask).convert('RGB')
        if self.transform is not None:
            masked = self.transform(masked)
            mask = self.transform(mask)
        mask = (mask > 0.5).float()

        return fname, masked, mask[:1]

    def __len__(self):
        return len(self.samples)


def data_loader(root, phase='train', batch_size=32, num_workers=2):
    if phase == 'train':
        is_train = True
        Dset = CustomDataset
    elif phase == 'test':
        is_train = False
        Dset = TestDataset
    else:
        raise KeyError
    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = Dset(root, transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train,
                                 num_workers=num_workers)
    return dataloader


def data_loader_with_split(root, train_split=0.9, batch_size=32, num_workers=2):
    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CustomDataset(root=root, transform=transform)
    train_split_size = int(len(dataset) * train_split)
    val_split_size = len(dataset) - train_split_size
    # val_split_size = 64
    # train_split_size = len(dataset) - val_split_size
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
