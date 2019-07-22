from torch.utils import data
from torchvision import datasets, transforms


def get_transform(random_crop=True):
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform = []
    transform.append(transforms.Resize(256))
    if random_crop:
        transform.append(transforms.RandomResizedCrop(224))
        transform.append(transforms.RandomHorizontalFlip())
    else:
        transform.append(transforms.CenterCrop(224))
    transform.append(transforms.ToTensor())
    transform.append(normalize)
    return transforms.Compose(transform)


class CustomDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image_id, sample, target) where target is class_index of
                the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        image_id = path.split('/')[-1]

        return image_id, sample, target


def data_loader(root, phase='train', batch_size=256):
    if phase == 'train':
        is_train = True
    elif phase == 'test':
        is_train = False
    else:
        raise KeyError
    dataset = CustomDataset(root, transform=get_transform(
        random_crop=is_train))
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=is_train)


def data_loader_with_split(root, train_split=0.9, batch_size=256, val_label_file='./val_label'):
    dataset = CustomDataset(root, transform=get_transform(
        random_crop=True))
    split_size = int(len(dataset) * train_split)
    train_set, valid_set = data.random_split(dataset, [split_size, len(dataset) - split_size])
    tr_loader = data.DataLoader(dataset=train_set,
                                batch_size=batch_size,
                                shuffle=True)
    val_loader = data.DataLoader(dataset=valid_set,
                                 batch_size=batch_size,
                                 shuffle=False)
    gt_labels = {valid_set[idx][0]: valid_set[idx][2] for idx in range(len(valid_set))}
    gt_labels_string = [' '.join([str(s) for s in l])
                        for l in list(gt_labels.items())]
    with open(val_label_file, 'w') as file_writer:
        file_writer.write("\n".join(gt_labels_string))
    return tr_loader, val_loader, val_label_file
