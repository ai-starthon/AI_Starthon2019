import os
from PIL import Image
import torch
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


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def make_dataset(dir, extensions):
    image_paths = []
    dir = os.path.expanduser(dir)

    def is_valid_file(x):
        return has_file_allowed_extension(x, extensions)

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                image_paths.append(path)

    return image_paths


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class CustomTestDataset(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/xxx.ext
        root/xxy.ext
        root/xxz.ext

        root/123.ext
        root/nsdf3.ext
        root/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.

     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """
    _repr_indent = 4

    def __init__(self, root, loader=default_loader,
                 extensions=IMG_EXTENSIONS, transform=None):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        self.transforms = transforms

        self.transform = transform
        samples = make_dataset(self.root, extensions)
        if len(samples) == 0:
            raise (RuntimeError(
                "Found 0 files in subfolders of: " + self.root + "\n"
                + "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image_id, sample)
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        image_id = path.split('/')[-1]

        return image_id, sample

    def __len__(self):
        return len(self.samples)


def test_data_loader(root, batch_size=256):
    dataset = CustomTestDataset(root, transform=get_transform(
        random_crop=False))
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=False)


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


def data_loader_with_split(root, train_split=0.9, batch_size=256,
                           val_label_file='./val_label'):
    dataset = CustomDataset(root, transform=get_transform(
        random_crop=True))
    split_size = int(len(dataset) * train_split)
    train_set, valid_set = data.random_split(
        dataset, [split_size, len(dataset) - split_size])
    tr_loader = data.DataLoader(dataset=train_set,
                                batch_size=batch_size,
                                shuffle=True)
    val_loader = data.DataLoader(dataset=valid_set,
                                 batch_size=batch_size,
                                 shuffle=False)
    gt_labels = {valid_set[idx][0]: valid_set[idx][2]
                 for idx in range(len(valid_set))}
    gt_labels_string = [' '.join([str(s) for s in l])
                        for l in list(gt_labels.items())]
    with open(val_label_file, 'w') as file_writer:
        file_writer.write("\n".join(gt_labels_string))
    return tr_loader, val_loader, val_label_file
