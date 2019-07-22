import numpy as np
from PIL import Image

import torch
from torchvision import utils as tutils
from torchvision import transforms


def compose(x_input, x_hat, mask):
    x_composed = x_input * mask + x_hat * (1 - mask)
    return x_composed


def l1_loss(x_composed, x_GT):
    loss = torch.abs(x_composed - x_GT).sum(dim=[1, 2, 3]).mean()
    return loss


def cutout(images):
    # for ease of use
    _, height, width = images.shape

    alpha = 1
    lam = np.random.beta(alpha, alpha)

    # cutout images
    mask = torch.ones_like(images)
    from_x, from_y, to_x, to_y = random_bbox(width, height, lam)
    mask[:, from_y:to_y, from_x:to_x] = 0
    images = images * mask
    return images, mask[:1, :, :]


def random_bbox(width, height, lam):
    cut_ratio = np.sqrt(1. - lam)
    cut_width = (width * cut_ratio).astype(np.int)
    cut_height = (height * cut_ratio).astype(np.int)
    # uniform
    from_x = np.random.randint(width - cut_width)
    from_y = np.random.randint(height - cut_height)
    to_x = from_x + cut_width
    to_y = from_y + cut_height
    return from_x, from_y, to_x, to_y


def normalize(x):
    out = x * 2 - 1
    return out


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, fname):
    x = denormalize(x)
    tutils.save_image(x, fname, nrow=1, padding=0)


def load_image(fname, image_size=128):
    transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor()])
    image = Image.open(fname).convert('RGB')
    image = transform(image)
    return image
