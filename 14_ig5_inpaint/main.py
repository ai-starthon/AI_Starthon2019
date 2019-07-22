import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

from data_local_loader import data_loader, data_loader_with_split
from utils import compose, l1_loss, load_image, normalize

try:
    import nsml
    dir_data_root = nsml.DATASET_PATH
    use_nsml = True
except ImportError:
    dir_data_root = '/home/data/nipa_inpaint'
    print('run without NSML')
    print('dir_data_root:', dir_data_root)
    use_nsml = False

try:
    from tqdm import tqdm, trange
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x, desc='', **kwargs):
        if len(desc) > 0:
            print(desc, end=' ')
        return x

    def trange(x, desc='', **kwargs):
        if len(desc) > 0:
            print(desc, end=' ')
        return range(x)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--eval_every', type=int, default=40)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--dir_ckpt', type=str, default='ckpt')
    # parser.add_argument()
    # reserved for nsml
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--iteration", type=str, default='0')
    parser.add_argument("--pause", type=int, default=0)
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Inpaint()
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    save, load = bind_nsml(model, optim)
    if args.pause == 1:
        nsml.paused(scope=locals())

    if args.mode == 'train':
        path_train = os.path.join(dir_data_root, 'train')
        path_train_data = os.path.join(dir_data_root, 'train', 'train_data')
        tr_loader, val_loader = data_loader_with_split(path_train, batch_size=args.batch_size)

        postfix = dict()
        total_step = 0
        for epoch in trange(args.num_epochs, disable=use_nsml):
            pbar = tqdm(enumerate(tr_loader), total=len(tr_loader), disable=use_nsml)
            for step, (_, x_input, mask, x_GT) in pbar:
                total_step += 1
                x_GT = x_GT.to(device)
                x_input = x_input.to(device)
                mask = mask.to(device)
                x_mask = torch.cat([x_input, mask], dim=1)

                model.zero_grad()
                x_hat = model(x_mask)
                x_composed = compose(x_input, x_hat, mask)
                loss = l1_loss(x_composed, x_GT)
                loss.backward()
                optim.step()
                postfix['loss'] = loss.item()

                if use_nsml:
                    postfix['epoch'] = epoch
                    postfix['step_'] = step
                    postfix['total_step'] = total_step
                    postfix['steps_per_epoch'] = len(tr_loader)

                if step % args.eval_every == 0:
                    vutils.save_image(x_GT, 'x_GT.png', normalize=True)
                    vutils.save_image(x_input, 'x_input.png', normalize=True)
                    vutils.save_image(x_hat, 'x_hat.png', normalize=True)
                    vutils.save_image(mask, 'mask.png', normalize=True)
                    metric_eval = local_eval(model, val_loader, path_train_data)
                    postfix['metric_eval'] = metric_eval
                if use_nsml:
                    if step % args.print_every == 0:
                        print(postfix)
                    nsml.report(**postfix, scope=locals(), step=total_step)
                else:
                    pbar.set_postfix(postfix)
            if use_nsml:
                nsml.save(epoch)
            else:
                save(epoch)


class Inpaint(nn.Module):
    def __init__(self):
        super(Inpaint, self).__init__()
        nf = 64
        self.conv1 = nn.Conv2d(4, nf, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        h = self.conv1(x)
        h = F.avg_pool2d(h, 2)
        h = self.conv2(h)
        h = F.avg_pool2d(h, 2)
        h = self.conv3(h)
        h = self.conv4(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.conv5(h)
        h = F.interpolate(h, scale_factor=2)
        x_hat = self.conv6(h)
        return x_hat


def _infer(model, root_path, test_loader=None):
    if test_loader is None:
        test_loader = data_loader(
            root=os.path.join(root_path, 'test_data'),
            phase='test')

    x_hats = []
    fnames = []
    desc = 'infer...'
    with torch.no_grad():
        for data in tqdm(test_loader, desc=desc, total=len(test_loader), disable=use_nsml):
            if isinstance(test_loader.dataset, torch.utils.data.dataset.Subset):
                fname, x_input, mask, _ = data
            else:
                fname, x_input, mask = data
            x_input = x_input.cuda()
            mask = mask.cuda()
            x_mask = torch.cat([x_input, mask], dim=1)
            x_hat = model(x_mask)
            x_hat = compose(x_input, x_hat, mask)
            x_hats.append(x_hat.cpu())
            fnames = fnames + list(fname)

    x_hats = torch.cat(x_hats, dim=0)

    return fnames, x_hats


def read_prediction_gt(dname, fnames):
    images = []
    for fname in fnames:
        fname = os.path.join(dname, fname)
        image = load_image(fname)
        image = normalize(image)
        images.append(image)
    return torch.stack(images, dim=0)


def local_eval(model, test_loader, path_GT):
    fnames, x_hats = _infer(model, None, test_loader=test_loader)
    x_GTs = read_prediction_gt(path_GT, fnames)
    loss = float(l1_loss(x_hats, x_GTs))
    print('local_eval', loss)
    return loss


def bind_nsml(model, optimizer):
    def save(dir_name, *args, **kwargs):
        if not isinstance(dir_name, str):
            dir_name = str(dir_name)
        os.makedirs(dir_name, exist_ok=True)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        fname = os.path.join(dir_name, 'model.pth')
        torch.save(state, fname)
        print('saved')

    def load(dir_name, *args, **kwargs):
        if not isinstance(dir_name, str):
            dir_name = str(dir_name)
        fname = os.path.join(dir_name, 'model.pth')
        state = torch.load(fname)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print('loaded')

    def infer(root_path):
        return _infer(model, root_path)

    if use_nsml:
        nsml.bind(save=save, load=load, infer=infer)
    return save, load


if __name__ == '__main__':
    main()
