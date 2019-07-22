import os
import math
import datetime

import numpy as np

import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import torchvision.models as models
import argparse

from data_loader import feed_infer
from data_local_loader import data_loader, data_loader_with_split
from evaluation import evaluation_metrics


try:
    import nsml
    from nsml import DATASET_PATH, IS_ON_NSML
    TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, 'train', 'train_data')
    VAL_DATASET_PATH = None
except:
    IS_ON_NSML=False
    TRAIN_DATASET_PATH = os.path.join('/media/yoo/Data/NIPAFaceCls/train/train_data')
    VAL_DATASET_PATH = None

class ClsResNet(models.ResNet):
    def forward(self, x, extract=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def get_resnet18(num_classes=100):
    return ClsResNet(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def _infer(model, root_path, test_loader=None):
    if test_loader is None:
        test_loader = data_loader(
            root=os.path.join(root_path, 'test_data'),
            phase='test')

    res_fc = None
    res_id = None
    for idx, (data_id, image, _) in enumerate(test_loader):
        image = image.cuda()
        fc = model(image, extract=True)
        fc = fc.detach().cpu().numpy()

        if idx == 0:
            res_fc = fc
            res_id = data_id
        else:
            res_fc = np.concatenate((res_fc, fc), axis=0)
            res_id = res_id + data_id

    res_cls = np.argmax(res_fc, axis=1)


    return [res_id, res_cls]


def local_eval(model, test_loader=None, test_label_file=None):
    prediction_file = 'pred_train.txt'
    feed_infer(prediction_file, lambda root_path: _infer(model, root_path, test_loader=test_loader))
    if not test_label_file:
        test_label_file = os.path.join(VAL_DATASET_PATH, 'test_label')
    metric_result = evaluation_metrics(
        prediction_file,
        test_label_file)
    print('Eval result: {:.4f}'.format(metric_result))
    return metric_result


def bind_nsml(model, optimizer, scheduler):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(state, os.path.join(dir_name, 'model.pth'))
        print('saved')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pth'))
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        print('loaded')

    def infer(root_path, top_k=1):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer)


def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


if __name__ == '__main__':
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--train_split", type=float, default=0.9)
    args.add_argument("--num_classes", type=int, default=100)
    args.add_argument("--lr", type=int, default=0.001)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=100)
    args.add_argument("--print_iter", type=int, default=10)
    args.add_argument("--eval_split", type=str, default='val')

    # reserved for nsml
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--iteration", type=str, default='0')
    args.add_argument("--pause", type=int, default=0)

    config = args.parse_args()

    train_split = config.train_split
    num_classes = config.num_classes
    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    print_iter = config.print_iter
    eval_split = config.eval_split
    mode = config.mode

    model = get_resnet18(num_classes=num_classes)
    loss_fn = nn.CrossEntropyLoss()
    init_weight(model)

    if cuda:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    optimizer = Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=base_lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

    if IS_ON_NSML:
        bind_nsml(model, optimizer, scheduler)

        if config.pause:
            nsml.paused(scope=locals())


    if mode == 'train':
        tr_loader, val_loader, val_label = data_loader_with_split(root=TRAIN_DATASET_PATH, train_split=train_split)
        time_ = datetime.datetime.now()
        num_batches = len(tr_loader)

        local_eval(model, val_loader, val_label)

        for epoch in range(num_epochs):
            scheduler.step()
            model.train()
            for iter_, data in enumerate(tr_loader):
                _, x, label = data
                if cuda:
                    x = x.cuda()
                    label = label.cuda()
                pred = model(x)
                loss = loss_fn(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (iter_ + 1) % print_iter == 0:
                    elapsed = datetime.datetime.now() - time_
                    expected = elapsed * (num_batches / print_iter)
                    _epoch = epoch + ((iter_ + 1) / num_batches)
                    print('[{:.3f}/{:d}] loss({}) '
                          'elapsed {} expected per epoch {}'.format(
                              _epoch, num_epochs, loss.item(), elapsed, expected))
                    time_ = datetime.datetime.now()

            if IS_ON_NSML:
                nsml.save(str(epoch + 1))

            local_eval(model, val_loader, val_label)
            time_ = datetime.datetime.now()
            elapsed = datetime.datetime.now() - time_
            print('[epoch {}] elapsed: {}'.format(epoch + 1, elapsed))
