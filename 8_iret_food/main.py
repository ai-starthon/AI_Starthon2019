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
from data_local_loader import test_data_loader, data_loader_with_split
from evaluation import evaluation_metrics

import nsml
from nsml import DATASET_PATH

TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, 'train', 'train_data')


def _infer(model, root_path, test_loader=None, local_val=False):
    """
    모델과 데이터가 주어졌을 때, 다음과 같은 데이터 구조를 반환하는 함수를 만들어야 합니다.

    [ [ query_image_id_1, predicted_database_image_id_1 ],
      [ query_image_id_2, predicted_database_image_id_2 ],
      ...
      [ query_image_id_N, predicted_database_image_id_N ] ]

    README 설명에서처럼 predicted_database_image_id_n 은 query_image_id_n 에 대해
    평가셋 이미지 1,...,n-1,n+1,...,N 를 데이터베이스로 간주했을 때에 가장 쿼리와 같은 카테고리를
    가질 것으로 예측하는 이미지입니다. 이미지 아이디는 test_loader 에서 extract 되는 첫번째
    인자인 data_id 를 사용합니다.

    Args:
      model: 이미지를 인풋으로 받아서 feature vector를 반환하는 모델
      root_path: 데이터가 저장된 위치
      test_loader: 사용되지 않음
      local_val: 사용되지 않음

    Returns:
      top1_reference_ids: 위에서 설명한 list 데이터 구조
    """
    if test_loader is None:
        test_loader = test_data_loader(
            root=os.path.join(root_path, 'test_data'))

    # TODO 모델의 아웃풋을 적당히 가공하고 연산하여 각 query에 대해 매치가 되는 데이터베이스
    # TODO 이미지의 ID를 찾는 모듈을 구현 (현재 구현은 베이스라인 - L2 정규화 및 내적으로 가장
    # TODO 비슷한 이미지 조회).
    feats = None
    data_ids = None
    s_t = time.time()
    for idx, data_package in enumerate(test_loader):
        if local_val:
            data_id, image, _ = data_package
        else:
            data_id, image = data_package
        image = image.cuda()
        feat = model(image, extract=True)
        feat = feat.detach().cpu().numpy()
        feat = feat / np.linalg.norm(feat, axis=1)[:, np.newaxis]
        if feats is None:
            feats = feat
        else:
            feats = np.append(feats, feat, axis=0)
        if data_ids is None:
            data_ids = data_id
        else:
            data_ids = np.append(data_ids, data_id, axis=0)

        if time.time() - s_t > 10:
            print('Infer batch {}/{}.'.format(idx + 1, len(test_loader)))

    score_matrix = feats.dot(feats.T)
    np.fill_diagonal(score_matrix, -np.inf)
    top1_reference_indices = np.argmax(score_matrix, axis=1)
    top1_reference_ids = [
        [data_ids[idx], data_ids[top1_reference_indices[idx]]] for idx in
        range(len(data_ids))]

    return top1_reference_ids


def local_eval(model, test_loader=None, test_label_file=None):
    prediction_file = 'pred_train.txt'
    feed_infer(prediction_file,
               lambda root_path: _infer(model,
                                        root_path, test_loader=test_loader,
                                        local_val=True))
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


class ImplementYourself(object):
    # TODO 베이스라인 image retrieval 모델입니다. 이것을 바탕으로 직접 구현하시면 됩니다.

    class FeatResNet(models.ResNet):
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
            if extract:
                return x
            x = self.fc(x)

            return x

    @staticmethod
    def get_resnet18(num_classes=150):
        return ImplementYourself.FeatResNet(models.resnet.BasicBlock,
                                            [2, 2, 2, 2],
                                            num_classes=num_classes)

    @staticmethod
    def init_weight(model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @staticmethod
    def get_optimizer():
        return Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=base_lr, weight_decay=1e-4)

    @staticmethod
    def get_scheduler(optimizer):
        return StepLR(optimizer, step_size=40, gamma=0.1)


if __name__ == '__main__':
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--train_split", type=float, default=0.9)
    args.add_argument("--num_classes", type=int, default=150)
    args.add_argument("--lr", type=int, default=0.01)
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

    model = ImplementYourself.get_resnet18(num_classes=num_classes)
    loss_fn = nn.CrossEntropyLoss()
    ImplementYourself.init_weight(model)

    if cuda:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    optimizer = ImplementYourself.get_optimizer()
    scheduler = ImplementYourself.get_scheduler(optimizer)

    bind_nsml(model, optimizer, scheduler)
    if config.pause:
        nsml.paused(scope=locals())

    if mode == 'train':
        # TODO 아래 트레이닝 코드도 베이스라인입니다. 변형 또는 새로 구현 가능합니다.

        tr_loader, val_loader, val_label = data_loader_with_split(
            root=TRAIN_DATASET_PATH, train_split=train_split)
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
                        _epoch, num_epochs, loss.item(),
                        elapsed, expected))
                    nsml.save(str(epoch + 1))
                    time_ = datetime.datetime.now()

            local_eval(model, val_loader, val_label)
            time_ = datetime.datetime.now()
            elapsed = datetime.datetime.now() - time_
            print('[epoch {}] elapsed: {}'.format(epoch + 1, elapsed))
