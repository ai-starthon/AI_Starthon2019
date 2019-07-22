from data_local_loader import get_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import argparse
import numpy as np
import time
import datetime

from data_loader import feed_infer
from evaluation import evaluation_metrics

try:
    import nsml
    DATASET_PATH = os.path.join(nsml.DATASET_PATH)
    print('start using nsml...!')
    print('DATASET_PATH: ', DATASET_PATH)
    use_nsml = True
except:
    DATASET_PATH = os.path.join('../nipa_video')
    print('use local gpu...!')
    use_nsml = False


class VideoResNet(models.ResNet):
    def __init__(self, block, layers, num_frames=5, num_classes=8):
        super().__init__(block, layers, num_classes=num_classes)
        self.conv1 = nn.Conv2d(3*num_frames, 64, 7, 2, 3)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
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


def get_resnet18(num_classes):
    return VideoResNet(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def bind_nsml(model, optimizer, task):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(dir_name, 'model.ckpt'))
        print('saved model checkpoints...!')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.ckpt'))
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print('loaded model checkpoints...!')

    def infer(root, phase):
        return _infer(root, phase, model=model, task=task)

    nsml.bind(save=save, load=load, infer=infer)
  

def _infer(root, phase, model, task):
    with torch.no_grad():
        model.eval()
        data_loader = get_loader(root=root, task=task, phase=phase)

        y_pred = []
        print('start infer')
        for i, (images, _, _) in enumerate(data_loader):
            images = images.cuda()
            logits = model(images)
            _, predicted = torch.max(logits, dim=1)
            y_pred += [predicted]

        print('end infer')
        y_pred = torch.cat(y_pred, dim=0).cpu().numpy().tolist()
    return y_pred

"""
def local_eval(model, task):
    prediction_file = 'pred.txt'
    feed_infer(prediction_file, lambda root, phase: _infer(root, phase, model, task))
    test_label_file = '../nipa_video/test/test_label'
    metric_result = evaluation_metrics(prediction_file, test_label_file)
    print(metric_result)
"""    

def main(config):
    model = get_resnet18(num_classes=config.num_classes)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    if use_nsml:
        bind_nsml(model, optimizer, config.task)
    if config.pause:
        nsml.paused(scope=locals())
    
    #print('start eval')
    #local_eval(model, config.task)
    
    if config.mode == 'train':
        train_loader = get_loader(
            root=DATASET_PATH, phase='train', task=config.task, batch_size=config.batch_size)
        
        # start training
        start_time = datetime.datetime.now()
        iter_per_epoch = len(train_loader)
        print('start training...!')
        for epoch in range(config.num_epochs):
            for i, (images, _, labels) in enumerate(train_loader):
                images = images.cuda()
                labels = labels.cuda()

                # forward
                logits = model(images)
                loss = F.cross_entropy(logits, labels)

                # backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % config.print_every == 0:
                    elapsed = datetime.datetime.now() - start_time
                    print ('Elapsed [%s], Epoch [%i/%i], Step [%i/%i], Loss: %.4f' 
                           % (elapsed, epoch+1, config.num_epochs, i+1, iter_per_epoch, loss.item()))

            if (epoch+1) % config.save_every == 0:
                nsml.save(str(epoch + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--task', type=str, default='age')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    
    # reserved for nsml
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--iteration", type=str, default='0')
    parser.add_argument("--pause", type=int, default=0)
    
    config = parser.parse_args()
    main(config)