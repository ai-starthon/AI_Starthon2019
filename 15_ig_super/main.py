import argparse, os
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from importlib import import_module
from data_local_loader import test_loader, data_loader_with_split
from data_loader import feed_infer
from evaluation import evaluation_metrics, evaluate

# Training settings
parser = argparse.ArgumentParser(description="NIPA Challenge SR")
parser.add_argument("--network_archi", default="edsr", type=str, help="network archi (default=edsr)")
parser.add_argument("--batchSize", type=int, default=32, help="batch size (default=32)")
parser.add_argument("--nEpochs", type=int, default=1, help="number of epochs to train for (default=1)")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate (default=1e-4)")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, (default: 0.9)")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay (default: 1e-4)")

# *** Reserved for nsml ***
# you should make sure that 'nsml-local' is already installed
import nsml
from nsml import DATASET_PATH, IS_ON_NSML 
from logger import Logger     

parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--iteration", type=str, default='0')
parser.add_argument("--pause", type=int, default=0)

def bind_nsml(model, optimizer):
    
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(dir_name, 'model.pth'))
        print('saved')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pth'))
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print('loaded')

    def infer(root_path):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer)
    
def _infer(model, root_path, data_loader=None):
    if data_loader is None:
        data_loader = test_loader(root=root_path)

    preds = []
    s_t = time.time()
    for idx, image in enumerate(data_loader):
        image = image.cuda()
        pred_x4 = model(image)                
        preds.append(pred_x4.detach().cpu())            
        if time.time() - s_t > 10:
            print('Infer batch {}/{}.'.format(idx + 1, len(data_loader)))
    preds = torch.cat(preds, dim=0).numpy()

    return preds
# *** Reserved for nsml *** (end) 

def local_eval(model, data_loader=None):    
    prediction_file = 'test_pred'
    feed_infer(prediction_file, lambda root_path: _infer(model, root_path, data_loader=data_loader))
    test_label_file = '/home/data/nipa_faces_sr_tmp2/test/test_label' # local datapath
    metric_result = evaluation_metrics(
        prediction_file,
        test_label_file)
    print('Eval result: {:.4f}'.format(metric_result))
    return metric_result

def main():
    
    global opt, model    
    opt = parser.parse_args()
    cudnn.benchmark = True
    
    log = Logger()
    
    # Building model
    module_net = import_module('model.'+opt.network_archi)        
    model = getattr(module_net, 'Net')()    
    criterion = getattr(module_net, 'criterion')() 
    model = model.cuda()
    criterion = criterion.cuda()
    
    # Setting Optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    
    # *** Reserved for nsml ***    
    bind_nsml(model, optimizer)
    if opt.pause:
        nsml.paused(scope=locals())    
    # *** Reserved for nsml *** (end) 
    
    if opt.mode == "train":
        if IS_ON_NSML:        
            opt.dataset_path = os.path.join(DATASET_PATH, 'train', 'train_data') 
        else:
            opt.dataset_path = '/home/data/nipa_faces_sr_tmp2/train/train_data' # local datapath
        training_data_loader, val_loader = data_loader_with_split(opt.dataset_path, train_split=0.9 , batch_size=opt.batchSize)

        # Training    
        for epoch in range(opt.nEpochs):        
            if opt.network_archi.startswith("edsr"):
                average_epoch_loss_train = train(training_data_loader, val_loader, optimizer, model, criterion, epoch)
                info = {'train_loss' : average_epoch_loss_train}

            nsml.save(str(epoch + 1))
            for tag, value in info.items():
                log.scalar_summary(tag, value, epoch)                
        
def train(training_data_loader, val_loader, optimizer, model, criterion, epoch):
    
    model.train()
    epoch_loss_train = []
    
    for iteration, batch in enumerate(training_data_loader, 1):
        input, label_x4 = batch[0], batch[1]
        input = input.cuda()        
        label_x4 = label_x4.cuda()

        pred_x4 = model(input)
        loss_x4 = criterion(pred_x4, label_x4)
        optimizer.zero_grad()
        loss_x4.backward()
        optimizer.step()
        epoch_loss_train.append(loss_x4.item())
        if iteration % 100 == 0:
            print('iteration : ', iteration, ' / ', len(training_data_loader) )
            PSNR_model = evaluate(label_x4.detach().cpu().numpy(), pred_x4.detach().cpu().numpy())
            PSNR_bilinear = evaluate(label_x4.detach().cpu().numpy(), F.interpolate(input.detach().cpu(),scale_factor=4, mode='bilinear',align_corners=True).numpy())
            print('PSNR (train, model): {}, PSNR (train, bilinear): {}'.format(PSNR_model,PSNR_bilinear))
    average_epoch_loss_train = sum(epoch_loss_train) / len(epoch_loss_train)
    print('epoch finished:', average_epoch_loss_train)        
    if not IS_ON_NSML:        
        local_eval(model)        
    return average_epoch_loss_train

if __name__ == "__main__":
    main()
