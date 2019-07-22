import torch
from torch.autograd import Variable
import numpy as np

from nsml import DATASET_PATH
import nsml

#import glob
import os
import argparse

FEATURE_DIM = 14 #지역(0~9), 연(2016~2019), 월, 일, t-5 ~ t-1의 미세 & 초미세
OUTPUT_DIM = 2 # t-time의 (미세, 초미세)

def bind_model(model):
    def save(path, *args, **kwargs):
        # save the model with 'checkpoint' dictionary.
        checkpoint = {
            'model': model.state_dict(),
        }
        torch.save(checkpoint, os.path.join(path, 'model.pth'))

    def load(path, *args, **kwargs):
        checkpoint = torch.load(os.path.join(path, 'model.pth'))
        model.load_state_dict(checkpoint['model'])

    def infer(path, **kwargs):
        return inference(path, model, config)

    nsml.bind(save, load, infer)


def inference(path, model, config, **kwargs):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    #test_dataset = Dataset(vocab)
    #test_path = os.path.join(path, 'test_data')
    #test_dataset.create_instances(test_path, config.max_seq_length, type='test')
    #test_loader = DataLoader(test_dataset, batch_size=1)
    test_path = DATASET_PATH+'/test/test_data'
    data = convData(np.load(test_path))
    mean10_val = np.mean(data[:][4::2])
    test_data = Variable(torch.tensor(data))
    pred_val = model(test_data/mean10_val).detach().numpy().tolist()
    pred_results = []
    for step, val in enumerate(pred_val):
        pred_results.append([step, val])
    #pred_results = model(test_data).detach().numpy().reshape(len(test_data))
    
    #pred_results = []
    #for step, batch in enumerate(test_loader):
    #    batch = tuple(t.to(device) for t in batch)
    #    batch = sort_batch(batch)
    #    input_ids, input_lengths, labels = batch

    #    outputs = model(input_ids, input_lengths)
    #    top_1_result = outputs['predicted_intents'][0].item()
    #    pred_results.append([step, top_1_result])

    return pred_results
 
def convData(data_arr):
    v = np.zeros(FEATURE_DIM, dtype=np.float32)
    v[1] = 2016
    new_d = np.asarray([d - v for d in data_arr])
    return new_d
    
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(FEATURE_DIM, OUTPUT_DIM)        
        return
    
    def forward(self, x):
        p_Y = self.linear(x)
        return p_Y

if __name__ == '__main__':
    
    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    
    config = args.parse_args()
    
    
    # Bind model
    model = Model()
    bind_model(model)
    
    
    # DONOTCHANGE: They are reserved for nsml
    # Warning: Do not load data before the following code!
    if config.pause:
        nsml.paused(scope=locals())
    
    if config.mode == "train":
    


        train_dataset_path = DATASET_PATH + '/train/train_data'
        #train_data_files = sorted(glob.glob(train_dataset_path + '/*.npy')) 
        data = convData(np.load(train_dataset_path))
        mean10_val = np.mean(data[:][4::2])
                
        tr_X = Variable(torch.tensor(data))
    
        train_label_file = DATASET_PATH + '/train/train_label' # All labels are zero in train data.
        tr_Y = Variable(torch.tensor(np.load(train_label_file))) # numpy array of labels. They are all zeros!!
    
    #tr_X = np.load(DATASET_PATH + '/train/train_data.npy')    
    #tr_Y = np.load(DATASET_PATH + '/train/train_label.npy') # numpy array of labels. They are all zeros!!

        EP, LR = 100, 0.01    
        crit = torch.nn.MSELoss(reduction='mean')
        opt = torch.optim.SGD(model.parameters(), lr=LR)
    # Train
        for ep in range(EP):
            p_Y = model(tr_X/mean10_val)    
            loss = 0.3*crit(p_Y[:][0], tr_Y[:][0])+0.7*crit(p_Y[:][1], tr_Y[:][1])    
        
            opt.zero_grad()
            loss.backward()
            opt.step()
    
        #te_p_Y = model(data.te_X)
        #np_val = te_p_Y.detach().numpy().reshape(len(te_p_Y))
        #te_p_Y = Variable(torch.Tensor(np.where(np_val < 0, 0, np_val)))
        #te_loss = crit(te_p_Y, data.te_Y)
            if ep % 10 == 0:
                nsml.save(ep)
                print(ep, np.sqrt(loss.data.item()))#, #np.sqrt(te_loss.data.item()))
    
        nsml.save(ep)    
        
    # Save
    #epoch = 1
    #nsml.save(epoch) # If you are using neural networks, you may want to use epoch as checkpoints
    
    # Load test (Check if load method works well)
    #nsml.load(epoch)
    
    # Infer test
    #for file in train_data_files[:10]:
    #    data = np.load(file)
    #    print(model.forward(data))    