import tensorflow as tf
import numpy as np

from nsml import DATASET_PATH
import nsml

#import glob
import os
import argparse

FEATURE_DIM = 14 #지역(0~9), 연(2016~2019), 월, 일, t-5 ~ t-1의 미세 & 초미세
OUTPUT_DIM = 2 # t-time의 (미세, 초미세)

def bind_model(sess, run_params):
    saver = tf.train.Saver(max_to_keep=0) # 0:no limit
    
    def save(path, **kwargs):
        saver.save(sess, os.path.join(path, 'model.tf'))

    def load(path):
        saver.restore(sess, os.path.join(path, 'model.tf'))

    def infer(path):
        return inference(path, sess, run_params)
    nsml.bind(save, load, infer)

def inference(path, sess, run_params):
    test_path = path+'/test_data'
    test_data = convData(np.load(test_path))
    mean10_val = np.mean(test_data[:][4::2])
        
    pred_val = sess.run(run_params[0], feed_dict={run_params[1]: test_data/mean10_val}) # feed data into placeholder x, y_target
    pred_val = pred_val.tolist()
    pred_results = []
    for step, val in enumerate(pred_val):
        pred_results.append([step, val])

    return pred_results

def convData(data_arr):
    v = np.zeros(FEATURE_DIM, dtype=np.float32)
    v[1] = 2016
    new_d = np.asarray([d - v for d in data_arr])
    return new_d
    
if __name__ == '__main__':
    
    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    
    config = args.parse_args()
    
    
    with tf.Graph().as_default():
        # placeholder is used for feeding data.
        x = tf.placeholder("float", shape=[None, FEATURE_DIM]) # none represents variable length of dimension.
        y_target = tf.placeholder("float", shape=[None, OUTPUT_DIM]) # shape argument is optional, but this is useful to debug.

        W1 = tf.Variable(tf.zeros([FEATURE_DIM, OUTPUT_DIM]))
        b1 = tf.Variable(tf.zeros([OUTPUT_DIM]))
        y = tf.matmul(x, W1) + b1
        
        delta = (y - y_target)
        
        L1_delta = delta[:, 0]
        L2_delta = delta[:, 1]
        MSE_loss = tf.reduce_mean(0.3* L1_delta*L1_delta + 0.7*L2_delta*L2_delta)
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(MSE_loss)
        
        sess = tf.Session() # open a session which is a envrionment of computation graph.
        sess.run(tf.global_variables_initializer())# initialize the variables
        
        # Bind model
        run_params = [y, x]
        bind_model(sess, run_params)
    
        # DONOTCHANGE: They are reserved for nsml
        # Warning: Do not load data before the following code!
        if config.pause:
            nsml.paused(scope=locals())

        if config.mode == "train":
            train_dataset_path = DATASET_PATH + '/train/train_data'
            train_label_file = DATASET_PATH + '/train/train_label' # All labels are zero in train data.

            x_data = convData(np.load(train_dataset_path))
            y_data = np.load(train_label_file)
            mean10_val = np.mean(x_data[:][4::2])
            

            EP = 200
            for ep in range(EP):
                _, loss = sess.run([train_step, MSE_loss], feed_dict={x: x_data/mean10_val, y_target: y_data}) # feed data into placeholder x, y_target
                if ep % 10 == 0:
                    print(ep, loss)#, #np.sqrt(te_loss.data.item()))
                    nsml.save(ep)

            nsml.save(ep)    
            sess.close()
