import os
import sys
import numpy as np
import argparse

def edit_distance(ref,hyp):
    n = len(ref)
    m = len(hyp)

    ins = dels = subs = corr = 0
    
    D = np.zeros((n+1,m+1))

    D[:,0] = np.arange(n+1)
    D[0,:] = np.arange(m+1)

    for i in range(1,n+1):
        for j in range(1,m+1):
            if ref[i-1] == hyp[j-1]:
                D[i,j] = D[i-1,j-1]
            else:
                D[i,j] = min(D[i-1,j],D[i,j-1],D[i-1,j-1])+1

    i=n
    j=m
    while i>0 and j>0:
        if ref[i-1] == hyp[j-1]:
            corr += 1
        elif D[i-1,j] == D[i,j]-1:
            ins += 1
            j += 1
        elif D[i,j-1] == D[i,j]-1:
            dels += 1
            i += 1
        elif D[i-1,j-1] == D[i,j]-1:
            subs += 1
        i -= 1
        j -= 1

    ins += i
    dels += j

    return D[-1,-1],ins,dels,subs,corr

def load_ref(path):
    ref_dict = dict()
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            ref_dict[key] = target

    return ref_dict

def load_hyp(path):
    hyp_dict = dict()
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            key = key.split('.')[0] # remove file-extention 'wav'
            hyp_dict[key] = target

    return hyp_dict

def evaluation_metrics(hyp_path, ref_path):
    hyp_dict = load_hyp(hyp_path)
    ref_dict = load_ref(ref_path)

    dist_sum = 0
    corr_sum = 0

    for k, hyp in hyp_dict.items():

        k = k.split('/')[-1]

        hyp = hyp.replace(' ', '')
        ref = ref_dict[k].replace(' ', '')
        
        dist, _, _, _, corr = edit_distance(ref, hyp)

        dist_sum += dist
        corr_sum += corr

    return float(dist_sum) / float(dist_sum + corr_sum)

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--prediction', type=str, default='pred.txt')
    config = args.parse_args()
   
    test_label_path = '/data/20_scls_stt/test/test_label'

    CER = evaluation_metrics(config.prediction, test_label_path)
    print('%0.4f' % (CER))
