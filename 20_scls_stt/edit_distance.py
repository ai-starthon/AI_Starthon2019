# -*- coding: utf-8 -*-

import sys
import numpy as np

def edit_distance(ref,hyp):
    """
    Edit distance between two sequences reference (ref) and hypothesis (hyp).
    Returns edit distance, number of insertions, deletions and substitutions to
    transform hyp to ref, and number of correct matches.
    """
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

def char_distance_error(ref,hyp):
    ref = ref.replace(' ', '')
    hyp = hyp.replace(' ', '')
    dist,ins,dels,subs,corr = edit_distance(ref,hyp)
    return float(dist) / float(dist + corr), dist, corr

def word_distance_error(ref,hyp):
    ref_ = ref.split()
    hyp_ = hyp.split()
    dist,ins,dels,subs,corr = edit_distance(ref_,hyp_)
    return float(dist) / float(dist + corr), dist, corr

def disp(ref,hyp):
    dist,ins,dels,subs,corr = edit_distance(ref,hyp)
    #print("Distance : %d" % (dist))

    err = float(dist) / float(dist + corr)
    print("== result Ins : %d, Dels : %d, Subs : %d, Corr : %d, Err : %f ==" % (ins,dels,subs,corr,err))

    return err

if __name__== "__main__":

    ref = 'abc def'
    hyp = 'ab1 def'

    print('ref:', ref)
    print('hyp:', hyp)
    print('CER:', char_distance_error(ref, hyp))
    print('WER:', word_distance_error(ref, hyp))

    ref = '안녕하세요'
    hyp = '안뇽 하세요'

    print('ref:', ref)
    print('hyp:', hyp)
    print('CER:', char_distance_error(ref, hyp))
    print('WER:', word_distance_error(ref, hyp))

