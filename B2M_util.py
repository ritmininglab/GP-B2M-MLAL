# -*- coding: utf-8 -*-

import numpy as np
import pickle as pk
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Hyperparameter, DotProduct
from scipy.optimize import lsq_linear
from sklearn.preprocessing import normalize


def split(data,label=None,train_rate=0.1,candidate_rate=0.6,test_rate=0.3,seed=25,even=False):
    index=np.arange(data.shape[0])
    np.random.seed(seed)
    np.random.shuffle(index)
    if(even is False):#do not require each label have a positive instance in train.
        train_index=index[0:int(len(index)*train_rate)]
        candidate_index=index[int(len(index)*train_rate):int(len(index)*(train_rate+candidate_rate))]
        test_index=index[int(len(index)*(train_rate+candidate_rate)):]
        return list(train_index),list(candidate_index),list(test_index)
    elif(even is True):#scan index to determine the train index first. 
        label_stack=label[0,:]#label_stack is 0/1 L length vector, indicating whether a label has already appreared in the training
        train_index=[index[0]]
        i=0
        while (len(train_index)<int(len(index)*train_rate)):
            i=i+1
            current_label=np.sum(label_stack)
            if(current_label<label.shape[1]):  #then need a new training data with new positive label          
                updated_label=np.sum(np.logical_or(label[index[i],:],label_stack))
                if(updated_label>current_label):#if introducing the next data introduce a new label(s),add it. 
                    train_index.append(index[i])
                    label_stack=np.logical_or(label[index[i],:],label_stack)#update label stack
                    #print(np.sum(label_stack))
                else:#skip this data point
                    pass
            else:
                train_index.append(index[i])
        #delete the train index from index
        index=[x for x in index if x not in train_index]
        candidate_index=index[0:int(data.shape[0]*(candidate_rate))]
        #delete the candidate index from index
        index=[x for x in index if x not in candidate_index]
        test_index=index[0:int(data.shape[0]*(test_rate))]
        return list(train_index),list(candidate_index),list(test_index)




