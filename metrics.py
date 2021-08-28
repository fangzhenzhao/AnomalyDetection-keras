from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def nums(preds, labels):
    fpr, tpr, _ = roc_curve(labels, preds)
    return fpr, tpr

def ErrorRateAt95Recall1(lens, scores, labels):
    sc = scores
    recall_point = 0.95
    labels1 = np.asarray(labels[0:lens])
    scores1 = np.asarray(scores[0:lens])  
    sorted_scores = sorted(scores1,reverse=True) 
    n_thresh = recall_point * lens
    FP = 0
    TN = 0
#     print('threshold is：',sorted_scores[int(n_thresh)-1])
    for i in range(lens,scores.shape[0]):
        if (sc[i]>=sorted_scores[int(n_thresh)-1]):
             FP += 1
        else:
            TN += 1    
    TP = 0
    FN = 0
    for i in range(lens):
        if (sc[i]>=sorted_scores[int(n_thresh)-1]):
             TP += 1
        else:
            FN += 1
    return FP,TN,TP,FN

def fpr_at_95_tpr(scores, labels):
    sc = scores
#     lens = int(scores.shape[0] / 2)
    lens = 2000  #if ID is MNIST or CIFAR10, lens=2000, SVHN-->5206 
    recall_point =0.95
    labels1 = np.asarray(labels[0:lens])
    scores1 = np.asarray(scores[0:lens])         
    sorted_scores = sorted(scores1,reverse=True) 
    n_thresh = recall_point * lens
    FP = 0
    TN = 0
    for i in range(lens,scores.shape[0]):
        if (sc[i]>=sorted_scores[int(n_thresh)-1]):
             FP += 1
        else:
            TN += 1    
    TP = 0
    FN = 0
    for i in range(lens):
        if (sc[i]>=sorted_scores[int(n_thresh)-1]):
             TP += 1
        else:
            FN += 1
    fpr =float(FP) / float(FP + TN+ 1e-7)
    return fpr


def detection_error(scores, labels):
    sc = scores
#     lens = int(scores.shape[0] / 2)
    lens = 2000  #if ID is MNIST or CIFAR10, lens=2000, SVHN-->5206
    recall_point =0.95
    labels1 = np.asarray(labels[0:lens])
    scores1 = np.asarray(scores[0:lens])
    sorted_scores = sorted(scores1,reverse=True) 
    n_thresh = recall_point * lens
    FP = 0
    TN = 0
    for i in range(lens,scores.shape[0]):
        if (sc[i]>=sorted_scores[int(n_thresh)-1]):
             FP += 1
        else:
            TN += 1    
    TP = 0
    FN = 0
    for i in range(lens):
        if (sc[i]>=sorted_scores[int(n_thresh)-1]):
             TP += 1
        else:
            FN += 1
 
    _detection_error = (1.0- float(TP) / float(TP + FN+ 1e-7)+ float(FP) / float(FP + TN+ 1e-7))/2
    return _detection_error
