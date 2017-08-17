#!/usr/bin/python
# -*- coding: UTF-8 -*-

__author__ = 'tianle.li'
__date__ = '2017-07-20'

import numpy as np
from sklearn.metrics import roc_auc_score
import sys

a = sys.argv[1]
a=float(0.2)
print "## a=" + str(a)


def precision_score(y_true, y_pred, a):
    return ((y_true == 1) * (y_pred > a)).sum() / float((y_pred > a).sum())


def recall_score(y_true, y_pred, a):
    return ((y_true == 1) * (y_pred > a)).sum() / float((y_true == 1).sum())


def f1_score(y_true, y_pred, a):
    num = 2 * precision_score(y_true, y_pred, a) * recall_score(y_true, y_pred, a)
    deno = (precision_score(y_true, y_pred, a) + recall_score(y_true, y_pred, a))
    return num / deno

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


result = []
# for line in open("/home/zhipengwu/secureCRT/hotel_test_20170813_10w.out.cal"):
for line in open("/home/zhipengwu/work/kaggle-2014-criteo/hotel_test_20170813_10w.out.cal"):
    # print line.replace("\n","")
    result.append(float(line.replace("\n", "")))

# print str(result)

test = []
# for line in open("/home/zhipengwu/secureCRT/hotel_test_20170813_10w_no_head.libsvm.csv"):
for line in open("/home/zhipengwu/work/kaggle-2014-criteo/hotel_test_20170813_10w_no_head.libsvm.csv"):
    # print line.split(" ")[0]
    test.append(int(line.split(",")[0]))

# print str(test)

# y_true = np.array([0, 0, 0, 0,0,0,0,0,0,1])
# y_scores = np.array([0.1, 0.4, 0.35, 0.8])
y_true = np.array(test)
y_scores = np.array(result)

# print str(y_true)
# print str(y_scores)
for a in frange (0.1, 0.3, 0.01):
    print "## a=" + str(a)
    TP = float(((y_scores > float(a)) * (y_true == 1)).sum())
    FP = float(((y_scores > float(a)) * (y_true == 0)).sum())
    FN = float(((y_scores <= float(a)) * (y_true == 1)).sum())
    TN = float(((y_scores <= float(a)) * (y_true == 0)).sum())



    print "TP=\t"+str(((y_scores>float(a))*(y_true==1)).sum())
    print "FP=\t"+str(((y_scores>float(a))*(y_true==0)).sum())
    print "FN=\t"+str(((y_scores<=float(a))*(y_true==1)).sum())
    print "TN=\t"+str(((y_scores<=float(a))*(y_true==0)).sum())

    print "auc:\t"+str(roc_auc_score(y_true, y_scores))
    print "p:\t"+str(precision_score(y_true,y_scores,float(a)))
    print "r:\t"+str(recall_score(y_true,y_scores,float(a)))
    print "f1:\t"+str(f1_score(y_true,y_scores,float(a)))
    print "ratio:\t" + str(FP / (FP + TN))

# print "precision="+str(TP/(TP+FP))
# print "recal="+str(TP/(TP+FN))
