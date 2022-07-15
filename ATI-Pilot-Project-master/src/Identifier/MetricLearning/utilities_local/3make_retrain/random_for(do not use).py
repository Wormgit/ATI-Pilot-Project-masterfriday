#!/usr/bin/env python
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."
import numpy as np  # notes are in main_test
import os, csv, time, glob, shutil, copy, random
import matplotlib.pyplot as plt
import sys, math, warnings
import argparse
import json
import shutil
import pandas as pd




parser = argparse.ArgumentParser()
parser.add_argument("--csv_GMM", default='/home/io18230/Desktop/correct.csv', type=str)
parser.add_argument("--out_path", default='/home/io18230/Desktop/', type=str)
parser.add_argument("--merge", default=0, type=int)
args = parser.parse_args()


def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


csvFile = csv.reader(open(args.csv_GMM, 'r'), delimiter=',')
reader = list(csvFile)
del reader[0]
ap_gt = []
ap_fold = []
for i in range(len(reader) - 1):
    ap_fold.append(reader[i][1])
    if reader[i][3] == '1':
        ap_gt.append((int(float(reader[i][4]))))
    else:
        ap_gt.append((int(float(reader[i][2]))))

#random.seed(10)
times = 10
count = 0
for k in range(times):
    for i in range(400):
        slice = random.sample(ap_fold, 2)
        idx1 = ap_fold.index( slice[0])
        idx2 = ap_fold.index(slice[1])
        if ap_gt[idx1] ==ap_gt[idx2]:
            print('k=',k,i,slice[0],slice[1],ap_gt[idx1])
            count +=1
print(f'avrage:',count/times)






