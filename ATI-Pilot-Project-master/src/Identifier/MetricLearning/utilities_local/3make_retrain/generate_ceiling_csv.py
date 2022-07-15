#!/usr/bin/env python
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."
import numpy as np  # notes are in main_test
import os, csv, time, glob, shutil, copy, random
import matplotlib.pyplot as plt
import sys, math, warnings
import argparse
import shutil
#tf
#import keras
#import tensorflow as tf
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", default='/home/io18230/Desktop/correct.csv', type=str)
parser.add_argument("--img_path", default='/home/io18230/Desktop/RGB (copy)', type=str)
args = parser.parse_args()


def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

csvFile = csv.reader(open(args.csv_path, 'r'), delimiter=',')
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

showed = []
listLabel = []
listanchor = []
listcandidate =[]
count_overlap = 0
count_non_lap = 0
for item in ap_gt:
    if item in showed:
        continue
    repeat_posi = [i for i, v in enumerate(ap_gt) if v == item]
    if len (repeat_posi) > 1:
        count_overlap +=1
        listLabel.append(item)
        listanchor.append(ap_fold[repeat_posi[0]])
        tmp = []
        for i in range(1,len(repeat_posi)):
            tmp.append(ap_fold[repeat_posi[i]])
        listcandidate.append(tmp)#
    else:
        count_non_lap += 1
    showed.append(item)

print(f'overlap tracklets= {count_overlap}\nNO_lap ={count_non_lap}\nTotal = {count_overlap+count_non_lap}')



data1 = pd.DataFrame({'listLabel': listLabel, 'listanchor': listanchor, 'listcandidate': listcandidate})
data1.to_csv(os.path.join('/home/io18230/Desktop', 'ceiling_wacv.csv'))