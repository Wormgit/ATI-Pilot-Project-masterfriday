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
#tf
#import keras
#import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", default='/home/io18230/Desktop/correct.csv', type=str)
parser.add_argument("--output", default='/home/io18230/Desktop/sdfdsg', type=str)
parser.add_argument("--img_path", default='/home/io18230/Desktop/RGBDCows2020/test_2class_4days_exclude_noblack', type=str)
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
ap_gt = list(set(ap_gt))

folder_name = []
for item in sorted(os.listdir(args.img_path)):
    folder_name.append(int(item))

makedirs(args.output)

count = 0
for item in ap_gt:
    if item in folder_name:
        sr = os.path.join(args.img_path,"%03d" % item)
        de = os.path.join(args.output,"%03d" % item)
        shutil.copytree(sr,de)
        count +=1
    else:
        print(f'**did not find {item}')

print(f'**the total copy folder is {count}')