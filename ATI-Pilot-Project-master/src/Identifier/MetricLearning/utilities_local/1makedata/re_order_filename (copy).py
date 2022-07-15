import random
import argparse
import pandas as pd
import os, copy
import csv

# output file                 # 1 is easy, 0 is difficult (similar)
# anchor  positive  negative  (similarity of anchor & p)  (similarity of anchor & n)
from shutil import copyfile
import shutil
parser = argparse.ArgumentParser()
parser.add_argument('--frame_file', default='/home/io18230/Desktop/Identification_WILL155_order/RGB', type=str)
args = parser.parse_args()




def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise





def make_train_data(items,it):
    dest = args.frame_file+'/'+ 'tmp'
    src = os.path.join(args.frame_file + '/' + items)


    makedirs(dest)
    shutil.move(src, dest) #original, target)
    os.rename(dest + '/' + items, dest + '/0')
    os.rename(dest, args.frame_file+'/'+items)


# load info
c_image = []
c_folder= []
pair_list= []
count_n_r = os.listdir(args.frame_file) #不管order
count_n_r.sort()
for i, items in enumerate(count_n_r):
    use = make_train_data(items,i)
    print(i, items)
print('Done')
