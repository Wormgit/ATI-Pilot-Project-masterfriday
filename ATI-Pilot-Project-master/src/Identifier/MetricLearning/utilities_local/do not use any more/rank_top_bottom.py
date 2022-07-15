import os
import numpy as np
import shutil

date = [2,5,2,13] # start month date, end    Dataset: start from 5 Feb, end in March 11. 30 days (Feb: No 6,7,8,9,17,30)
#first 5 days: 5 10 11 12 13

destination_top = '/home/io18230/Desktop/top/'
destination_btm = '/home/io18230/Desktop/btm/'
destination_rev = '/home/io18230/Desktop/rev/'

path = '/home/io18230/Desktop/output/3visual/current084_model_state20'  #20
path = '/home/io18230/Desktop/output/3visual/exclusd4day'               # 182

path = '/home/io18230/Desktop/output/2-5_2-13/3visual/current035_retrain_for_animal'
path = '/home/io18230/Desktop/output/2-5_2-13/3visual/current056_16days_tembest'
filename = 'top_bottom.npz'

rootImagePath = '/home/io18230/Desktop/Will20'
rootImagePath = '/home/io18230/Desktop/RGBDCows2020w/test_2class_4days_exclude_noblack'




embeddings = np.load(os.path.join(path,filename))

list_top = embeddings['top_path']
list_btm = embeddings['bottom_path']
list_rev = embeddings['revolt_path']

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

Third_class = '0'
def copy_imags(list, destination, date= [1,1,12,30]):
    for item in list:
        pos = item.find('2020-')
        month_ = int(item[pos + 5:pos + 7])
        date_ = int(item[pos + 8:pos + 10])
        if (date[0]<=month_)* (month_<=date[2]) * (date[1] <= date_)*(date_ <= date[3]):
            imgPath = os.path.join(rootImagePath, item)
            m = os.path.dirname(item)
            rest = item[4:]
            makedirs(os.path.join(destination, m, Third_class))
            shutil.copy(imgPath, os.path.join(destination, m, Third_class,rest))


copy_imags(list_top, destination_top, date)
copy_imags(list_btm, destination_btm, date)
copy_imags(list_rev, destination_rev, date)