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
parser.add_argument('--frame_file', default='/home/io18230/Desktop/Crop', type=str)
parser.add_argument('--min_image_number', default=3, type=float) # 3 for the old model
parser.add_argument('--agument_n', default=5, type=int)
args = parser.parse_args()
save_path = args.frame_file + '_csv_pari/'
agument_n = args.agument_n
min_image = agument_n * args.min_image_number

# need to input the start date
last_date = '2020-03-01'

shift = 19+5

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

def make_train_data(csv_path, items, number_order, day_count):
    use_this_folder = 0
    csvFile = csv.reader(open(csv_path, 'r'), delimiter=',')

    reader = list(csvFile)
    del reader[0]
    for i in range(len(reader)): # delete blanks from csv format and track points without valid box
        while '' in reader[i]:
            reader[i].remove('')
    count = len(reader)-1
    while True:
        #print(reader)
        if len(reader[count]) < 2:
            del reader[count]
        count -= 1
        if count < 0 :
            break


    for i in range(len(reader)):
        item_n = 0
        xmls = sorted(os.listdir(args.frame_file + '/' + items))
        same_list = []
        #dest = args.frame_file + '_split/' + str(number_order).zfill(3) + '/'+str(i)

        for tmm in range(1,len(reader[i])): # random 2 items search for 2 different items else go on   #positive pairs
            # if len(reader[i])==2:
            #     break
            if tmm == 1:
                ig = eval(reader[i][tmm])
                xc = ig[0]
                yc = ig[1]

                while 1:
                    if item_n == len(xmls):
                        break
                    xml = xmls[item_n] # from csv
                    # centres
                    x, y = int(xml[17+shift:21+shift]), int(xml[22+shift:26+shift])
                    if xc == x and yc == y:

                        for kkk in range(0,5):
                            same_list.append(xmls[item_n+kkk])
                        item_n = 0
                        break
                    item_n+=5
            else:
                ig = eval(reader[i][tmm])
                xc = ig[0]
                yc = ig[1]
                    #ig_number = ig[-1]
                #while item_n < len(xmls):
                while 1:
                    if item_n == len(xmls):
                        break
                    xml = xmls[item_n]
                    x, y = int(xml[17+shift:21+shift]), int(xml[22+shift:26+shift])
                    if xc == x and yc == y:

                        for kkk in range(0, 5):
                            same_list.append(xmls[item_n + kkk])
                        item_n = 0
                        break
                    item_n += 5

        if len(same_list) > min_image:
            dest = args.frame_file + '_split/' + date[5:7]+date[8:10] + '/' + str(day_count).zfill(3)
            makedirs(dest)
            day_count += 1
            for it in same_list:
                shutil.copy(os.path.join(args.frame_file+'/'+ items, it), os.path.join(dest, it))
            use_this_folder = 1
    return (use_this_folder, day_count)


makedirs(save_path)

# load info
c_image = []
c_folder= []
pair_list= []
count_p=0
count_n=0
count_n_r = 0


i_count = 0
m = os.listdir(args.frame_file)
m.sort()
for i, items in enumerate(m):
    File = args.frame_file + '/' + items
    csv_path = save_path+'csv/'+items+'/same.csv'
    date = items[:10]
    if date!=last_date:
        i_count = 0
    print(i_count)
    use, i_count = make_train_data(csv_path, items, i, i_count)
    # if use == 0:
    #     print(items)
    last_date = items[:10]

print('Done')
