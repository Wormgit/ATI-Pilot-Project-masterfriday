#!/usr/bin/env python
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."
import os, csv
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--csv_annotator", default='/home/io18230/Desktop/C_annotator.csv', type=str)
parser.add_argument("--csv_GMM", default='/home/io18230/Desktop/C_GMM_SELECT_DIS.csv', type=str)
parser.add_argument("--csv_inter", default='/home/io18230/Desktop/C_inter_distance.csv', type=str)
parser.add_argument("--img_path", default='/home/io18230/Desktop/RGB (copy)', type=str)
parser.add_argument("--out_path", default='/home/io18230/Desktop/', type=str)
parser.add_argument("--folder_last", default='001/0', type=str)

args = parser.parse_args()


def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise



def inter(args):
    csvFile = csv.reader(open(args.csv_inter, 'r'), delimiter=',')
    reader = list(csvFile)
    del reader[0]
    int_distance = []
    int_forder1 = []
    int_forder2 = []
    int_l1 = []
    int_l2 = []

    for i in range(len(reader)):
        int_distance.append(reader[i][1])
        int_forder1.append(reader[i][2])
        int_forder2.append(reader[i][3])
        int_l1.append(reader[i][4])
        int_l2.append(reader[i][5])

    print(f'Original length:', len(int_l1))

    gmm_count_raw_yes=0
    gmm_count_raw=0

    f1new = []
    f2new = []
    l1new = []
    l2new = []
    dnew = []

    for i,(l1,l2,f1,f2) in enumerate(zip(int_l1,int_l2,int_forder1,int_forder2)):
        skip_mark = 0
        if f1 in f1new:
            index = [i for i, val in enumerate(f1new) if val == f1]
            for item in index:
                if f2 == f2new[item]:
                    skip_mark = 1
        if f1 in f2new:
            index = [i for i,val in enumerate(f2new) if val==f1]
            for item in index:
                if f2 == f1new[item]:
                    skip_mark = 1
        if not skip_mark:
            f1new.append(f1)
            f2new.append(f2)
            l1new.append(l1)
            l2new.append(l2)
            dnew.append(int_distance[i])

    for i,(l1,l2) in enumerate(zip(l1new,l2new)):
        if l1 == l2:
            gmm_count_raw_yes +=1
        gmm_count_raw += 1
    print(f'The number of inter Q', gmm_count_raw, f', True matches pairs', gmm_count_raw_yes, f', rate', gmm_count_raw_yes / gmm_count_raw)

    data2 = pd.DataFrame(
        {'Distance': dnew, 'folder1': f1new, 'folder2': f2new, 'label1': l1new, 'label2': l2new})
    data2.to_csv(os.path.join(args.out_path, 'C_inter_distance_filter.csv'))


def gmm_filter_multiple(args):
    csvFile = csv.reader(open(args.csv_GMM, 'r'), delimiter=',')
    reader = list(csvFile)
    del reader[0]
    gmm_distance = []
    gmm_forder1 = []
    gmm_forder2 = []
    gmm_l1 = []
    gmm_l2 = []
    gmm_pri = []

    for i in range(len(reader)):
        gmm_distance.append(reader[i][1])
        gmm_forder1.append(reader[i][2])
        gmm_forder2.append(reader[i][3])
        gmm_l1.append(reader[i][4])
        gmm_l2.append(reader[i][5])
        gmm_pri.append(reader[i][6])
    print(f'Original length:', len(gmm_l2))


    done = []
    count_n = 0
    count_2 = 0
    count_total = 0
    count_wrong = 0
    cn = 0

    aaa_distance = []
    aaa_forder1 = []
    aaa_forder2 = []
    aaa_l1 = []
    aaa_l2 = []
    aaa_pri= []

    for i,(l1,l2,f1,f2) in enumerate(zip(gmm_l1,gmm_l2,gmm_forder1,gmm_forder2)):

        # exclude rare
        if f1 not in f2 and float(gmm_distance[i])>8:# just abandon them
            if int(gmm_l1[i]) - int(gmm_l2[i])==0:
                print( f1, f'distance', gmm_distance[i], gmm_pri[i], int(gmm_l1[i]) - int(gmm_l2[i]))
                cn +=1
            count_n +=1
        else:
            aaa_distance.append(gmm_distance[i])
            aaa_forder1.append(f1)
            aaa_forder2.append(f2)
            aaa_l1.append(l1)
            aaa_l2.append(l2)
            aaa_pri.append(gmm_pri[i])

        #in case something wrong
        if f1 in f2 and gmm_l1[i] != gmm_l2[i]: # yes the distance here is small # maybe we do not need human here.
            print( f1, f'distance', gmm_distance[i], gmm_pri[i], int(gmm_l1[i]) - int(gmm_l2[i]),'wrong22')
            count_2 +=1



    for i, (l1, l2, f1, f2) in enumerate(zip(aaa_l1, aaa_l2, aaa_forder1,aaa_forder2)):
        #exclude the big one in each tracklet.
        maxdistmp = 0
        if f1 not in done:
            index = [i for i, val in enumerate(aaa_forder1) if val == f1]
            if len(index)>1:
                for item in index:
                    if float(aaa_distance[item])>maxdistmp:
                        maxdistmp = float(aaa_distance[item])
                        choosen_distance_i = item

                #double check
                if aaa_l1[choosen_distance_i] == aaa_l2[choosen_distance_i]:
                    count_wrong +=1
                    print('folder', f1, f'distance', aaa_distance[choosen_distance_i], 'wrong')
                count_total +=1
            done.append(f1)
    print('exclude', count_n, 'and', cn, 'is wrong')
    if count_2 > 0:
        print('****')
    print('exclude', count_total, 'and', count_wrong, 'is wrong')


    #filter repeated
    gmm_distance = aaa_distance
    gmm_forder1 = aaa_forder1
    gmm_forder2 = aaa_forder2
    gmm_l1 = aaa_l1
    gmm_l2 = aaa_l2
    gmm_pri = aaa_pri
    print(f'Original length:', len(gmm_l2))

    # calculate the number of questions
    gmm_count_raw = 0
    gmm_count_raw_yes = 0
    f1new = []
    f2new = []
    l1new = []
    l2new = []
    dnew = []
    pnew = []

    # skip 重复的
    for i,(l1,l2,f1,f2) in enumerate(zip(gmm_l1,gmm_l2,gmm_forder1,gmm_forder2)):
        if f1 == f2:
            continue
        else:
            skip_mark = 0
            if f1 in f1new:
                index = [i for i,val in enumerate(f1new) if val==f1]
                for item in index:
                    if f2 == f2new[item]:
                        skip_mark = 1
                        #print(f'skip it')
            if f1 in f2new:
                index = [i for i,val in enumerate(f2new) if val==f1]
                for item in index:
                    if f2 == f1new[item]:
                        skip_mark = 1
                        #print(f'skip it')
            if not skip_mark:
                f1new.append(f1)
                f2new.append(f2)
                l1new.append(l1)
                l2new.append(l2)
                dnew.append(gmm_distance[i])
                pnew.append(gmm_pri[i])

    for i, (l1, l2) in enumerate(zip(l1new, l2new)):
        if l1 == l2:
            gmm_count_raw_yes += 1
        gmm_count_raw += 1
    print(f'The number of GMM Q', gmm_count_raw, f', True matches pairs', gmm_count_raw_yes, f', rate:', gmm_count_raw_yes / gmm_count_raw)
    data2 = pd.DataFrame(
        {'Distance': dnew, 'folder1': f1new, 'folder2': f2new, 'label1': l1new, 'label2': l2new, 'priority': pnew})
    data2.to_csv(os.path.join(args.out_path, 'C_gmm_distance_filter.csv'))


if __name__ == '__main__':
    gmm_filter_multiple(args)

    #inter(args)