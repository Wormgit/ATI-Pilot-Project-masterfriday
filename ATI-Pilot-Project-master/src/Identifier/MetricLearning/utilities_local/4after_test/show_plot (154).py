
import numpy as np, csv,argparse
import matplotlib.pyplot as plt

munite = 1
NUMBER = 0
top_n = 1


# *** Should merge 188 tracklets ***
# *** Negtive q 393 rate: 32.36 %
parser = argparse.ArgumentParser()
parser.add_argument("--csv", default='/home/io18230/Desktop/for_plot.csv', type=str)
args = parser.parse_args()




def show_number(x,y, up= 0.007):
    for a, b in zip(x, y):
        f = '%0.00f' % b
        if a > 0:
            plt.text(a, b + up, round(b,3), ha='center', va='bottom', fontsize=12)

######### number of questions N ########

size = 20
size2 = 20

if munite:
    csvFile = csv.reader(open(args.csv, 'r'), delimiter=',')
    reader = list(csvFile)
    del reader[0]
    error = []
    dis = []

    for i in range(len(reader)):
        if reader[i][1] != '':
            error.append(reader[i][9])
            dis.append(float(reader[i][1]))

    error_sum = 0

    error_rate = []
    for count_1, item in enumerate(error):
        if item == '-1':
            cont = 1
        else:
            cont = 0
        error_sum = cont + error_sum
        rate = error_sum / (count_1 + 1)
        error_rate.append(rate)
        m = 1

    with plt.rc_context(
            {'ytick.color': 'tab:blue'}):
        plt.rcParams['font.size'] = size2

        # plot accuracy
        fig, ax1 = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(8)
        ax1.set_xlabel("Query",fontsize =size2)
        ax1.set_ylabel('Percentage of false-matched pairs',fontsize =size2)
        #ax1.set_ylim((0, 60))
        ax1.plot(error_rate, "tab:blue", linewidth=4)
        from matplotlib.ticker import FuncFormatter
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda error_rate, _: '{:.0%}'.format(error_rate)))

    with plt.rc_context(
            {'ytick.color': 'tab:orange'}):
        plt.rcParams['font.size'] = size2
        # plot accuracy
        ax2 = ax1.twinx()
        # ax2.set_ylim((0, 1))
        ax2.set_ylabel('Tracklet-to-tracklet distance', fontsize=size2)
        # percentage = [1, 5, 18, 54, 100, 82, 60, 72, 49, 36]
        ax2.plot(dis, "tab:orange", linewidth=4)

    plt.tight_layout()
    plt.savefig("error.png")
    plt.show()  # 显示图

if NUMBER:

    with plt.rc_context(
            {'ytick.color': 'k'}):
        plt.rcParams['font.size'] = size

        # plot accuracy
        fig, ax1 = plt.subplots()
        fig.set_figheight(4)
        fig.set_figwidth(6)
        ax1.set_xlabel("Number of tracklets per individual",fontsize =size)
        ax1.set_ylabel('Number of individuals',fontsize =size)
        # ax1.set_xlim((-max_steps / 50, max_steps))
        #ax1.set_ylim((0.60, 0.88))

    #plt.figure(figsize=(6,4.5)) #创建绘图对象
    ###  lightskyblue   deepskyblue   mediumblue  royalblue  navy  tab:blue
    # ax1.plot(x, ceilling_merge,"royalblue",linewidth=1, label='ideal', linestyle='--')

    x =[1,2,3,4,5,6,7]
    data = [32, 36, 49, 20, 13, 1, 3 ]
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    plt.bar(x, data)
    plt.tight_layout()
    plt.savefig("train_distribution.png")
    plt.show()  # 显示图






size2= 20
if NUMBER:

    with plt.rc_context(
            {'ytick.color': 'tab:blue'}):
        plt.rcParams['font.size'] = size2

        # plot accuracy
        fig, ax1 = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(8)
        ax1.set_xlabel("Distance between tracklets",fontsize =size2)
        ax1.set_ylabel('Number of Tracklet Pairs',fontsize =size2)
    x =[1,2,3,4,5,6,7,8,9,10]
    data = [153, 214, 176, 129, 139, 101, 67, 76, 53, 36]
    data = [66, 123, 107, 99, 120, 90, 61, 72, 49, 35]

    plt.xticks(fontsize=size2)
    plt.yticks(fontsize=size2)
    plt.bar(x, data,width=0.6)

    with plt.rc_context(
            {'ytick.color': 'tab:orange'}):
        plt.rcParams['font.size'] = size2
        # plot accuracy
        ax2 = ax1.twinx()
        ax2.set_ylim((0, 1))
        ax2.set_ylabel('Percentage of True-Matched Pairs', fontsize=17)
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # percentage = [1, 5, 18, 54, 100, 82, 60, 72, 49, 36]
        percentage =[(153-1)/153, (214-5)/214, (176-18)/176, (129-54)/129, (139-100)/139,
                     (101-82)/101, (67-60)/67, (76-72)/76, (53-49)/53, (36-36)/36,]
        percentage = [(66 - 1) / 66, (123 - 4) / 123, (107 - 13) / 107, (99 - 47) / 99, (120 - 89) / 120,
                      (90 - 77) / 90, (61 - 56) / 61, (72 - 68) / 72, (49 - 47) / 49, (35 - 35) / 35]
        ax2.plot(x, percentage, "tab:orange", linewidth=4)
    plt.tight_layout()
    plt.savefig("distance_distribution.png")
    plt.show()  # 显示图


size2= 20
markersize = 12
if top_n:

    with plt.rc_context(
            {'ytick.color': 'k'}):
        plt.rcParams['font.size'] = size2

        # plot accuracy
        fig, ax1 = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(8)
        ax1.set_xlim((-4, 500))
        ax1.set_ylim((58, 100))
        ax1.set_xlabel("Number of queries",fontsize =size2)
        ax1.set_ylabel('Accuracy',fontsize =size2)

        ax1.axvspan(-4, 173, facecolor='green', alpha=0.3)
        ax1.axvspan(173, 500, facecolor='orange', alpha=0.3)
                                  #          173+53=226 +104=277 +235 =408 +298=471
    x =     [0,      50,    101,   154,   173,   226,   277,    408,  471 ]    #776 45 epoch 173+226 = 399

    #open24=   [60, 70.44,    73.19, 77.43, 82.94,  83.22, 85.54 ,85.72, 88.4 ] #177
    open24 = [68.83, 78.16 , 78.77, 87.05,    81.78,   84.94,   79.37,   86.90,  90.81]

    top1 =  [76.12, 78.56, 76.29, 84.69, 85.95, 84.11, 86.28, 90.62, 92.44]
    top2 =  [87.13,87.59, 86.99,90.85 ,91.39 , 91.9,92.05,94.59 ,95.92]
    top4 =  [89.86, 90.21,88.87,92.50,92.35,92.68,92.85,94.91 ,96.2]
    top8 = [90.92, 91.04 ,89.95,93.26,92.66,93.23,93.39 ,95.08,96.41]
    top16= [91.56 ,91.47,90.53,93.90,93.0,93.51,93.81,95.23 ,96.59]

    font = {
        'color': 'k',
        'size': 18,
        'family': 'Times New Roman',
        'style':'italic'}
    plt.text(46,  100, "Phase #2", fontdict=font)
    plt.text(300, 100, "Phase #3", fontdict=font)

    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(0)
    ax1.spines['top'].set_linewidth(0)

    ax1.plot(x, top16, "y", linewidth=2, label='top 16', marker='.',markersize =markersize, linestyle='--')
    ax1.plot(x, top8, "k", linewidth=2, label='top 8',  marker='.',markersize =markersize,  linestyle='--')
    ax1.plot(x, top4, "r", linewidth=2, label='top 4',  marker='.',markersize =markersize,  linestyle='--')
    ax1.plot(x, top2, "g", linewidth=2, label='top 2',  marker='.',markersize =markersize,  linestyle='--')
    ax1.plot(x, top1, "b", linewidth=2, label='top 1',  marker='.',markersize =markersize,  linestyle='--')
    ax1.plot(x, open24, "orange", linewidth=2, label='open top 1', marker='.', markersize=markersize)

    plt.xticks(fontsize=size2)
    plt.yticks(fontsize=size2)

    plt.legend(loc="lower right", prop={"size": 16})
    plt.tight_layout()
    plt.savefig("result_topn.png")
    plt.show()  # 显示图






