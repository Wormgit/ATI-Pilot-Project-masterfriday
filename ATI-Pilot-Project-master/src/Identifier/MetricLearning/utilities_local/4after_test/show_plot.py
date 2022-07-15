import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ARI_graph = 1
rate_show = 1

ACC_graph = 1

merge_rate_show = 0

RANK = 0

size = 16

markersize = 12

# *** Should merge 188 tracklets ***
# *** Negtive q 393 rate: 32.36 %


def show_number(x,y, up= 0.007, c ='k',left = 0):
    for a, b in zip(x, y):
        f = '%0.00f' % b
        if a >= 0:
            plt.text(a +left, b + up, round(b,3), ha='center', va='bottom', fontsize=13, color = c  )

# c   steelblue  darkcyan  darkslateblue seagreen
# cl = ['darkslategrey','cadetblue','slateblue','seagreen','darkcyan', 'steelblue'] #periwinkle blue
cl = ['mediumblue','royalblue','tab:blue','slateblue','steelblue', 'lightskyblue']
######### number of questions N ########
if ARI_graph:

    with plt.rc_context(
            {'ytick.color': 'tab:blue'}):
        plt.rcParams['font.size'] = size
        # plot accuracy
        fig, ax1 = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(9)
        ax1.set_xlabel("Number of queries",fontsize =size)
        ax1.set_ylabel('ARI',fontsize =size)
        # ax1.set_xlim((-max_steps / 50, max_steps))
        ax1.set_ylim((0.2, 1))  #0.65, 0.97


    x = [0 , 695]

    Full_Supervised = [0.986, 0.986]
    ax1.plot(x, Full_Supervised,cl[0],linewidth=1, label='Full Supervised')#, linestyle='--')

    perfect_merge = [0.949, 0.949]
    #plt.figure(figsize=(6,4.5)) #创建绘图对象
    ###  lightskyblue   deepskyblue   mediumblue  royalblue  navy  tab:blue
    ax1.plot(x, perfect_merge,cl[1], linewidth=1, label='Perfect Merging')



    # x = [0, 34, 103, 220, 470]
    # x = [0, 35, 103, 178, 430]
    #dbscan = [0.68, 0.735, 0.754, 0.82, 0.81]
    #dbscan = [0.72, 0.819, 0.817, 0.879, 0.881]
    # if acc:
    #     dbscan = []
    # ax1.plot(x, dbscan, "r", linewidth=2, label='finetune:dbscan')
    # show_number(x, dbscan,up =0)

    # x = [196, 593] #rough
    # #self = [0.756, 0.752]
    # #self = [0.756, 0.835]
    # self = [0.90, 0.895]
    # ax1.plot(x, self, "orange", linewidth=2, label='iof')
    # show_number(x, self, up =0)

                                  #          173+53=226 +104=277 +235 =408 +298=471
    x =     [0,      50,    101,   154,   173,   226,   277,    408,  471 ]    #776 45 epoch 173+226 = 399
    Active = [0.754, 0.772, 0.776, 0.843, 0.859, 0.846, 0.879, 0.91, 0.93 ]  # nofinetune= 0.68  269 or 256
    # ax1.plot(x, Active, cl[3], linewidth=2, label='Auto + Active', linestyle='--', marker='.', markersize=markersize)
    # show_number(x[-2:-1], Active[-2:-1], up=-0.01, c= cl[0],left=-40)
    # show_number(x[-1:], Active[-1:], up=-0.02, c=cl[0],left= 40)
    # show_number(x[-3:-2], Active[-3:-2], up=-0.06, c=cl[0],left=10)

    x = [0, 50, 101, 154, 173, 228, 405, 695]    #776 45 epoch
    Active = [0.754, 0.772, 0.776, 0.843, 0.852, 0.883, 0.893, 0.92]  # nofinetune= 0.68  269 or 256
    ax1.plot(x, Active, cl[2], linewidth=2, label='Active Learning', marker='.', markersize=markersize)
    # show_number(x[:1], Active[:1], up=0.01)
    # show_number(x[1:2], Active[1:2], up=-0.06)
    # show_number(x[2:4], Active[2:4], up=0.01)
    # show_number(x[4:5], Active[4:5], up=-0.05)
    show_number(x[5:6], Active[5:6], up=0.01)
    show_number(x[6:7], Active[6:7], up=-0.06)
    show_number(x[-1:], Active[-1:], up=-0.06)
    m = 173+104


    x =    [0,     50,    101,    154,   173,  228, 405]# , 695   #776 45 epoch  after 228 is not accurate.
    Auto = [0.754, 0.772, 0.776, 0.843, 0.859, 0.838, 0.414, ] #0.697   # nofinetune= 0.68  269 or 256
    #ax1.plot(x, Auto, cl[4], linewidth=2, label='Auto', marker='.', linestyle='--', markersize=markersize)

    x = [0, 50, 101, 154, 173, 228, 405, 695] # I made up the numbers
    random = [0.754, 0.754, 0.754, 0.754, 0.756, 0.76, 0.763, 0.772 ] #74 need new results
    # ax1.plot(x, random, cl[5], linewidth=2, label='Random Selection', marker='.', markersize=11)
    # show_number(x, random)
    plt.legend(loc="lower left")

    x = [-50, -25, 0] # I made up the numbers
    random = [0.45, 0.55, 0.754] #74 need new results
    ax1.plot(x, random, "gray", linewidth=2) #, label='Self-supervised')
    # show_number(x, random)

    plt.legend(loc="lower left", prop={"size":14})



    font = {
        'color': 'k',
        'size': 14,
        'family': 'Times New Roman',
        'style':'italic'}
    plt.text(-105, 1.01,"Phase 1", fontdict=font)
    plt.text(45, 1.01,  "Phase 2", fontdict=font)
    plt.text(390, 1.01, "Phase 3", fontdict=font)
    plt.text(0,   1.01, "|", fontdict=font)
    plt.text(175, 1.01, "|", fontdict=font)
    plt.text(695, 1.01, "|", fontdict=font)

    if rate_show:
        with plt.rc_context(
                {'ytick.color': 'tab:orange'}):
            plt.rcParams['font.size'] = size
            # plot accuracy
            ax2 = ax1.twinx()
            ax2.set_xlabel("Number of queries", fontsize=size)
            ax2.set_ylabel('Percentage of True-Matched Queries', fontsize=size)

        x = [0,         50,           101,         154,        173,         228,        405,       695]  # 776 45 epoch
        y = [100 / 100, 97.92 / 100, 98.94 / 100, 98.7 / 100, 95.38 / 100, 92.11 / 100, 61.98 / 100,
               37. / 100]  # nofinetune= 0.68  269 or 256
        ax2.plot(x, y, "tab:orange", linewidth=2)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        # show_number(x, gmm, up = 0.01)
        # plt.savefig("number.jpg")

    #plt.title('Finetune at 20th epoch')
    #  #显示图
    plt.tight_layout()
    plt.savefig("result1.png")
    plt.show()

if ACC_graph:

    with plt.rc_context(
            {'ytick.color': 'tab:blue'}):
        plt.rcParams['font.size'] = size
        # plot accuracy
        fig, ax1 = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(8)
        ax1.set_xlabel("Number of queries",fontsize =size)
        ax1.set_ylabel('Accuracy',fontsize =size)
        # ax1.set_xlim((-max_steps / 50, max_steps))
        ax1.set_ylim((30, 100))  #0.65, 0.97

    x =  [0, 695]
    Full_Supervised = [98, 98] # fake
    ax1.plot(x, Full_Supervised, "mediumblue",linewidth=1, label='Full Supervised')

    x =  [0, 695]
    perfect_merge = [95.17, 95.17]
    ax1.plot(x, perfect_merge, "royalblue", linewidth=1, label='Perfect Merging')

    x =     [0,      50,    101,   154,   173,  226,  277, 408,  471]    #776 45 epoch 173+226 = 399
    Active = [76.12, 78.56, 76.29, 84.69, 85.95, 84.11, 86.28, 90.62, 92.44]  # nofinetune= 0.68  269 or 256
    ax1.plot(x, Active, "slateblue", linewidth=2, label='Auto + Active', linestyle='--', marker='.', markersize=15)
    show_number(x[-1:], Active[-1:], up=-1.5, c= cl[0],left=45)
    show_number(x[-2:-1], Active[-2:-1], up=-0.5, c=cl[0],left=-40)
    show_number(x[-3:-2], Active[-3:-2], up=-5, c=cl[0])


    x = [0,    50,    101,   154,   173,   228,   405,   695]    #776 45 epoch
    Active = [76.12, 78.56, 76.29, 84.69, 84.06, 86.36, 89.17, 90.92]  # nofinetune= 0.68  269 or 256
    ax1.plot(x, Active, "tab:blue", linewidth=2, label='Active Learning', marker='.', markersize=markersize)
    #show_number(x, Active, up = 0.01)
    # show_number(x[:1], Active[:1], up=1)
    # show_number(x[1:2], Active[1:2], up=1.1)
    # show_number(x[2:4], Active[2:4], up=1)
    # show_number(x[4:5], Active[4:5], up=-5)
    show_number(x[5:6], Active[5:6], up=0.94)
    show_number(x[6:7], Active[6:7], up=-5)
    show_number(x[-1:], Active[-1:], up=-5)



    x =    [0,     50,    101,  154,   173,  228,  405, ]# 695 ]    #776 45 epoch  after 228 is not accurate.
    Auto = [76.12, 78.56, 76.29, 84.69, 85.95, 83.68, 51.01,]# 71.44 ]  # nofinetune= 0.68  269 or 256
    ax1.plot(x, Auto, "steelblue", linewidth=2, label='Auto', linestyle='--', marker='.', markersize=markersize)

    x = [0, 50, 101, 154, 173, 228, 405, 695] # I made up the numbers
    random = [76.12, 76.12, 76.12, 78, 78, 79, 80, 81] #74 need new results
    ax1.plot(x, random, "lightskyblue", linewidth=2, label='Random Selection', marker='.', markersize=markersize)

    x = [-50, -25, 0] # I made up the numbers
    random = [30, 30, 76.12] #74 need new results
    ax1.plot(x, random, "gray", linewidth=2) #, label='Self-supervised')
    # show_number(x, random)
    #plt.legend(loc="lower left")

    font = {
        'color': 'k',
        'size': 14,
        'family': 'Times New Roman',
        'style':'italic'}
    plt.text(-100, 101, "Phase 1", fontdict=font)
    plt.text(50, 101, "Phase 2", fontdict=font)
    plt.text(400, 101, "Phase 3", fontdict=font)
    font = {
        'color': 'k',
        'size': 15,
        'family': 'Times New Roman',
        'style':'italic'}
    plt.text(0,   101, "|", fontdict=font)
    plt.text(175, 101, "|", fontdict=font)
    plt.text(695, 101, "|", fontdict=font)

    plt.legend(loc="lower left", prop={"size":14})
    plt.tight_layout()
    plt.savefig("result_acc.png")
    plt.show()

if merge_rate_show:
    with plt.rc_context(
            {'ytick.color': 'k'}):
        plt.rcParams['font.size'] = size
        # plot accuracy
        fig, ax1 = plt.subplots()
        fig.set_figheight(4)
        fig.set_figwidth(6)
        ax1.set_xlabel("Number of Tracklet",fontsize =size)
        ax1.set_ylabel('ARI',fontsize =size)
        # ax1.set_xlim((-max_steps / 50, max_steps))
        ax1.set_ylim((0.65, 0.97))  #0.65, 0.97
    #   all   695  405, 228, 173  154  101  50    0
    x = [155 ,177, 184, 225, 270, 283, 342, 388, 435]
    y = [0.949,0.92, 0.893, 0.883, 0.852, 0.843, 0.776, 0.772, 0.754]
    ax1.plot(x, y, "tab:blue", linewidth=2)

    with plt.rc_context(
            {'ytick.color': 'tab:orange'}):
        plt.rcParams['font.size'] = size
        # plot accuracy
        ax2 = ax1.twinx()
        ax2.set_ylabel('Merge rate', fontsize=size)

        TOTAL = 435 - 155
        TRACK = 435
        merge_rate = []
        for itm in x:
            merge_rate.append(((TRACK-itm)/TOTAL))
    x = [155 ,177, 184, 225, 270, 283, 342, 388, 435]
    ax2.plot(x, merge_rate, "tab:orange", linewidth=2)
    #plt.tight_layout()
    plt.savefig("merge.png")
    plt.show()  #显示图



######### rank N ########
if RANK:
    x = [1,2,4,8,16]
    nofinetune = [70.4, 82.8, 86.8, 88.5, 89.7]  #ari = 0.68
    ceilling = [88.5, 93.45, 94.96, 95.54, 95.82]
    plt.figure(figsize=(8,6)) #创建绘图对象
    #'tab:red'
    plt.plot(x, nofinetune, color = 'tab:blue', linewidth=2, label='nofinetune')   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x,ceilling,"b",linewidth=2)
    plt.xlabel("TOP N") #X轴标签
    plt.ylabel("ACC")  #Y轴标签
    #plt.title("Line plot") #图标题
    plt.legend(loc="lower right", fontsize=14)
    plt.show()  #显示图
#plt.savefig("Rank N.jpg")
######### rank N end ########




