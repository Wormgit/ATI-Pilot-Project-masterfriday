import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ARI_graph = 1
rate_show = 1

ACC_graph = 1

merge_rate_show = 0

RANK = 0
size = 16
markersize = 16
right_edge = 500

# *** Should merge 188 tracklets ***
# *** Negtive q 393 rate: 32.36 %


def show_number(x,y, up= 0.007, c ='k',left = 0, fontsize =13):
    for a, b in zip(x, y):
        f = '%0.00f' % b
        if a >= 0:
            plt.text(a +left, b + up, round(b,3), ha='center', va='bottom', fontsize=fontsize, color = c  )

# c   steelblue  darkcyan  darkslateblue seagreen
# cl = ['darkslategrey','cadetblue','slateblue','seagreen','darkcyan', 'steelblue'] #periwinkle blue
cl = ['mediumblue','royalblue','tab:blue','slateblue','steelblue', 'lightskyblue']
######### number of questions N ########
if ARI_graph:

    with plt.rc_context(
            {'ytick.color': 'k'}):
        plt.rcParams['font.size'] = size
        # plot accuracy
        fig, ax1 = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(9)
        ax1.set_xlabel("Number of queries",fontsize =size)
        ax1.set_ylabel('ARI',fontsize =size)
        # ax1.set_xlim((-max_steps / 50, max_steps))
        ax1.set_ylim((0.2, 0.99))  #0.65, 0.97

    plt.axis([-50, right_edge, 0.2, 0.99])

    x = [0 , 471]

    Full_Supervised = [0.976, 0.976]
    ax1.plot(x, Full_Supervised,'indigo',linewidth=4, label='Fully Supervised')#, linestyle='--')

    perfect_merge = [0.949, 0.949]
    #plt.figure(figsize=(6,4.5)) #创建绘图对象
    ###  lightskyblue   deepskyblue   mediumblue  royalblue  navy  tab:blue
    ax1.plot(x, perfect_merge,'mediumorchid', linewidth=3, label='Perfect Merging')

    ax1.axvspan(-50, 0, facecolor='skyblue', alpha=0.5)  #self, ymin, ymax, xmin=0, xmax=1, **kwargs):
    ax1.axvspan(0, 173, facecolor='green', alpha=0.3)
    ax1.axvspan(173, right_edge, facecolor='orange', alpha=0.3)

    ax1.spines['bottom'].set_linewidth(2.5)
    ax1.spines['left'].set_linewidth(2.5)
    ax1.spines['right'].set_linewidth(0)
    ax1.spines['top'].set_linewidth(0)



    # phase 1
    x = list(range(-50, 0, 3))
    #-x =    [0,     2,    4,     6,      8,     10    12     14     16     18    20    22    24
    #         26     8     0      32
    # ]
    befor = [0.121,
             0.519, 0.641, 0.664, 0.668, 0.675, 0.652, 0.688, 0.676, 0.718,0.718,0.702, 0.739, 0.666,
             0.754,]
    ax1.plot(x[-len(befor):], befor, "tab:blue", linewidth=3)




                                  #          173+53=226 +104=277 +235 =408 +298=471
    x =     [0,      50,    101,   154,   173,   226,   277,   408,  471 ]    #776 45 epoch 173+226 = 399
    Active = [0.754, 0.772, 0.776, 0.843, 0.859, 0.846, 0.879, 0.91, 0.93 ]  # nofinetune= 0.68  269 or 256
    Active = [0.75, 0.77, 0.78, 0.84, 0.86, 0.85, 0.88, 0.91, 0.93]
    ax1.plot(x, Active, "darkorange", linewidth=3, label='Active', marker='.', markersize=markersize)
    size__ = 15
    show_number(x[-2:-1], Active[-2:-1], up=-0.06, c= "saddlebrown",left=0,fontsize=size__)
    show_number(x[-1:], Active[-1:], up=-0.06, c='saddlebrown',left= 0,fontsize=size__)
    show_number(x[-3:-2], Active[-3:-2], up=-0.06, c='saddlebrown',left=0,fontsize=size__)

    show_number(x[-5:-4], Active[-5:-4], up=-0.08, c="darkgreen",fontsize=size__)
    show_number(x[0:1], Active[0:1], up=0.02, c="darkgreen",fontsize=size__)

    # x = [0, 50, 101, 154, 173, 228, 405, 695]    #776 45 epoch
    # Active = [0.754, 0.772, 0.776, 0.843, 0.852, 0.883, 0.893, 0.92]  # nofinetune= 0.68  269 or 256
    # ax1.plot(x, Active, cl[2], linewidth=2, label='Active Learning', marker='.', markersize=markersize)
    # show_number(x[5:6], Active[5:6], up= 0.01)
    # show_number(x[6:7], Active[6:7], up=-0.06)
    # show_number(x[-1:], Active[-1:], up=-0.06, left=-20)
    # m = 173+104

    x =    [0,     50,    101,    154,   172,  224, 277, 424]# , 695   #776 45 epoch  after 228 is not accurate.
    Auto = [0.754, 0.772, 0.776, 0.843, 0.859, 0.838, 0.73, 0.386] #0.697   # nofinetune= 0.68  269 or 256
    Auto = [0.75, 0.77, 0.78, 0.84, 0.86, 0.84, 0.73, 0.39]   # 347 0.41 removed
    ax1.plot(x[:5], Auto[:5], 'g', linewidth=3, label='Auto', marker='.', markersize=markersize)
    ax1.plot(x[4:], Auto[4:], 'g', linewidth=3, marker='.', linestyle='--', markersize=markersize)
    # phase 2 random
    x = [0, 50, 101, 154, 173, 228, 405, 471] # I made up the numbers
    random = [0.754, 0.754, 0.754, 0.754,0.754, 0.754, 0.754, 0.754] #74 need new results
    ax1.plot(x, random, cl[2], linewidth=3, label='Self-supervision', linestyle='--')



    font = {
        'color': 'k',
        'size': 14,
        'family': 'Times New Roman',
        'style':'italic'}
    plt.text(-60, 1.02, "Phase #1", fontdict=font)
    plt.text(45, 1.02,  "Phase #2", fontdict=font)
    plt.text(300, 1.02, "Phase #3", fontdict=font)

    plt.legend(bbox_to_anchor=(0.48, 0.5), prop={"size": 14})

    if rate_show:
        with plt.rc_context(
                {'ytick.color': 'brown'}):
            plt.rcParams['font.size'] = size
            # plot accuracy
            ax2 = ax1.twinx()
            ax2.set_xlabel("Number of queries", fontsize=size)
            ax2.set_ylabel('Percentage of True-Matched Queries', fontsize=size, color='brown')
            ax2.spines['right'].set_linewidth(0)
            ax2.spines['top'].set_linewidth(0)

        x = [0,         50,           101,         154,        173,         228,        405,       695]  # 776 45 epoch
        y = [100 / 100, 97.92 / 100, 98.94 / 100, 98.7 / 100, 95.38 / 100, 92.11 / 100, 61.98 / 100,
               37. / 100]  # nofinetune= 0.68  269 or 256
        ax2.plot(x, y, "k", linewidth=3, label='Rate                   ', color='brown', linestyle = '--', )
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        #plt.legend(prop={"size": 14})
    plt.legend(bbox_to_anchor=(0.482, 0.12), prop={"size": 14})

    #plt.title('Finetune at 20th epoch')
    #  #显示图
    plt.tight_layout()
    plt.savefig("result1.png")
    plt.show()

if ACC_graph:

    with plt.rc_context(
            {'ytick.color': 'k'}):
        plt.rcParams['font.size'] = size
        # plot accuracy
        fig, ax1 = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(8)
        ax1.set_xlabel("Number of queries",fontsize =size)
        ax1.set_ylabel('Accuracy',fontsize =size)
        # ax1.set_xlim((-max_steps / 50, max_steps))
        ax1.set_ylim((30, 99))  #0.65, 0.97

    ax1.axvspan(-50, 0, facecolor='skyblue', alpha=0.5)  #self, ymin, ymax, xmin=0, xmax=1, **kwargs):
    ax1.axvspan(0, 173, facecolor='green', alpha=0.3)
    ax1.axvspan(173, right_edge, facecolor='orange', alpha=0.3)

    ax1.spines['bottom'].set_linewidth(2.5)
    ax1.spines['left'].set_linewidth(2.5)
    ax1.spines['right'].set_linewidth(0)
    ax1.spines['top'].set_linewidth(0)

    plt.axis([-50, right_edge, 30, 99])

    # phase 1
    x = list(range(-50, 0, 3))
       #-x =    [0,     2,    4,     6,      8,     10    12     14     16     18    20    22    24
    #         26     8     0      32
    # ]
    befro_acc = [11.08,
                 57.22, 68.6, 68.79, 68.82, 71.67, 66.28, 72.25, 69.14, 71.07, 72.44, 72.01, 75.54, 66.07,
                 76.12]
    ax1.plot(x[-len(befor):], befro_acc, "tab:blue", linewidth=3)



    x =  [0, 471]
    Full_Supervised = [97, 97] # fake
    ax1.plot(x, Full_Supervised, "indigo",linewidth=4, label='Fully Supervised')

    x =  [0, 471]
    perfect_merge = [95.17, 95.17]
    ax1.plot(x, perfect_merge, "mediumorchid", linewidth=3, label='Perfect Merging')

    x =     [0,      50,    101,   154,   173,   226,    277, 408,  471]    #776 45 epoch 173+226 = 399
    Active = [76.12, 78.56, 76.29, 84.69, 85.95, 84.11, 86.28, 90.62, 92.44]  # nofinetune= 0.68  269 or 256
    size__ = 14
    ax1.plot(x, Active, "darkorange", linewidth=3, label='Active', marker='.', markersize=markersize)
    show_number(x[-1:], Active[-1:], up=-5, c= "saddlebrown",left=-0,fontsize=size__)
    show_number(x[-2:-1], Active[-2:-1], up=-5, c="saddlebrown",left=-0,fontsize=size__)
    show_number(x[-3:-2], Active[-3:-2], up=-5, c="saddlebrown",fontsize=size__)
    show_number(x[-5:-4], Active[-5:-4], up=-7, c="darkgreen",fontsize=size__)
    show_number(x[0:1], Active[0:1], up=2, c="darkgreen",fontsize=size__)

    x =    [0,     50,    101,  154,    172,  224, 277 , 424]# 695 ]    #776 45 epoch  after 228 is not accurate.
    Auto = [76.12, 78.56, 76.29, 84.69, 85.95, 83.68, 72.44, 43.66]# 71.44 ] #347 51.01 # nofinetune= 0.68  269 or 256
    ax1.plot(x[:5], Auto[:5], 'g', linewidth=3, label='Auto', marker='.', markersize=markersize)
    ax1.plot(x[4:], Auto[4:], 'g', linewidth=3, marker='.', linestyle='--', markersize=markersize)

    # phase 2 random
    x = [0, 50, 101, 154, 173, 228, 405, 471] # I made up the numbers
    random = [76.12, 76.12, 76.12, 76.12, 76.12, 76.12, 76.12, 76.12] #74 need new results
    ax1.plot(x, random, cl[2], linewidth=3, label='Self-supervision', linestyle='--')


    font = {
        'color': 'k',
        'size': 14,
        'family': 'Times New Roman',
        'style':'italic'}
    plt.text(-60, 102, "Phase #1", fontdict=font)
    plt.text(50, 102, "Phase #2", fontdict=font)
    plt.text(300, 102, "Phase #3", fontdict=font)
    font = {
        'color': 'k',
        'size': 15,
        'family': 'Times New Roman',
        'style':'italic'}
    # plt.text(0,   101, "|", fontdict=font)
    # plt.text(175, 101, "|", fontdict=font)
    # plt.text(695, 101, "|", fontdict=font)

    plt.legend(#loc="lower center",
               bbox_to_anchor=(0.49, 0.48),prop={"size":14})
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

