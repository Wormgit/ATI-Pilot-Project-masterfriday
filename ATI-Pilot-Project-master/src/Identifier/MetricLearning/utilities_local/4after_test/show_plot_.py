
import numpy as np
import matplotlib.pyplot as plt

RANK = 0
NUMBER = 1


# *** Should merge 188 tracklets ***
# *** Negtive q 393 rate: 32.36 %


def show_number(x,y, up= 0.007):
    for a, b in zip(x, y):
        f = '%0.00f' % b
        if a > 0:
            plt.text(a, b + up, round(b,3), ha='center', va='bottom', fontsize=12)

######### number of questions N ########
if NUMBER:

    with plt.rc_context(
            {'ytick.color': 'k'}):
        plt.rcParams['font.size'] = 11
        # plot accuracy
        fig, ax1 = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(7)
        ax1.set_xlabel("Number of Questions",fontsize =11)
        ax1.set_ylabel('ARI',fontsize =11)
        # ax1.set_xlim((-max_steps / 50, max_steps))
        ax1.set_ylim((0.65, 0.96))
    x = [0 , 21, 101, 269, 581]
    ceilling_merge = [0.929, 0.929, 0.929, 0.929, 0.929]


    #plt.figure(figsize=(6,4.5)) #创建绘图对象
    ###  lightskyblue   deepskyblue   mediumblue  royalblue  navy  tab:blue
    ax1.plot(x, ceilling_merge,"royalblue",linewidth=1, label='ideal', linestyle='--')




    x = [0, 34, 103, 220, 470]
    x = [0, 35, 103, 178, 430]
    #dbscan = [0.68, 0.735, 0.754, 0.82, 0.81]
    dbscan = [0.72, 0.819, 0.817, 0.879, 0.881]
    # if acc:
    #     dbscan = []
    ax1.plot(x, dbscan, "r", linewidth=2, label='finetune:dbscan')
    show_number(x, dbscan,up =0)

    # x = [196, 593] #rough
    # #self = [0.756, 0.752]
    # #self = [0.756, 0.835]
    # self = [0.90, 0.895]
    # ax1.plot(x, self, "orange", linewidth=2, label='iof')
    # show_number(x, self, up =0)

    x = [0, 34, 103, 220, 470]
    x = [0, 21, 101, 255, 581]
    #gmm = [0.68, 0.68, 0.73, 0.79, 0.81]
    gmm = [0.72, 0.72, 0.829, 0.883, 0.882]  # nofinetune= 0.68  269 or 256
    ax1.plot(x, gmm, "tab:blue", linewidth=2, label='finetune:violent')
    show_number(x,gmm, up = 0)

    x = [0, 21, 101, 255, 581]
    random = [0.72, 0.72, 0.72, 0.72, 0.729] #74 need new results
    ax1.plot(x, random, "lightskyblue", linewidth=2, label='finetune:Random')
    show_number(x, random)

    plt.legend(loc="lower right")

    plt.show()  #显示图
    #plt.savefig("number.jpg")
######### rank N end ########


#### merging ratio ###



#### merging ratio end ###





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
    plt.legend(loc="lower right")
    plt.show()  #显示图
#plt.savefig("Rank N.jpg")
######### rank N end ########




