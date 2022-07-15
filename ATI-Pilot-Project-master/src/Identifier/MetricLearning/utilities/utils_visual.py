# Core libraries
import os,math, random
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score

# Local libraries
from utils_visual_acc_roud import FristRd, SecondRd, printtopN


# toplabels, topNbyPoints, makedirs, count_if_enough, del_first, ACC,  draw_ellipse
# about k means: expand, getEuclidean, k_means, plot_cluster_no_label,

def toplabels(top_label = 0, t_p = 0, n = 8):
    count = 0
    while count < n:
        for i, item in enumerate(top_label):
            t_p[i][item] = float('-inf')
        top_next_label = np.argmax(t_p, axis=1)
        top_label = np.column_stack((top_label, top_next_label))
        count += 1
    return top_label

def topNbyPoints(label_input, pre_label_gmm_feedback, label_topN, All_pair, n, cal_recall = 0):

    '''
    ACC AND RECALL
    '''

    # Accuracy
    colle_acc = []
    cluster = []
    plot_accum = []
    count = 1
    Top_l = pre_label_gmm_feedback
    while count < n:
        for m2label in label_topN[:, count]:
            idx = np.where(m2label == All_pair[:, 0])
            if len(idx[0]) > 0:
                assert len(idx[0]) <= 1
                cluster.append(int(All_pair[idx, 1]))
            else:   # if did not find labels
                cluster.append(-2)
        Top_l = np.vstack((Top_l, cluster))
        count +=1
        cluster = []

        # calculate top N accuracy
        plot_single = []
        c_acc = 0
        for i in range(len(label_input)):
            gt = int(label_input[i])
            if gt in Top_l[:, i]: #find a label in top n
                c_acc += 1
                plot_single.append(0)
            else:                   #did not find. mark as 1
                plot_single.append(1)
        colle_acc.append(round(c_acc / len(label_input) * 100, 2))
        plot_accum.append(plot_single)

    # calculate top N recall,f1,precision
    result_prf = []
    if cal_recall:
        del_ = 0
        result_prf = np.zeros((n,3))
        for j in range(0,n):
            weighted_prf = []
            for i in np.unique(label_input):
                label_mask = (label_input == i)  # for gt
                label_indices = np.where(label_mask)[0]
                candidate = Top_l[:j+1, label_indices]
                TP = 0
                FP = 0
                gt = len(label_indices)
                for k in range(gt):
                    if i in candidate[:, k]:
                        TP += 1
                FN = gt - TP
                label_mask = 1 - label_mask # the rest
                label_indices = np.where(label_mask)[0]
                expt_gt = len(label_indices)
                candidate_expt_gt = Top_l[:j + 1, label_indices]
                for k in range(expt_gt):
                    if i in candidate_expt_gt[:, k]:
                        FP += 1
                ALL = len(label_input)
                #TN = ALL-FN-FP-TP

                if FP == 0 and TP==0:
                    recall = 0
                else:
                    recall = TP / (TP + FN)

                if TP == 0 and FP==0:
                    precision = 0
                else:
                    precision = TP / (TP + FP)

                if precision == 0 and recall==0:
                    F1 = 0
                else:
                    F1 = 2 * precision * recall / (precision + recall)

                weight = gt/ALL
                del_ += weight
                weighted_prf.append([precision*weight,recall*weight,F1*weight])
            weighted_prf = np.array(weighted_prf)
            result_prf[j] = np.round(np.sum(weighted_prf,axis=0),2)

    return colle_acc, np.array(plot_accum), result_prf


def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def count_if_enough(top_2_highlight, args):
    #count if there is enough spots with same high likely hodd labels ----log in save_calculated
    b = []
    save_calculated = []
    pb = []
    for i, item in enumerate(top_2_highlight):
        b.append(item[0] + item[1])
        tm0 = item[0]
        tm1 = item[1]
        count = 0
        if item not in save_calculated:
            for k, t in enumerate(top_2_highlight):
                if tm0 == t[1]:
                    if tm1 == t[0]:
                        count += 1
                if item == t:
                    count += 1
            if count >= args.lowRandNumber:
                save_calculated.append(item)
                save_calculated.append([item[1], item[0]])
                pb.append(count / len(top_2_highlight))
                pb.append(count / len(top_2_highlight))
    return save_calculated


def accBasedCluster(top_2_exchange_label, All_pair, label_input, pre_label_gmm_feedback):
    lover_idex = []
    SaveEnough = top_2_exchange_label
    first_SE = []  # get initial number for convinence.
    for tmtmt in SaveEnough:
        first_SE.append(tmtmt[0])

    f_tmp = []
    tmpAppend = []
    for item in All_pair:
        gt_l = item[1]
        gmm_l = item[0]
        if int(gmm_l) in first_SE:
            for i, pp in enumerate(SaveEnough):
                if pp[0] == gmm_l:
                    if len(SaveEnough[i]) == 2:
                        SaveEnough[i] = [pp[0], pp[1], gt_l]  # GMM ORIGINAL, SECOND CONFIDENT, GT OF FIRST
                        f_tmp.append(gt_l)
                    else:
                        tmpAppend.append([pp[0], pp[1], gt_l])
                        f_tmp.append(gt_l)
    save_enough = SaveEnough + tmpAppend  # now if 3 in []: top1 original gmm top2 original gmm, gmm cluster label of top1

    collect_exchange = []
    for i, pp in enumerate(save_enough):
        if len(save_enough[i]) == 3:
            collect_exchange.append(save_enough[i])

    for i, pp in enumerate(collect_exchange):
        SECOND_gmm = pp[1]
        for item in All_pair:
            gmm_l = item[0]
            gt_l = item[1]
            if SECOND_gmm == gmm_l:
                collect_exchange[i] = [*pp, gt_l]  # pp[0], pp[1], gt_l]  # GMM ORIGINAL, SECOND CONFIDENT, GT OF FIRST
                # f_tmp.append(gt_l)

    ####accuracy
    c_acc = 0
    count_black = 0
    tmppp = []
    for i in collect_exchange:  # find the reason why length is 3 (because some did not find label)
        if len(i) == 4:
            tmppp.append(i)
    collect_exchange = tmppp

    for i in range(len(label_input)):
        A_gt = label_input[i]
        B_cluster = pre_label_gmm_feedback[i]

        if A_gt == B_cluster:
            c_acc += 1
        elif B_cluster in f_tmp:  # Bbb in f_tmp:   # judge if Bbb is the sesond choice
            for item in collect_exchange:
                if B_cluster == item[2] and A_gt == item[3]:
                    c_acc += 1
                    lover_idex.append(i)
        if B_cluster < 0:  # double check
            count_black += 1
    AccLover = c_acc / len(label_input) * 100
    blank2 = count_black / len(label_input) * 100

    assert(len(set(lover_idex)) == len(lover_idex)) # KNN can not pass
    print(f'Accuracy exchange 2clusters: {round(AccLover, 3)}   Blank_rate: {round(blank2, 2)}% ')
    return lover_idex


def ACC(label_input, pre_label_gmm, top_2_exchange_label, label_topN, args):  # cluster and vote
    #global class_gmm, not_class_gt, not_class_gmm,  class_gt, luckboy, First_unity, count_minus

    ######################### assign the gmm clusters labels  with 3 round ##############################


    class_gt = set(label_input)
    class_gmm = set(pre_label_gmm)
    count_minus = -1
    # first rd
    blank_rate1, accFrist, pre_label_gmm_feedback, First_unity, luckboy = FristRd(label_input, pre_label_gmm, class_gt, class_gmm, count_minus, args )

    pre_label_gmm_feedback, not_class_gt, not_class_gmm, second_unity = SecondRd(pre_label_gmm_feedback, class_gt, class_gmm, luckboy, First_unity, count_minus, label_input, pre_label_gmm)

    diff_mask = (label_input != pre_label_gmm_feedback)
    difference_label = np.where(diff_mask)[0]

    #####print, blank rate acc ##############
    label_mask = (np.array(pre_label_gmm_feedback) <= -1)
    label_indices = np.where(label_mask)[0]
    blank_rate2 = (len(label_indices) / len(pre_label_gmm_feedback))*100
    print(f'\nTop1 results after 2 rds:', end ='          ')
    print('Did not calculate Precision, Recal and F1')
    #precision_we = precision_score(label_input, pre_label_gmm_feedback, average='weighted') # average='macro' micro
    #f1_we = f1_score(label_input, pre_label_gmm_feedback, average='weighted')
    #rc_we = recall_score(label_input, pre_label_gmm_feedback, average='weighted')
    #print(f'Recal: {round(rc_we,2)}    Precision:{round(precision_we,2)},   F1:{round(f1_we,2)}')
    print(f'Accuracy gmm cluster 1st rd: {round(accFrist, 3)}   Blank_rate: {round(blank_rate1, 2)}%   overlap_th:{args.overlap_rate} ')
    acc2nd = sum(1 for a, b in zip(label_input, pre_label_gmm_feedback) if a == b) / len(label_input) * 100
    print(f'Accuracy gmm cluster 2nd rd: {round(acc2nd,3)}   Blank_rate: {round(blank_rate2, 2)}%   overlap_th:{0}' )
    # print(f'2rd Acc:{round(accuracy_score(label_input, pre_label_gmm_feedback)*100,3)}') the same as acc2nd
    #####print, blank rate acc end ###########


    # merge all trustble matched paris (1st 2nd)
    if len(second_unity) == 0:
        All_pair = First_unity
    else:
        All_pair = np.concatenate((First_unity, second_unity), axis=0)

    ########### 3rd round randomly assigned for unclustered (clusters) #################
    Third_unity = []
    if len(not_class_gt) > 0:
        print('**In the 3rd rd match gmm to gt with 0 overlap in the second rd')
        for x, y in zip(not_class_gmm, not_class_gt):
            Third_unity.append([x,y,0])
        All_pair = np.concatenate((All_pair, Third_unity), axis=0)

    ######################### assign the gmm clusters labels  with 3 round end##############################

    if args.topN: # TOP N ACC BASEED ON points #####
        TopN_Acc, top_4animal, result_prf = topNbyPoints(label_input, pre_label_gmm_feedback, label_topN, All_pair, args.topN, args.cal_recall)
        printtopN(TopN_Acc, acc2nd, result_prf)

    ######################### TOP 2 exchage highlight ACC ############################
    AccLover = 0
    lover_idex = 0
    if args.top2acc:
        lover_idex = accBasedCluster(top_2_exchange_label, All_pair, label_input, pre_label_gmm_feedback)
                                                                                          # Lover 2 works when --top2acc=1
    return np.array(pre_label_gmm_feedback), np.array(difference_label), round(acc2nd, 2), np.array(lover_idex), round(AccLover, 2), top_4animal






# ************************** k means***************************#

# if args.plotKmeans:
# 	ax2 = fig.add_subplot(234)
# 	plt.title("K-means-Tsne-2d")
# 	ax2.axis('off')
# 	plot_cluster_no_label(C, plt=ax2)

# if args.plotKmeans:
# 	print("Plotting Kmeans")
# 	plt.close('all')
# 	C, labels_C = k_means(reduction, args.components, 2)
# 	plot_cluster_no_label(C)
# 	plt.legend(loc='best')
# 	if not args.png_only:
# 		plt.savefig(os.path.join(args.out_path, "Kmeans.pdf"))
# 	plt.savefig(os.path.join(args.out_path, "Kmeans.png"))


def expand(x, y, gap=1e-4):
    add = np.tile([0, gap, np.nan], len(x))
    x1 = np.repeat(x, 3) + add
    y1 = np.repeat(y, 3) + add
    return x1, y1


def scatter_overlap(x, labels, filename, palette=None):
    # Choose a color palette with seaborn Randomly shuffle with the same seed
    num_classes = np.unique(labels).shape[0]

    # Create our figure/plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')

    # Convert labels to int
    labels = labels.astype(int)
    # reorder lists

    x_sort = (x[np.argsort(labels)])
    # l =labels.tolist()
    # m=x.tolist()
    # l, m = (zip(*sorted(zip(l, m))))
    # Map the colours to different labels

    labels_sort = np.sort(labels)

    last_label = 0
    last_i = 0
    plt.rcParams['lines.solid_capstyle'] = 'round'
    for i in range(len(labels_sort)):
        label_colours = np.array([palette[labels_sort[i - 1]]])
        if labels_sort[i] != last_label:
            ax.plot(*expand(x_sort[:, 0][last_i:i], x_sort[:, 1][last_i:i]), lw=7, c=label_colours[0],
                    alpha=0.5)
            last_i = i
        last_label = labels_sort[i]

    # Do some formatting
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('on')
    ax.axis('tight')
    plt.title('{} Images Graph'.format(str(len(labels))), fontsize='large', fontweight='bold')  # 设置字体大小与格式
    plt.tight_layout()

    # plt.suptitle('s')
    print(f"Saved visualisation")

    # plt.show()
    plt.savefig(filename + "_overlap.pdf")
    # plt.savefig(f"{os.path.join(self.__folder_path, subtitle)}.pdf")


def getEuclidean(point1, point2):
    dimension = len(point1)
    dist = 0.0
    for i in range(dimension):
        dist += (point1[i] - point2[i]) ** 2
    return math.sqrt(dist)


def k_means(dataset, k, iteration):
    # 初始化簇心向量,抽k个,每个是data 的 vector
    index = random.sample(list(range(len(dataset))), k)
    vectors = []
    for i in index:
        vectors.append(dataset[i])
    # 初始化标签 data数量
    labels = []
    for i in range(len(dataset)):
        labels.append(-1)
    # 根据迭代次数重复k-means聚类过程
    while (iteration > 0):
        # 初始化簇
        C = []
        for i in range(k):
            C.append([])
        for labelIndex, item in enumerate(dataset):
            classIndex = -1
            minDist = 1e6
            for i, point in enumerate(vectors):
                dist = getEuclidean(item, point)
                if (dist < minDist):
                    classIndex = i
                    minDist = dist
            C[classIndex].append(item)
            labels[labelIndex] = classIndex
        for i, cluster in enumerate(C):
            clusterHeart = []
            dimension = len(dataset[0])
            for j in range(dimension):
                clusterHeart.append(0)
            for item in cluster:
                for j, coordinate in enumerate(item):
                    clusterHeart[j] += coordinate / len(cluster)
            vectors[i] = clusterHeart
        iteration -= 1
    return C, labels


def plot_cluster_no_label(C, plt=plt):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(C)):
        coo_X = []  # x坐标列表
        coo_Y = []  # y坐标列表
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        plt.scatter(coo_X, coo_Y, marker='x', color=colValue[i % len(colValue)], label=i)


# ************************** k means*************************** #


# **************************  GMM  *************************** #

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(3, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))




# **************************  GMM  *************************** #
    #markers = ['o', 'd', 'x', 'h', 'H', 7, 4, 5, 6, '8', 'p', ',', '+', '.', 's', '*', 3, 0, 1, 2]
    #ll = random.choice(markers)

# plt.title("GMM Clusters")
# ax4.get_yaxis().set_visible(False)
# n_components = np.arange(2, 15)
# models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(reduction) #full’,‘tied’, ‘diag’, ‘spherical
# 		  for n in n_components]
# b_value = [m.bic(reduction) for m in models]
# Difference_Max = 0
# for i in range(len(b_value)-1):
# 	Difference_tmp = b_value[i] - b_value[i + 1]
# 	if Difference_tmp > Difference_Max:
# 		Difference_Max = Difference_tmp
# 		best_component = i + 1
#
# plt.plot(n_components, b_value, label='BIC')
# plt.plot(n_components, [m.aic(reduction) for m in models], label='AIC')
# plt.legend(loc='best')
# plt.xlabel('components')

# ax3.legend(loc='best')#, bbox_to_anchor=(0.2, 1.12), ncol=3)



    # # check if unique, if not, vote some to the others
    # First_unity = np.array(normal_p)
    # tmp_gmm_flo = First_unity[:, 0]
    # tmp_gmm_asiign = First_unity[:, 1]
    # tmp_gmm = [int(i) for i in tmp_gmm_flo]
    #
    # overlap_posation = []
    # compareOverlap = []
    # duplicate = [item for item, count in collections.Counter(tmp_gmm).items() if count > 1]
    #
    # for items in duplicate:
    #     tmp = []
    #     for i, the in enumerate(tmp_gmm):
    #         if the == items:
    #             tmp.append(normal_p[i])
    #     compareOverlap.append(tmp)
    #     #overlap_posation.append(tmp)
    # for item in compareOverlap:
    #     # who is higer
    #     ls = np.array(item)
    #     _, _, nn = ls.argmax(axis=0)
    # gather_delete = []
    # for i, jjj in enumerate(tmp):
    #     if i != nn:
    #         gather_delete.append(jjj)
    #
    # for i in gather_delete:
    #      normal_p.remove(i)