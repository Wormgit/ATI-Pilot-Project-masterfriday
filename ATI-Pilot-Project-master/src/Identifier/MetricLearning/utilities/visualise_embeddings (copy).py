# Core libraries
import os,sys
import argparse, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import matplotlib as mpl
import matplotlib.colors as mcolors

# Local libraries
from utils_visual import k_means, draw_ellipse, ACC, makedirs, toplabels, count_if_enough
from tsne_embedding import scatter_singel, scatter_density, scatter_images, scatter_overlap
from utils_visual_other import gmm_study

size_dot = 10
sizeHigh = 20


def GrayTopN(reduction, GT_LabelColor, top_4animal, name, size):
    bias = 1

    # make gray color
    colors = []
    for t in range(0, 55, 1):
        m = t / 100
        colors.append((m, m, m))  # 'Gray'0.5 'k'=0
    colors = colors[::-1]
    cm = mpl.colors.ListedColormap(colors[::])  # 让颜色倒过来，原来的

    graytem = top_4animal.sum(axis=0)
    label_4animal = np.where(graytem)
    value = [graytem[i] for i in label_4animal]
    m = len(label_4animal[0])
    select_top_4animal = np.zeros(shape=(m, 2))
    for i, item in enumerate(label_4animal[0]):
        select_top_4animal[i] = reduction[item]

    plt.figure(figsize=(10, 8))
    plt.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=size, c=GT_LabelColor, marker="o", alpha=0.7)
    m = np.array(value[0]).reshape(m, 1)
    data_plot = np.hstack((select_top_4animal, m))
    sorted_array = data_plot[np.argsort(data_plot[:, 2])]

    plt.scatter(sorted_array[:, 0], sorted_array[:, 1], c=sorted_array[:, 2], lw=0, s=25, cmap=cm, marker="o", vmin=1,
                vmax=16)

    plt.axis('off')
    plt.axis('tight')
    plt.tight_layout()
    cb = plt.colorbar(shrink=0.9)
    cb.set_label('Top N class', size=16)
    cb.ax.tick_params(labelsize=16)  # 设置色标刻度字体大小。
    if not args.png_only:
        plt.savefig(os.path.join(args.out_path, name + 'TopN' + ".pdf"))
    plt.savefig(os.path.join(args.out_path, name + 'TopN' + ".png"))
    if os.path.exists("/home/io18230/Desktop"):
        plt.show()
    plt.close()

    if args.plot_1by1:
        for count in range(len(top_4animal)):
            label_4animal = np.where(top_4animal[count])
            m = len(label_4animal[0])
            select_top_4animal = np.zeros(shape=(m, 2))
            for i, item in enumerate(label_4animal[0]):
                select_top_4animal[i] = reduction[item]
            plt.figure(figsize=(8, 8))
            plt.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=size, c=GT_LabelColor, marker="o")
            plt.scatter(select_top_4animal[:, 0], select_top_4animal[:, 1], lw=0, s=size_dot, c='k', marker="o")
            acc = (len(reduction) - len(select_top_4animal)) / len(reduction)
            if args.show_title:
                plt.title(f"{str(count + bias), round(acc * 100, 2)}%")
            plt.axis('off')
            plt.axis('tight')
            plt.tight_layout()

            if not args.png_only:
                plt.savefig(os.path.join(args.out_path, '1by1', name + str(count + bias) + ".pdf"))
            plt.savefig(os.path.join(args.out_path, '1by1', name + str(count + bias) + ".png"))
            if os.path.exists("/home/io18230/Desktop"):
                plt.show()
            plt.close()


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# Define our own plot function
def scatter_color_rainbow(reduction, select_reduction, cm, fn2, non_zero_likely=[0], alpha=1):
    if len(non_zero_likely) > 1:
        plt.figure(figsize=(10, 8))
    else:
        plt.figure(figsize=(8, 8))

    plt.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=size_dot, c=[0.8, 0.8, 0.8], marker="o")

    if len(non_zero_likely) > 1:
        plt.scatter(select_reduction[:, 0], select_reduction[:, 1], c=non_zero_likely, lw=0, s=25,
                    cmap=cm, marker="o", vmin=min(non_zero_likely), vmax=max(non_zero_likely), alpha=alpha)
        cb = plt.colorbar(shrink=0.9)
        cb.ax.tick_params(labelsize=16)  # 设置色标刻度字体大小。
        cb.set_label('Likelihood', size=16)
    else:
        plt.scatter(select_reduction[:, 0], select_reduction[:, 1], lw=0, s=size_dot, c='deepskyblue', marker="o")

    plt.axis('off')
    plt.axis('tight')
    plt.tight_layout()

    if not args.png_only:
        plt.savefig(os.path.join(args.out_path, fn2 + ".pdf"))
    plt.savefig(os.path.join(args.out_path, fn2 + ".png"))
    if os.path.exists("/home/io18230/Desktop"):
        plt.show()
    plt.close()


def scatter_p(x, labels, filename, highlight=False):
    # Get the number of classes (number of unique labels)
    num_classes = np.unique(labels).shape[0]
    # Choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 200))
    # Randomly shuffle with the same seed
    np.random.seed(26)
    np.random.shuffle(palette)

    # Convert labels to int
    labels = labels.astype(int)

    # Map the colours to different labels
    label_colours = np.array([palette[labels[i]] for i in range(labels.shape[0])])
    for i in range(len(labels)):
        if labels[i] < 0:
            label_colours[i] = [0, 0, 0]

    if 'xxx' in filename:
        return label_colours
    # Create our figure/plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    # Do we want to highlight some particular (e.g. difficult) labels
    if highlight:
        # Which labels should we highlight (the "difficult" individuals)
        highlight_labels = [54, 69, 73, 173]
        # Colour for non-highlighted points
        label_colours = np.zeros(label_colours.shape)
        # Alpha value for non-highlighted points
        alpha = 1.0
        # Plot all the points with some transparency
        ax.scatter(x[:, 0], x[:, 1], lw=0, s=10, c=label_colours, marker="o", alpha=alpha)
        # Highlighted points
        h_pts = np.array([x[i, :] for i in range(labels.shape[0]) if labels[i] in highlight_labels])
        # Colours
        h_colours = np.array([palette[labels[i]] for i in range(labels.shape[0]) if labels[i] in highlight_labels])
        # There may not have been any occurences of that label
        if h_pts.size != 0:
            # Replot highlight points with no alpha
            ax.scatter(h_pts[:, 0], h_pts[:, 1], lw=0, s=10, c=h_colours, marker="o")
        else:
            print(f"Didn't find any embeddings with the label: {highlight_labels}")
    # Just colour each point normally
    else:
        # Plot the points
        ax.scatter(x[:, 0], x[:, 1], lw=0, s=10, c=label_colours, marker="o")

    # Do some formatting
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.tight_layout()
    # Save it to file
    # plt.show()
    if not args.png_only:
        plt.savefig(filename + ".pdf")
    plt.savefig(filename + ".png")
    plt.close()
    return label_colours, palette


# Load and visualise embeddings via t-SNE

def plotEmbeddings_tsneMul(args, emd_path):

    # Ensure there's something there
    if not os.path.exists(args.embeddings_file):
        print(f"No embeddings file at path: {args.embeddings_file}, exiting.")
        sys.exit(1)

    # Load the embeddings into memory
    embeddings = np.load(emd_path)
    # Visualise the learned embedding via t-SNE
    visualiser = TSNE(n_components=2, perplexity=args.perplexity)
    # 2d data after TSNE
    reduction = visualiser.fit_transform(embeddings['embeddings'])


    #Dataset: start from 5 Feb, end in March 11. 30 days (Feb: No 6,7,8,9,17,30)
    date = args.date
    keepX= []
    for i, item in enumerate(embeddings['path']):
        pos = item.find('2020-')
        month_ = int(item[pos + 5:pos + 7])
        date_ = int(item[pos + 8:pos + 10])
        if (date[0] <= month_) * (month_ <= date[2]) * (date[1] <= date_) * (date_ <= date[3]):
            keepX.append(i)

    # Plot the results and save to file
    # corrected knn label
    name_knn = 'xxx'  # discard knn labels
    if args.acc_KNN:
        name_knn = 'KNN_label on tsne'
    tmplabels_knn = embeddings['labels_folder']  #embeddings['labels_knn']
    label_colours = scatter_p(reduction[keepX], tmplabels_knn[keepX], os.path.join(args.out_path, name_knn))
    # label from folder name
    label_colours_model, palette = scatter_p(reduction[keepX], embeddings['labels_folder'][keepX],
                                    os.path.join(args.out_path, 'Folder_label on tsne'))
    return reduction[keepX], embeddings['embeddings'][keepX], label_colours, tmplabels_knn[keepX], label_colours_model, \
           embeddings['labels_folder'][keepX],  embeddings['path'][keepX], palette


def plt_gmm_2d(gmm, X, ax=None):
    '''
    input x must be 2 d !!!
    '''

    ax = ax or plt.gca()
    gmm.fit(X)
    labels = gmm.predict(X)

    gmm.fit(X)
    # if the input is 2d , plot and return label (normally we do not need it)
    if X.shape[1] == 2:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap='plasma', zorder=2)
        ax.axis('equal')
        w_factor = 0.2 / gmm.weights_.max()

        for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
            draw_ellipse(pos, covar, alpha=w * w_factor)
        plt.savefig(os.path.join(args.out_path, "Zellipse_of_2d.png"))
        plt.close()





def PlotGMM(gmm, X, reduction, args, path, ax=None):
    '''
    return labels, log every hilikly hood spot and their top2 label [4,2] , colour of hilighted ,  second top label
    '''

    from scipy.special import logsumexp
    ax = ax or plt.gca()
    gmm.fit(X)
    PreLabelGMM = gmm.predict(X)

    label_colours_gmm,_ = scatter_p(reduction, PreLabelGMM, os.path.join(args.out_path, 'GMM-zFirstPredictColor'))

    # which point has high prob in an other cluster
    prob_list = gmm._estimate_weighted_log_prob(X)
    log_prob_norm = logsumexp(prob_list, axis=1)

    prob_list = prob_list - log_prob_norm[:, np.newaxis]
    prob_list = prob_list.astype(np.float128)
    #prob_list = gmm.predict_proba(X)
    prob_list = np.exp(prob_list)

    sortedDist = np.sort(prob_list)  # sort to get dots with the biggest 2 probability

    top_1 = sortedDist[:, -1]
    top_2 = sortedDist[:, -2]


    top_label = toplabels(PreLabelGMM, prob_list, args.topN)

    #Rank them
    RankLikely = []
    for i in range(0, len(top_1)):
        RankLikely.append(round(top_1[i] - top_2[i], 4))
    RankLikely = sorted(enumerate(RankLikely), key=lambda e: e[1]) # position and value
    min = RankLikely[0][1]


    non_zero_likely2 = np.zeros(shape=(len(RankLikely), 1))
    for item in RankLikely:
        if math.isinf(item[1]):
            non_zero_likely2[item[0]] = -1
        else:
            non_zero_likely2[item[0]] = item[1] / min
    max = np.max(non_zero_likely2)
    for i, item in enumerate(non_zero_likely2):
        if item== -1 :
            non_zero_likely2[i] = max
    k = np.min(non_zero_likely2)

    plt.figure()
    plt.hist(non_zero_likely2, 1000, range=(0 ,non_zero_likely2.max()/1),  label='number', edgecolor='deepskyblue', density = 0)
    plt.legend(loc='upper right')
    plt.title('Histogram likely hood top 1  - top 2')
    plt.savefig(os.path.join(args.out_path, "Histogram.png"))
    plt.show()


    likely_median = np.median(non_zero_likely2)
    index_bottom = np.where(non_zero_likely2>=likely_median)[0]
    index_top = np.where(non_zero_likely2 < likely_median)[0]

    index_revolt = np.where(top_2 > 1)[0]

    top_path = path[index_top]
    bottom_path = path[index_bottom]
    revolt_path = path[index_revolt]
    np.savez(args.out_path + "/top_bottom.npz", top_path=top_path, bottom_path=bottom_path, revolt_path=revolt_path)


    # 寻找造反的 二王子 revolt
    top_label_high = []
    colour_va = []
    interaction_2 = top_label[:, :2].tolist()
    for i in range(0, len(top_1)):
        if top_2[i] > 0:
            colour_va.append(int(top_1[i] - top_2[i]))
            top_label_high.append(interaction_2[i])
        else:
            colour_va.append(0)

    SaveEnough = count_if_enough(top_label_high, args)  # exchangable labels if the number is enough

    return PreLabelGMM, label_colours_gmm, SaveEnough, colour_va, top_label, RankLikely


def plot4image(reduction, select_reduction_md, GT_LabelColor, label_colours_gmm_md,
               select_label_colours_gmm_md, select_label_coloursanot_md, acc2, ARI_Score, args, name):
    # plot big picture
    scatter_color_rainbow(reduction, select_reduction_md, select_label_coloursanot_md, 'differ' + name)

    fig = plt.figure(figsize=(8, 8))
    ax3 = fig.add_subplot(221)
    if args.show_title:
        plt.title(name)
    ax3.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=5, c=GT_LabelColor, marker="o")
    ax3.axis('off')

    # not important
    ax6 = fig.add_subplot(223)
    # plt.title("GMM-Tsne-2d")
    # PlotGMM(gmm, reduction, ax=ax6)
    # ax6.axis('off')
    if args.show_title:
        plt.title("GMM-Correct")
    ax6.axis('off')
    ax6.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=5, c=label_colours_gmm_md, marker="o")

    ax7 = fig.add_subplot(224)
    if args.show_title:
        plt.title("Highlight Difference")  # originan   c=label_colours_gmm
    ax7.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=5, c=[0.8, 0.8, 0.8], marker="o")
    ax7.scatter(select_reduction_md[:, 0], select_reduction_md[:, 1], lw=0, s=5, c=select_label_colours_gmm_md,
                marker="o")
    ax7.axis('off')

    ax8 = fig.add_subplot(222)
    ax8.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=5, c=[0.8, 0.8, 0.8], marker="o")
    ax8.scatter(select_reduction_md[:, 0], select_reduction_md[:, 1], lw=0, s=5, c=select_label_coloursanot_md,
                marker="o")
    ax8.axis('off')

    plt.suptitle("Set {} Classes. Acc: {} ARI: {}".format(args.components, acc2, ARI_Score))
    plt.axis('tight')
    plt.tight_layout()
    if not args.png_only:
        plt.savefig(os.path.join(args.out_path, "All_folder.pdf"))
    plt.savefig(os.path.join(args.out_path, "All_folder.png"))
    if os.path.exists("/home/io18230/Desktop"):
        plt.show()
    plt.close()

    # ****************************"Plotting all in one"*********************************#


def acc_plotall(GTLabel, GTcolor, PreLabelGMM, col, label_topN, top_2_exchange_label, RankLikely, args, name=''):
    # ari
    ARI_Score = round(metrics.adjusted_rand_score(PreLabelGMM, GTLabel), 2)
    print(f'ARI = {ARI_Score}')

    feb_pre, difference_label, acc, lover_label, acc_top2, GrayPoints = ACC(GTLabel, PreLabelGMM, top_2_exchange_label,
                                                                            label_topN, args)

    path_tmp = os.path.join(args.out_path, os.path.basename(args.embeddings_file)[:-4])
    scatter_overlap(reduction, feb_pre, path_tmp, palette=palette, show_class=0, name='gmm')
    scatter_density(reduction, feb_pre, path_tmp, args, palette=palette, name='_Gmm', show_class = 0, textup=4, imgSize = 30)
    scatter_density(reduction, feb_pre, path_tmp, args, palette=palette, name='_Gmm2', textup=4, imgSize=30)
    if args.single_highlight:
        singleImag(reduction, feb_pre, args, palette, path, name='Single_Gmm')

    label_colours_gmm_fd,_ = scatter_p(reduction, feb_pre, os.path.join(args.out_path, name))



    # print CHANGE COLOUR NAMES of different labels between gt and prediciton -- select-
    highlight_reduction = np.zeros(shape=(np.count_nonzero(col), 2))
    non_zero_likely = np.zeros(shape=(np.count_nonzero(col), 1))
    i = 0
    for count, item in enumerate(col):
        if item != 0:
            highlight_reduction[i] = reduction[count]
            non_zero_likely[i] = col[count]
            i = i + 1

    colors = ['lightskyblue', 'deepskyblue', 'dodgerblue', 'royalblue', 'mediumblue', 'navy', 'k']
    cm = mcolors.LinearSegmentedColormap.from_list("mycmap", colors)
    if len(highlight_reduction) > 0:
        non_zero_likely = NormalizeData(non_zero_likely)
        scatter_color_rainbow(reduction, highlight_reduction, cm,
                              f'GrayTopN_High likely {len(non_zero_likely)} out of {len(reduction)}', non_zero_likely)



    ##not very sucessfull
    # label_mask = np.zeros(len(GTLabel))
    # for item in lover_label:
    #     label_mask[item] = 1
    #
    # i = 0
    # highlight_reduction = np.zeros(shape=(np.count_nonzero(label_mask), 2))
    # for count, item in enumerate(list(label_mask)):
    #     if item != 0:
    #         highlight_reduction[i] = reduction[count]
    #         i = i+1
    # scatter_color_rainbow(reduction, highlight_reduction, 'b', plt.cm.rainbow, 'b_top2')

    # plot GrayTopN
    label_mask = [0] * len(PreLabelGMM)
    for item in difference_label:
        label_mask[item] = 1
    GrayTopN(reduction, GTcolor, np.vstack((np.array(label_mask), GrayPoints)), 'Gray', size=5)

    # the last part
    select_reduction = np.zeros(shape=(len(difference_label), 2))
    select_label_colours_gmm_fd = np.zeros(shape=(len(difference_label), 3))
    select_label_coloursanothre = np.zeros(shape=(len(difference_label), 3))
    for i, item in enumerate(difference_label):
        select_reduction[i] = reduction[item]
        select_label_colours_gmm_fd[i] = label_colours_gmm_fd[item]
        select_label_coloursanothre[i] = GTcolor[item]

    if name == 'GMM-CKNN-Label on tsne':
        plot4image(reduction, select_reduction, GTcolor, label_colours_gmm_fd,
                   select_label_colours_gmm_fd, select_label_coloursanothre, acc, ARI_Score, args, name='KNN')
    if name == 'GMM-folder-Label on tsne':
        plot4image(reduction, select_reduction, GTcolor, label_colours_gmm_fd,
                   select_label_colours_gmm_fd, select_label_coloursanothre, acc, ARI_Score, args, name='Folder')

    return feb_pre, acc, ARI_Score


def singleImag(reduction, label, args, palette, path, name):
    makedirs(os.path.join(args.out_path, name))
    path_tmp = os.path.join(args.out_path, name, os.path.basename(args.embeddings_file)[:-4])
    x, y, l = scatter_singel(reduction, label, path_tmp, args, palette=palette, img_path=path,
                             position=1)  # position 0 dots 1 images 2 image on the side

    # histogram
    if name == 'Single_gt':
        xxx = list(range(len(x)))
        total_width, n = 0.6, 2
        width = total_width / n

        plt.figure(figsize=(8, 8))

        for i in range(len(xxx)):
            xxx[i] = xxx[i] + width
        plt.bar(xxx, l, width=width, label='number of Instance', fc='mediumblue')
        # for a, b in zip(x, y):
        #     plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)
        plt.legend()
        plt.savefig(path_tmp + "/Histogram.pdf")

# Main/entry method
if __name__ == '__main__':
    # Collate command line arguments
    parser = argparse.ArgumentParser(description='Parameters for visualising the embeddings via TSNE')

    parser.add_argument('--out_path', type=str, default='/home/io18230/Desktop/output/')
    parser.add_argument('--img_dir', type=str, default='/home/io18230/Desktop/RGBDCows2020w/test_2class_4days_exclude_noblack')
    #parser.add_argument('--img_dir', type=str, default='/home/io18230/Desktop/RGBDCows2020w/Will20')
    parser.add_argument('--embeddings_file', type=str, help='it is a folder name',
                        # RRR output sub_will_lab will_lab    combined Will
                        default='/home/io18230/Desktop/output/')  # train_embeddings folder_embeddings
    # default = '/home/io18230/Desktop/output/folder_embeddings.npz')  2combined 11 same1will8
    parser.add_argument('--perplexity', type=int, default=10,  # or 10 15        20 8
                        help="Perplexity parameter for t-SNE, consider values between 5 and 50")
    parser.add_argument('--components', type=int, default=167)  ### 20 182  8   186 will #167  171
    parser.add_argument('--png_only', type=int, default=1)
    parser.add_argument('--overlap_rate', type=float, default=0.3)

    parser.add_argument('--acc_folder', type=int, default=1)
    parser.add_argument('--acc_KNN', type=int, default=0)
    parser.add_argument('--show_title', type=int, default=1)
    parser.add_argument('--plot_2d', type=int, default=1)
    parser.add_argument('--save_txt', type=int, default=1)

    parser.add_argument('--lowRandNumber', type=int, default=1)
    parser.add_argument('--top2acc', type=int, default=1, help='exchange 2 clusters')

    parser.add_argument('--topN', type=int, default=16, help='point based')
    parser.add_argument('--gmm_max_iter', type=int, default=150)
    parser.add_argument('--plot_1by1', type=int, default=0)
    parser.add_argument('--cal_recall', type=int, default=0)
    parser.add_argument('--date', default=[2,14, 2, 19])  #  2, 14, 2, 19   1, 1, 12, 30  2, 1, 5, 5   2, 14, 2, 24   2, 20, 2, 24
    parser.add_argument('--single_highlight', type=int, default=0)


    # did not use it
    parser.add_argument('--overboseAcc', type=int, default=1)
    # parser.add_argument('--train_embeddings_file', type=str, default='/home/io18230/Desktop/I_train/test_embeddings.npz')
    args = parser.parse_args()

    print("*******************   Loading embeddings   ************************")
    print(f"\nMax iter of GMM :{args.gmm_max_iter}")
    # Get colour, reduction and original data, plot the tsne
    acc_test_knn = []
    ARI_test_knn = []
    acc_test_folder = []
    ARI_test_folder = []



    date_folder = str(args.date[0]) + '-' + str(args.date[1]) + '_' + str(args.date[2]) + '-' + str(args.date[3])
    tmp = args.embeddings_file
    args.embeddings_file = os.path.join(tmp, date_folder, '2combined/')
    file1 = os.listdir(args.embeddings_file)

    # st = ''
    # for item in args.date:
    #     st = st+ '_' + str(item)

    tmp = args.out_path
    for item in file1:
        for i in os.listdir(os.path.join(args.embeddings_file, item)):
            if i == 'folder_embeddings.npz':
                print(f'Output to folder: {file1}')
                args.out_path = os.path.join(tmp, date_folder, '3visual', item)
                makedirs(args.out_path)
                emd_path = os.path.join(args.embeddings_file, item, i)
                # 2d after tsne, 128d data before tsne, corrected knn label, and colour, label from folder name, and colour
                # load embedding and labels here
                reduction, X_128, Tsne_LabelColor, Tsne_Label, GT_LabelColor, GT_Label, path, palette = plotEmbeddings_tsneMul(args,
                                                                                                                emd_path)  # both gt and tsne label

                path_tmp = os.path.join(args.out_path, os.path.basename(args.embeddings_file)[:-4])
                #scatter_overlap(reduction, GT_Label, path_tmp, palette=palette, show_class = 0, name = 'gt')
                # all images in the labeled position

                # scatter_density(reduction, GT_Label, path_tmp, args, palette=palette, name='_gt', show_class = 0, textup=4, imgSize = 30)
                # scatter_density(reduction, GT_Label, path_tmp, args, palette=palette, name='_gt2', textup=4, imgSize=30)
                #
                # scatter_images(reduction, path_tmp, args, img_path=path)
                #
                # if args.single_highlight:
                #     singleImag(reduction, GT_Label, args, palette, path, name='Single_gt')

                gmm = GaussianMixture(n_components=args.components, covariance_type='full', random_state=0,
                                      max_iter=args.gmm_max_iter)

                #gmm_study(gmm, X_128, GT_Label, args)

                # if args.plot_2d:
                #     print("..Plotting 2d Gmm after Tsne")  # tsne--2d reduction--Gmm   # input reduction, output ellipse
                #     plt_gmm_2d(gmm, reduction)  # do not save big pic



                # PreLabelGMM, and colour, the rest???
                # not working# label and rank (top 4), if probability > 0.95, rank = -1
                print("..2d Tsne coloured with 128d Gmm")
                PreLabelGMM, label_colours_gmm, top_2_exchange_label, col, label_topN, RankLikely = PlotGMM(gmm, X_128,
                                                                                                            reduction,
                                                                                                            args,
                                                                                                            path,
                                                                                                            ax=plt)

                ###ACC AND DIFFERENCE between the label from will and GMM on a tsne plot     # # ACC AND DIFFERENCE


                if args.acc_folder:
                    # col, label_top8, top_2_highlight, args, name = '')
                    feb_pre_model, acc2, ARI2 = acc_plotall(GT_Label, GT_LabelColor, PreLabelGMM, col, label_topN,
                                                            top_2_exchange_label, RankLikely, args,
                                                            name='GMM-folder-Label on tsne')
                    if not 'best' in item:
                        acc_test_folder.append(acc2)
                        ARI_test_folder.append(ARI2)

                if args.acc_KNN:
                    feb_pre, acc1, ARI = acc_plotall(Tsne_Label, Tsne_LabelColor, PreLabelGMM, col, label_topN,
                                                     top_2_exchange_label, RankLikely, args,
                                                     name='GMM-CKNN-Label on tsne')
                    if not 'best' in item:
                        acc_test_knn.append(acc1)
                        ARI_test_knn.append(ARI)

                # ****************************"Save "*********************************#
                np.savez(os.path.join(tmp, f"acc.npz"), acc_test_knn=acc_test_knn, ARI_test_knn=ARI_test_knn,
                         acc_test_folder=acc_test_folder, ARI_test_folder=ARI_test_folder)

                if args.save_txt:
                    np.savez(os.path.join(args.out_path, f"ings.npz"), embeddings=X_128, reduction=reduction,
                             labels_folder=GT_Label, GMM_label=feb_pre_model)
