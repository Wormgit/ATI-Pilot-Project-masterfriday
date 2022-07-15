# Core libraries
import os
import sys, math
import cv2, random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
from sklearn.manifold import TSNE
from utils_visual import k_means, draw_ellipse, ACC, scatter_overlap, makedirs, toplabels
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import matplotlib as mpl

#import pandas as pd

# Define our own plot function

def scatter_color(reduction, select_reduction, c2, fn2):

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=40, c=[0.8, 0.8, 0.8], marker="o")
    plt.scatter(select_reduction[:, 0], select_reduction[:, 1], lw=0, s=40, c=c2, marker="o")
    plt.axis('off')
    plt.axis('tight')
    plt.tight_layout()
    if not args.png_only:
        plt.savefig(os.path.join(args.out_path, fn2 + ".pdf"))
    plt.savefig(os.path.join(args.out_path, fn2 + ".png"))
    if os.path.exists("/home/io18230/Desktop"):
        plt.show()
    plt.close()







def scatter_p(x, labels, filename, highlight=True):
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
            label_colours[i] = [0,0,0]


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
    #plt.show()
    if not args.png_only:
        plt.savefig(filename + ".pdf")
    plt.savefig(filename + ".png")
    plt.close()
    return label_colours

# Load and visualise embeddings via t-SNE

def plotEmbeddings_tsneMul(args, emd_path):
    # Load the embeddings into memory
    embeddings = np.load(emd_path)

    # Visualise the learned embedding via t-SNE
    visualiser = TSNE(n_components=2, perplexity=args.perplexity)
    #2d data after TSNE
    reduction = visualiser.fit_transform(embeddings['embeddings'])
    return reduction, embeddings['embeddings']




def plot_scikit_gmm(gmm, X, args, colour=True, ax=None, plt2d=0):

    '''
    returen  labels, log every hilikly hood spot and their top2 label [4,2] , colour of hilighted ,  second top label
    '''

    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    ax = ax or plt.gca()
    gmm.fit(X)
    labels = gmm.predict(X)

    # get rank value
    x_row=np.size(X,1)
    top_label_high = []
    colour_va = []
    if x_row!=2:
        # get some new samples based on the predicted GMM
        #samle, smaple_label = gmm.sample(6)
        # BBB = gmm._estimate_log_prob(samle)
        #io, ok = gmm._estimate_log_prob_resp(samle)
        # for pos_u, covar, w_weights in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        # for zzzz, vxvxv, w_weights in zip(gmm.precisions_, gmm.precisions_cholesky_, gmm.weights_):
        # score = gmm.score(X)
        # score_samples=gmm.score_samples(X)
        # confidence = gmm.predict_proba(X)  #posterior probability
        #per - sample average log - likelihood  of the given data  X

        # which point has high prob in an other cluster
        prob_list = gmm._estimate_weighted_log_prob(X)
        ind = np.argpartition(prob_list, -1, axis = 1)[-1:]
        sortedDist = np.sort(prob_list)   # sort to get dots with the biggest 2 probability
        sortedDist_top4 = np.maximum(sortedDist[:, (args.components - 4):], 0)
        top_1 = sortedDist_top4[:, 3]
        top_2 = sortedDist_top4[:, 2]

        top_label = toplabels(labels, prob_list, args.topN)

        interaction_2 = top_label[:,:2].tolist()
        for i in range(0,len(top_1)):
            if top_2[i] > 0:
                colour_va.append(int(top_1[i]/top_2[i]))
                top_label_high.append(interaction_2[i])
            else:
                colour_va.append(0)

    # if the input is 2d , plot and return label (normally we do not need it)
    if plt2d:
        if X.shape[1] == 2:
            if colour:
                ax.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap='plasma', zorder=2)
            else:
                ax.scatter(X[:, 0], X[:, 1], s=10, zorder=2)
            ax.axis('equal')
            w_factor = 0.2 / gmm.weights_.max()

            for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
                draw_ellipse(pos, covar, alpha=w * w_factor)

            if not args.png_only:
                plt.savefig(os.path.join(args.out_path, "ellipse_of_2d.pdf"))
            plt.savefig(os.path.join(args.out_path, "ellipse_of_2d.png"))
            plt.close()


    if x_row==2:
        return labels
    else:
        return labels, top_label_high, colour_va, top_label

def plotAllfolderGT(gmm, reduction, select_reduction_md, GT_LabelColor,label_colours_gmm_md,
                  select_label_colours_gmm_md, select_label_coloursanot_md, acc2, ARI_Score, args):
    # plot big picture
    scatter_color(reduction, select_reduction_md, select_label_coloursanot_md, 'differ_folder')

    fig = plt.figure(figsize=(8, 8))
    ax3 = fig.add_subplot(221)
    if args.show_title:
        plt.title("Folder")
    ax3.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=5, c = GT_LabelColor, marker="o")
    ax3.axis('off')

    # not important
    ax6 = fig.add_subplot(223)
    # plt.title("GMM-Tsne-2d")
    # plot_scikit_gmm(gmm, reduction, ax=ax6)
    # ax6.axis('off')
    if args.show_title:
        plt.title("GMM-Correct")
    ax6.axis('off')
    ax6.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=5, c=label_colours_gmm_md, marker="o")

    ax7 = fig.add_subplot(224)
    if args.show_title:
        plt.title("Highlight Difference")   #originan   c=label_colours_gmm
    ax7.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=5, c=[0.8, 0.8, 0.8], marker="o")
    ax7.scatter(select_reduction_md[:, 0], select_reduction_md[:, 1], lw=0, s=5, c=select_label_colours_gmm_md, marker="o")
    ax7.axis('off')

    ax8 = fig.add_subplot(222)
    ax8.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=5, c=[0.8, 0.8, 0.8], marker="o")
    ax8.scatter(select_reduction_md[:, 0], select_reduction_md[:, 1], lw=0, s=5, c=select_label_coloursanot_md, marker="o")
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



def plotAllKNN_GT(reduction, select_reduction,Tsne_LabelColor,select_label_coloursanothre,
                  select_label_colours_gmm_fd,label_colours_gmm_fd, acc, ARI_Score, args):
    #plot big picture
    scatter_color(reduction, select_reduction, select_label_coloursanothre, 'knn_differ')

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(221)
    if args.show_title:
        plt.title("KNN")
    ax1.axis('off')
    ax1.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=5, c=Tsne_LabelColor, marker="o")

    ax3 = fig.add_subplot(223)
    if args.show_title:
        plt.title("GMM-Correct")   #originan   c=label_colours_gmm
    ax3.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=5, c=label_colours_gmm_fd, marker="o")
    ax3.axis('off')

    ax5 = fig.add_subplot(222)
    ax5.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=5, c=[0.8, 0.8, 0.8], marker="o")
    ax5.scatter(select_reduction[:, 0], select_reduction[:, 1], lw=0, s=5, c=select_label_coloursanothre, marker="o")
    ax5.axis('off')

    ax2 = fig.add_subplot(224)
    if args.show_title:
        plt.title("Highlight Difference")
    ax2.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=5, c=[0.8, 0.8, 0.8], marker="o")
    ax2.scatter(select_reduction[:, 0], select_reduction[:, 1], lw=0, s=5, c=select_label_colours_gmm_fd, marker="o")
    ax2.axis('off')

    plt.suptitle("Set {} Classes. Acc: {} ARI: {}".format(args.components, acc, ARI_Score))
    plt.axis('tight')
    plt.tight_layout()
    if not args.png_only:
        plt.savefig(os.path.join(args.out_path, "All_KNN.pdf"))
    plt.savefig(os.path.join(args.out_path, "All_KNN.png"))
    if os.path.exists("/home/io18230/Desktop"):
        plt.show()
    plt.close()





    # ****************************"Plotting all in one"*********************************#


def acc_plotall(Label, color1, PreLabelGMM, col, label_top8, top_2_highlight, args, name = ''):
    #print("*********************************************")
    feb_pre, difference_label, acc, lover_label, acc_top2, top_4animal = ACC(Label, PreLabelGMM, top_2_highlight, label_top8, args)
    print('Accuracy of {}: {}\n'.format(name, acc))
    label_colours_gmm_fd = scatter_p(reduction, feb_pre, os.path.join(args.out_path, name),
                                     highlight=False)

    the_1 = [0]*len(PreLabelGMM)
    for i,itm in enumerate(difference_label):
        the_1 [itm] = 1
    np.array(the_1)

    ftop_4animal(reduction, color1, np.vstack((np.array(the_1), top_4animal)), 'fig7', size = 10)


    cccc = np.zeros(len(Label))
    for item in lover_label:
        cccc[item] = 1

    i = 0
    highlight_reduction = np.zeros(shape=(np.count_nonzero(cccc), 2))
    for count, item in enumerate(list(cccc)):
        if item != 0:
            iiii = highlight_reduction[i]
            ii = reduction[count]
            highlight_reduction[i] = reduction[count]
            i = i+1
    scatter_color_rainbow(reduction, highlight_reduction, plt.cm.rainbow, 'b_top2')

    # CHANGE COLOUR NAMES of different labels between gt and prediciton -- select-
    highlight_reduction = np.zeros(shape=(np.count_nonzero(col), 2))
    non_zero_likely = np.zeros(shape=(np.count_nonzero(col), 1))
    i = 0
    for count, item in enumerate(col):
        if item != 0:
            highlight_reduction[i] = reduction[count]
            non_zero_likely[i] = col[count]
            i = i+1

    cmap = plt.cm.rainbow  # summer  rainbow
    #norm = matplotlib.colors.Normalize(vmin=0, vmax=max(non_zero_likely))
    cm = plt.cm.get_cmap('RdYlBu')
    #ff = norm(non_zero_likely)
    #cmap(norm(non_zero_likely))
    scatter_color_rainbow(reduction, highlight_reduction, cm, 'dHigh likely')



    select_reduction = np.zeros(shape=(len(difference_label), 2))
    select_label_colours_gmm_fd = np.zeros(shape=(len(difference_label), 3))
    select_label_coloursanothre = np.zeros(shape=(len(difference_label), 3))
    for i, item in enumerate(difference_label):
        select_reduction[i] = reduction[item]
        select_label_colours_gmm_fd[i] = label_colours_gmm_fd[item]
        select_label_coloursanothre[i] = color1[item]

    #ari
    ARI_Score = round(metrics.adjusted_rand_score(PreLabelGMM, Label),2)
    print(f'ARI = {ARI_Score}')

    if name == 'GMM-CKNN-Label on tsne':
        plotAllKNN_GT(reduction, select_reduction, color1, select_label_coloursanothre,
                      select_label_colours_gmm_fd, label_colours_gmm_fd, acc, ARI_Score, args)
    if name == 'GMM-folder-Label on tsne':
        plotAllfolderGT(gmm, reduction, select_reduction, color1, label_colours_gmm_fd,
                       select_label_colours_gmm_fd, select_label_coloursanothre, acc, ARI_Score, args)
    #print("*********************************************\n")
    return feb_pre, acc, ARI_Score


# Main/entry method
if __name__ == '__main__':
    # Collate command line arguments
    parser = argparse.ArgumentParser(description='Parameters for visualising the embeddings via TSNE')

    parser.add_argument('--out_path', type=str, default='/home/io18230/Desktop/output/3visual')
    parser.add_argument('--embeddings_file', type=str,help='it is a folder name',  #RRR output sub_will_lab will_lab    combined Will
                        default='/home/io18230/Desktop/output/2combined/')  # train_embeddings folder_embeddings
                        #default = '/home/io18230/Desktop/output/folder_embeddings.npz')  2combined
    parser.add_argument('--perplexity', type=int, default=10, #or 10 15
                        help="Perplexity parameter for t-SNE, consider values between 5 and 50")
    parser.add_argument('--components', type=int, default=182) ### 20 182  8 20  186 will
    parser.add_argument('--png_only', type=int, default=1)
    parser.add_argument('--overlap_rate', type=float, default=0.3)
    parser.add_argument('--thres_h', type=float, default= 0.95) #0.95

    parser.add_argument('--acc_folder', type=int, default= 1)
    parser.add_argument('--acc_KNN', type=int, default=0)
    parser.add_argument('--show_title', type=int, default=1)
    parser.add_argument('--plot_2d', type=int, default=0)
    parser.add_argument('--save_txt', type=int, default=0)
    parser.add_argument('--control', type=int, default=1)
    parser.add_argument('--top2acc', type=int, default=0)
    parser.add_argument('--top16acc', type=int, default=1)
    parser.add_argument('--overboseAcc', type=int, default=1)
    parser.add_argument('--topN', type=int, default=16)
    parser.add_argument('--gmm_max_iter', type=int, default=200)
    parser.add_argument('--plot_1by1', type=int, default=0)


    # parser.add_argument('--train_embeddings_file', type=str, default='/home/io18230/Desktop/I_train/test_embeddings.npz')
    args = parser.parse_args()

    print(f"gmm_max_iter :{args.gmm_max_iter}")
    print("*********************************************")
    print("Loading embeddings") # Get colour, reduction and original data, plot the tsne
    acc_test_knn = []
    ARI_test_knn = []
    acc_test_folder =[]
    ARI_test_folder =[]
    file1 = os.listdir(args.embeddings_file)

    tmp = args.out_path
    for item in file1:
        for i in os.listdir(os.path.join(args.embeddings_file,item)):
            if i == 'folder_embeddings.npz':
                print(f'Output to folder: {file1}')
                args.out_path = os.path.join(tmp,item)
                makedirs(args.out_path)
                emd_path = os.path.join(args.embeddings_file, item, i)
                # 2d after tsne,data before tsne, corrected knn label, and colour , label from folder name, and colour
                reduction, X_128 = plotEmbeddings_tsneMul(args, emd_path)  # both gt and tsne label
                gmm = GaussianMixture(n_components=args.components, covariance_type='full', random_state=0, max_iter =150)
                PreLabelGMM, top_2_highlight, col, label_topN = plot_scikit_gmm(gmm, X_128, args,
                                                       ax=plt)  # label and rank (top 4), if probability > 0.95, rank = -1
                label_colours_gmm = scatter_p(reduction, PreLabelGMM, os.path.join(args.out_path, 'GMM-zFirstPredictColor'),
                                              highlight=False)

                ###ACC AND DIFFERENCE between the label from will and GMM on a tsne plot     # # ACC AND DIFFERENCE

                if args.acc_folder:
                    feb_pre_model, acc2, ARI2 = acc_plotall(GT_Label, GT_LabelColor, PreLabelGMM, col, label_topN, top_2_highlight, args,
                                                            name='GMM-folder-Label on tsne')
                    if not 'best' in item:
                        acc_test_folder.append(acc2)
                        ARI_test_folder.append(ARI2)

                if args.acc_KNN:
                    feb_pre, acc1, ARI = acc_plotall(Tsne_Label, Tsne_LabelColor, PreLabelGMM, col, label_topN, top_2_highlight, args, name='GMM-CKNN-Label on tsne')
                    if not 'best' in item:
                        acc_test_knn.append(acc1)
                        ARI_test_knn.append(ARI)

                print("*********************************************\n")

                # ****************************"Save "*********************************#
                np.savez(os.path.join(tmp, f"acc.npz"), acc_test_knn=acc_test_knn, ARI_test_knn=ARI_test_knn,acc_test_folder=acc_test_folder,ARI_test_folder=ARI_test_folder)

                if args.save_txt:
                    np.savez(os.path.join(args.out_path, f"ings.npz"), embeddings=X_128, reduction=reduction,
                             labels_folder= GT_Label, GMM_label= feb_pre_model)
                    # dataframe = pd.DataFrame({'second': a, 'b_name': b})
                    # np.savetxt(os.path.join(args.out_path, 'PreRank.txt'), PreRank)