# Core libraries
import os
import sys, math
import cv2, random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from utils_visual import k_means, draw_ellipse, ACC, scatter_overlap, makedirs
from sklearn.mixture import GaussianMixture
import pandas as pd

# Define our own plot function
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

    if filename == ' ':
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
        ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=label_colours, marker="o", alpha=alpha)
        # Highlighted points
        h_pts = np.array([x[i, :] for i in range(labels.shape[0]) if labels[i] in highlight_labels])
        # Colours
        h_colours = np.array([palette[labels[i]] for i in range(labels.shape[0]) if labels[i] in highlight_labels])
        # There may not have been any occurences of that label
        if h_pts.size != 0:
            # Replot highlight points with no alpha
            ax.scatter(h_pts[:, 0], h_pts[:, 1], lw=0, s=40, c=h_colours, marker="o")
        else:
            print(f"Didn't find any embeddings with the label: {highlight_labels}")
    # Just colour each point normally
    else:
        # Plot the points
        ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=label_colours, marker="o")

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
def plotEmbeddings_tsne(args):
    # Ensure there's something there
    if not os.path.exists(args.embeddings_file):
        print(f"No embeddings file at path: {args.embeddings_file}, exiting.")
        sys.exit(1)

    # Load the embeddings into memory
    embeddings = np.load(args.embeddings_file)
    if 'npz' in args.train_embeddings_file:
        embeddings_train = np.load(args.train_embeddings_file)

    print("   --Done")
    # Visualise the learned embedding via t-SNE
    visualiser = TSNE(n_components=2, perplexity=args.perplexity)
    reduction = visualiser.fit_transform(embeddings['embeddings'])
    print("Plotting Labels with the help of Tsne visualisation")

    # Plot the results and save to file
    label_colours = scatter_p(reduction, embeddings['labels'], os.path.join(args.out_path, 'Tsne_label'), highlight=False)
    label_colours_model = scatter_p(reduction, embeddings['labels_model'], os.path.join(args.out_path, 'Correct_label'), highlight=False)
    return reduction, embeddings['embeddings'], label_colours, embeddings['labels'], label_colours_model, embeddings['labels_model']

def plotEmbeddings_tsneMul(args, emd_path):
    # Load the embeddings into memory
    embeddings = np.load(emd_path)
    visualiser = TSNE(n_components=2, perplexity=args.perplexity)


    embeddings_train = np.load(args.train_embeddings_file)
    test_len = np.size(embeddings['embeddings'],0)
    # Visualise the learned embedding via t-SNE
    reduction = visualiser.fit_transform(np.concatenate((embeddings['embeddings'], embeddings_train['embeddings']),axis=0))
    b = reduction[test_len:]
    # Plot the results and save to file
    label_colours = scatter_p(reduction[0:test_len], embeddings['labels'], os.path.join(args.out_path, 'Tsne_label'), highlight=False)
    label_colours_model = scatter_p(reduction[0:test_len], embeddings['labels_model'], os.path.join(args.out_path, 'Correct_label'), highlight=False)
    return reduction[0:test_len], embeddings['embeddings'], label_colours, embeddings['labels'], label_colours_model, embeddings['labels_model'], reduction[test_len:], embeddings_train['embeddings']


def plot_scikit_gmm(gmm, X, args, colour=True, ax=None):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    ax = ax or plt.gca()
    gmm.fit(X)
    labels = gmm.predict(X)
    confidence = gmm.predict_proba(X)

    mm = np.max(confidence, axis=1)
    conf_rank = []
    for row in range(0,confidence.__len__()):
        if mm[row] >args.thres_h:
            # top_k_idx = np.where(confidence[row, :]>args.thres_h)
            #conf_rank.append((top_k_idx[0].tolist()))
            conf_rank.append(-1)
        else:
            top_k = 4
            top_k_idx = confidence.argsort()[row, ::-1][0:top_k]
            conf_rank.append(top_k_idx.tolist())

    # if the input is 2d , plot and return lable (normaly we donot need it)
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
            plt.savefig(os.path.join(args.out_path, "ellipse.pdf"))
        plt.savefig(os.path.join(args.out_path, "ellipse.png"))
        plt.close()

    return labels, conf_rank     # else: return the labels

def plotAllRealGT(gmm, reduction, select_reduction_md, GT_LabelColor,label_colours_gmm_md,
                  select_label_colours_gmm_md,select_label_coloursanot_md, acc2, args):
    fig = plt.figure()
    ax3 = fig.add_subplot(221)
    plt.title("Model label")
    ax3.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=10, c = GT_LabelColor, marker="o")
    ax3.axis('off')

    # not important
    ax6 = fig.add_subplot(223)
    # plt.title("GMM-Tsne-2d")
    # plot_scikit_gmm(gmm, reduction, ax=ax6)
    # ax6.axis('off')
    plt.title("Gmm model correct")
    ax6.axis('off')
    ax6.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=10, c=label_colours_gmm_md, marker="o")

    ax7 = fig.add_subplot(224)
    #originan   c=label_colours_gmm
    ax7.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=10, c=[0.8, 0.8, 0.8], marker="o")
    ax7.scatter(select_reduction_md[:, 0], select_reduction_md[:, 1], lw=0, s=10, c=select_label_colours_gmm_md, marker="o")
    ax7.axis('off')

    ax8 = fig.add_subplot(222)
    plt.title("Highlight")   #originan   c=label_colours_gmm
    ax8.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=10, c=[0.8, 0.8, 0.8], marker="o")
    ax8.scatter(select_reduction_md[:, 0], select_reduction_md[:, 1], lw=0, s=10, c=select_label_coloursanot_md, marker="o")
    ax8.axis('off')

    plt.suptitle("Set {} Classes. Acc{}".format(args.components, acc2))
    if not args.png_only:
        plt.savefig(os.path.join(args.out_path, "AllReal.pdf"))
    plt.savefig(os.path.join(args.out_path, "AllReal.png"))
    if os.path.exists("/home/io18230/Desktop"):
        plt.show()
    plt.close()

def plotAllKNN_GT(reduction, select_reduction,Tsne_LabelColor,select_label_coloursanothre,
                  select_label_colours_gmm_fd,label_colours_gmm_fd, acc, args):
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    plt.title("Tsne + Will's vote")
    ax1.axis('off')
    ax1.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=10, c=Tsne_LabelColor, marker="o")

    ax5 = fig.add_subplot(222)
    plt.title("Highlight")
    ax5.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=10, c=[0.8, 0.8, 0.8], marker="o")
    ax5.scatter(select_reduction[:, 0], select_reduction[:, 1], lw=0, s=10, c=select_label_coloursanothre, marker="o")
    ax5.axis('off')

    ax2 = fig.add_subplot(224)
    ax2.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=10, c=[0.8, 0.8, 0.8], marker="o")
    ax2.scatter(select_reduction[:, 0], select_reduction[:, 1], lw=0, s=10, c=select_label_colours_gmm_fd, marker="o")
    ax2.axis('off')

    ax5 = fig.add_subplot(223)
    plt.title("GMM-corrct color")   #originan   c=label_colours_gmm
    ax5.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=10, c=label_colours_gmm_fd, marker="o")
    ax5.axis('off')

    plt.suptitle("Set {} Classes. Acc{}".format(args.components, acc))
    if not args.png_only:
        plt.savefig(os.path.join(args.out_path, "AllKNN.pdf"))
    plt.savefig(os.path.join(args.out_path, "AllKNN.png"))
    if os.path.exists("/home/io18230/Desktop"):
        plt.show()
    plt.close()

    # ****************************"Plotting all in one"*********************************#


def acc_plotall(Label, color, PreLabelGMM, args, name = ''):
    print("*********************************************")
    feb_pre, difference_label, acc = ACC(Label, PreLabelGMM, args)
    print('Accuracy of {}: {}'.format(name, acc))
    label_colours_gmm_fd = scatter_p(reduction, feb_pre, os.path.join(args.out_path, name),
                                     highlight=False)

    select_reduction = np.zeros(shape=(len(difference_label), 2))
    select_label_colours_gmm_fd = np.zeros(shape=(len(difference_label), 3))
    select_label_coloursanothre = np.zeros(shape=(len(difference_label), 3))
    for i, item in enumerate(difference_label):
        select_reduction[i] = reduction[item]
        select_label_colours_gmm_fd[i] = label_colours_gmm_fd[item]
        select_label_coloursanothre[i] = color[item]

    if name == 'GMM-Lable on tsne':
        plotAllKNN_GT(reduction, select_reduction, Tsne_LabelColor, select_label_coloursanothre,
                      select_label_colours_gmm_fd, label_colours_gmm_fd, acc, args)
    else:
        plotAllRealGT(gmm, reduction, select_reduction, Tsne_LabelColor, label_colours_gmm_fd,
                       select_label_colours_gmm_fd, select_label_coloursanothre, acc, args)

    print("*********************************************\n")
    return feb_pre, acc

def scatter_aaa(x, labels, filename, x_t):
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

    if filename == ' ':
        return label_colours
    # Create our figure/plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    # Do we want to highlight some particular (e.g. difficult) labels

    # Just colour each point normally

    # Plot the points
    ax.scatter(x_t[:, 0], x_t[:, 1], lw=0, s=40, c=[0.8,0.8,0.8], marker="o")
    ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=label_colours, marker="o")

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


def scatter_ptrain(x, filename):
    # Create our figure/plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=[0.8,0.8,0.8], marker="o")

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

# Main/entry method
if __name__ == '__main__':
    # Collate command line arguments
    parser = argparse.ArgumentParser(description='Parameters for visualising the embeddings via TSNE')
    parser.add_argument('--out_path', type=str, default='/home/io18230/Desktop/visual')
    parser.add_argument('--embeddings_file', type=str,  #RRR output sub_will_lab will_lab    combined Will
                        default='/home/io18230/Desktop/op/')  # train_embeddings folder_embeddings
                        #default = '/home/io18230/Desktop/output/folder_embeddings.npz')
    parser.add_argument('--perplexity', type=int, default=10, #or 10 15
                        help="Perplexity parameter for t-SNE, consider values between 5 and 50")
    parser.add_argument('--save_txt', type=int, default=1)
    parser.add_argument('--components', type=int, default=9) ###170 for big  9  186 will
    parser.add_argument('--png_only', type=int, default=1)
    parser.add_argument('--overlap_rate', type=float, default=0.4)
    parser.add_argument('--thres_h', type=float, default= 0.95) #0.95       #/home/io18230/Desktop/I_train
    parser.add_argument('--train_embeddings_file', type=str, default='/home/io18230/Desktop/will_trained_186/train_embeddings.npz') #/home/io18230/Desktop/will_trained_186
    args = parser.parse_args()

    gmm = GaussianMixture(n_components=args.components, covariance_type='full', random_state=0)

    acc_test = []
    file1 = os.listdir(args.embeddings_file)
    print(file1)

    tmp = args.out_path
    for item in file1:
        for i in os.listdir(os.path.join(args.embeddings_file,item)):
            if i == 'folder_embeddings.npz':
                args.out_path = os.path.join(tmp,item)
                makedirs(args.out_path)
                emd_path = os.path.join(args.embeddings_file,item,i)
                reduction, X_128, Tsne_LabelColor, Tsne_Label, GT_LabelColor, GT_Label, train_reduc, train_X = plotEmbeddings_tsneMul(
                    args, emd_path)

                scatter_ptrain(train_reduc, os.path.join(args.out_path,'traint'))

                # 128d--tsne--2d--Gmm
                gmm = GaussianMixture(n_components=args.components, covariance_type='full', random_state=0)
                plot_scikit_gmm(gmm, reduction, args, ax=plt)  # do not save big pic
                # 128d--Gmm--get colour-- show on tsne
                PreLabelGMM, PreRank = plot_scikit_gmm(gmm, X_128, args,
                                                       ax=plt)  # label and rank (top 4), if probability > 0.95, rank = -1
                scatter_aaa(reduction, PreLabelGMM, os.path.join(args.out_path, 'Atraint'), train_reduc)
                label_colours_gmm = scatter_p(reduction, PreLabelGMM, os.path.join(args.out_path, 'GMM-Lable on tsne'),
                                              highlight=False)
                # plt.show()
                print("*********************************************\n")

                ###ACC AND DIFFERENCE between the label from will and GMM on a tsne plot     # # ACC AND DIFFERENCE
                feb_pre, acc1 = acc_plotall(Tsne_Label, Tsne_LabelColor, PreLabelGMM, args, name='GMM-KNN-Lable on tsne')
                feb_pre_model, acc2 = acc_plotall(GT_Label, GT_LabelColor, PreLabelGMM, args, name='GMM-GT-Lable on tsne')
                if not 'best' in item:
                    acc_test.append(acc2)

                # ****************************"Save "*********************************#
                if args.save_txt:
                    np.savez(os.path.join(args.out_path, f"ings.npz"), embeddings=X_128, reduction=reduction,
                             tsne_label=GT_Label, GMM_label=feb_pre_model)
                    # dataframe = pd.DataFrame({'second': a, 'b_name': b})
                    # np.savetxt(os.path.join(args.out_path, 'PreRank.txt'), PreRank)
            np.savez(os.path.join(tmp, f"acc.npz"), acc_test = acc_test)

    plt.close()