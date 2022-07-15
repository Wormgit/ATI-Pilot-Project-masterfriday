import os
import numpy as np
import matplotlib.pyplot as plt



def distEclud(vecA, vecB):
    """
    计算两个向量的欧式距离
    """
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def gmm_study(gmm, X, GT_Label, args):
    gmm.fit(X)

    import matplotlib.mlab as mlab
    from scipy.stats import norm
    from scipy.spatial.distance import mahalanobis

    # real data distance
    # for item in np.unique(GT_Label):
    #     k = int(item)
    #     distance = []
    #     distancem = []
    #     idx = np.where(GT_Label == k)[0]
    #     min_dis = []
    #     for jj in range(len(gmm.means_)):
    #         min_dis.append(distEclud(X[idx], gmm.means_[jj]))
    #         best_gmm = min_dis.index(min(min_dis))
    #     for i in idx:
    #         distance.append(distEclud(X[i], gmm.means_[best_gmm]))
    #         distancem.append(mahalanobis(gmm.means_[best_gmm], X[i], gmm.covariances_[best_gmm]))
    #     plt.figure()
    #     plt.hist(distance, 10, alpha=0.5, label='euclidean' + str(k), edgecolor='deepskyblue')
    #     plt.hist(distancem, 10, alpha=0.5, label='mahalanobis' + str(k))
    #     plt.legend(loc='upper right')
    #     plt.title(f'{len(idx)} Data Points')
    #     plt.savefig(os.path.join(args.out_path, 'Data' + str(k) + ".png"))
    #     plt.show()

    # simulated data
    # samle, smaple_label = gmm.sample(len(gmm.means_) * 500)  # get some new samples based on the predicted GMM
    # for k in range(len(gmm.means_)):
    #     distance = []
    #     distancem = []
    #     idx = np.where(smaple_label == k)[0]
    #     for i in idx:
    #         distance.append(distEclud(samle[i], gmm.means_[k]))
    #         distancem.append(mahalanobis(gmm.means_[k], samle[i], gmm.covariances_[k]))
    #     plt.figure()
    #     n, bins, patches = plt.hist(distance, 100, alpha=0.5, label='euclidean' + str(k), edgecolor='deepskyblue')
    #     plt.hist(distancem, 100, alpha=0.5, label='mahalanobis' + str(k))
    #     # (mu, sigma) = norm.fit(distance)
    #     # y = norm.pdf(bins, mu, sigma)
    #     # l = plt.plot(bins, y, 'r--', linewidth=2)
    #     plt.legend(loc='upper right')
    #     plt.title('Generated sample')
    #     plt.savefig(os.path.join(args.out_path, str(k) + ".png"))
    #     plt.show()

    score_samples = gmm.score_samples(X)

    samle, smaple_label = gmm.sample(len(gmm.means_) * 10)
    confidence = gmm.predict_proba(samle)
    BBB = gmm._estimate_log_prob(samle)
    io, ok = gmm._estimate_log_prob_resp(samle)
    m = gmm._estimate_weighted_log_prob(samle)
    #for pos_u, covar, w_weights in zip(gmm.means_, gmm.covariances_, gmm.weights_):
    for zzzz, vxvxv, w_weights in zip(gmm.precisions_, gmm.precisions_cholesky_, gmm.weights_):
        score = gmm.score(X)
 #posterior probability
    #per - sample average log - likelihood  of the given data  X