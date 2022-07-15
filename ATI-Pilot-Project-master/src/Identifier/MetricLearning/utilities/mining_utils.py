# Core libraries
import numpy as np
from itertools import combinations

# PyTorch
import torch

"""
File contains a selection of mining utilities for selecting triplets
Code is adapted from - https://github.com/adambielski/siamese-triplet
"""

# Find the distance between two vectors
def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

# Should return indices of selected anchors, positive and negative samples
# e.g. np array of shape [N_triplets x 3]
class TripletSelector:
    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels, sub_p):
        raise NotImplementedError

# Return all possible triplets
class AllTripletSelector(TripletSelector):
    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))

def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None

def random_easy_negative(loss_values):
    #hard_negatives = np.where(loss_values > 0)[0]
    #easy_negatives = np.where(loss_values < np.median(loss_values))[0]
    easy_negatives = np.where(loss_values < 10)[0]  #np.median(loss_values)
    return np.random.choice(easy_negatives) if len(easy_negatives) > 0 else None

def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """
    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels, sub_p, labels_neg, sub_n):
                                     #positive negtive     p
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        labels_neg = labels_neg.cpu().data.numpy()
        sub_p = sub_p.cpu().data.numpy()
        sub_n = sub_n.cpu().data.numpy()
        PosandNeg = np.concatenate((labels, labels_neg), axis=0)
        PosandNeg_sub = np.concatenate((sub_p, sub_n), axis=0)

        triplets = []
        for i in np.unique(PosandNeg):
            label_mask_first = (PosandNeg == i)
            label_indices_first = np.where(label_mask_first)[0]
            if len(label_indices_first) < 2:
                continue
            sub_label = [PosandNeg_sub[m,0] for m in label_indices_first]

            if len(set(sub_label)) == 1:
                label_mask = label_mask_first
                label_indices = label_indices_first
            else:
                # print('should not be here?')
                # for j in set(sub_label):
                #     label_mask = np.logical_and(labels == i, sub_p == j)
                #     label_indices  = np.where(label_mask)[0]
                #     if len(label_indices) < 2:
                #         continue
                continue

            if len(label_indices) > 2:
                pass

            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = np.array(list(combinations(label_indices, 2)))  # All anchor-positive pairs in this batch?

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                if loss_values.size != 0:
                    hard_negative = self.negative_selection_fn(loss_values) ### find one with big loss
                    if hard_negative is not None:
                        hard_negative = negative_indices[hard_negative]
                        triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])
                        # if anchor_positive[0] > 15:
                        #     m = 1
        return torch.LongTensor(np.array(triplets)), len(triplets)


class FunctionPositiveTripletSelector(TripletSelector):

    def __init__(self, margin, marginP, negative_selection_fn, cpu=True):
        super(FunctionPositiveTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.marginP = marginP
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels, sub_p, labels_neg, sub_n):
                                     #positive negtive     p
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        labels_neg = labels_neg.cpu().data.numpy()
        sub_p = sub_p.cpu().data.numpy()
        sub_n = sub_n.cpu().data.numpy()

        if labels.shape[1] > 1:
            anchor_l = labels[:, 0].reshape(-1, 1)
            anchor_subl = sub_p[:, 0].reshape(-1, 1)
            post_l = labels[:, 1].reshape(-1, 1)
            post_subl = sub_p[:, 1].reshape(-1, 1)
            PosandNeg = np.concatenate((anchor_l, labels_neg), axis=0)
            PosandNeg_sub = np.concatenate((anchor_subl, sub_n), axis=0)

            triplets = []
            for i, t in enumerate(PosandNeg):
                label_mask_same = (PosandNeg == t)

                if i < len(post_l):
                    label_mask_first_pos_match = (PosandNeg == post_l[i])
                    label_indices_pos_match = np.where(label_mask_first_pos_match)[0]
                    for item in label_indices_pos_match:
                        if PosandNeg_sub[item] != post_subl[i]:
                            label_mask_first_pos_match[item] = False

                    label_mask_first_all = label_mask_same + label_mask_first_pos_match
                else:
                    label_mask_first_all = label_mask_same

                label_indices_first_all = np.where(label_mask_first_all)[0]


                if len(label_indices_first_all) < 2:
                    continue
                sub_label = [PosandNeg_sub[m, 0] for m in label_indices_first_all]

                if len(sub_label) > 1:
                    label_mask = label_mask_first_all
                    label_indices = label_indices_first_all
                else:
                    continue
                negative_indices = np.where(np.logical_not(label_mask))[0]
                anchor_positives = np.array(
                    list(combinations(label_indices, 2)))  # All anchor-positive pairs in this batch?

                ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
                idx_PosMax = np.array(torch.max(ap_distances, 0)[1])
                for i_idex, (anchor_positive, ap_distance) in enumerate(zip(anchor_positives, ap_distances)):
                    if i_idex in idx_PosMax:
                        loss_values = ap_distance - distance_matrix[
                            torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(
                                negative_indices)] + self.margin
                        loss_values = loss_values.data.cpu().numpy()
                        if loss_values.size != 0:
                            hard_negative = self.negative_selection_fn(loss_values)  ### find one with big loss
                            if hard_negative is not None:
                                hard_negative = negative_indices[hard_negative]
                                triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        else:
            PosandNeg = np.concatenate((labels, labels_neg), axis=0)
            PosandNeg_sub = np.concatenate((sub_p, sub_n), axis=0)

            triplets = []
            for i in np.unique(PosandNeg):
                label_mask_first = (PosandNeg == i)
                label_indices_first = np.where(label_mask_first)[0]

                if len(label_indices_first) < 2:
                    continue
                sub_label = [PosandNeg_sub[m,0] for m in label_indices_first]

                if len(set(sub_label)) == 1:
                    label_mask = label_mask_first
                    label_indices = label_indices_first
                else:
                    # print('should not be here?')
                    # for j in set(sub_label):
                    #     label_mask = np.logical_and(labels == i, sub_p == j)
                    #     label_indices  = np.where(label_mask)[0]
                    #     if len(label_indices) < 2:
                    #         continue
                    continue

                negative_indices = np.where(np.logical_not(label_mask))[0]
                anchor_positives = np.array(list(combinations(label_indices, 2)))  # All anchor-positive pairs in this batch?

                ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
                idx_PosMax = np.array(torch.max(ap_distances,0)[1])
                for i_idex, (anchor_positive, ap_distance) in enumerate(zip(anchor_positives, ap_distances)):
                    if i_idex in idx_PosMax:
                        loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                        loss_values = loss_values.data.cpu().numpy()
                        if loss_values.size != 0:
                            hard_negative = self.negative_selection_fn(loss_values) ### find one with big loss
                            if hard_negative is not None:
                                hard_negative = negative_indices[hard_negative]
                                triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])
        return torch.LongTensor(np.array(triplets)), len(triplets)


def HardestNegativeTripletSelector(margin=0, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)

def HardestPositiveTripletSelector(margin=0, marginP=0, cpu=False): return FunctionPositiveTripletSelector(margin=margin, marginP=marginP,
                                                                                 negative_selection_fn=random_easy_negative,
                                                                                 cpu=cpu)

def RandomNegativeTripletSelector(margin=0, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector(margin=0, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)