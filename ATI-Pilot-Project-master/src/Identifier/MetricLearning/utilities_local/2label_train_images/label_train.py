# Core libraries
import os
import sys
import cv2
import json
import torch , csv
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# PyTorch
import torch
from torch.utils import data
from torch.autograd import Variable

# Local libraries
from utilities.utils import Utilities
from models.embeddings import resnet50
from utilities.utils_test import inferEmbedding_from_folder, inferEmbedding_from_folder_3, KNNAccuracy, makedirs

"""
File for inferring the embeddings of the test portion of a selected database and
evaluating its classification performance using KNN
"""
def knn_correct(name, f_preds):
    correct_pre = []
    if args.test_folder_class == 3:
        name5 = [item[:5] for item in name]
    else:
        name5 = [item[:3] for item in name]

    piece = []
    for i in range(len(name5) - 1):
        if name5[i + 1] != name5[i]:
            piece.append(i)
    piece.insert(0, -1)
    piece.append(len(name5) - 1)

    for i in range(len(piece) - 1):
        m1 = piece[i]
        m2 = piece[i + 1]
        sub_piece = f_preds[m1 + 1:m2 + 1]
        sub_piece = sub_piece.tolist()
        maxlabel = max(sub_piece, key=sub_piece.count)
        if sub_piece.count(maxlabel) != len(sub_piece):
            tem = [maxlabel for it in range(0, len(sub_piece))]
        else:
            tem = sub_piece

        correct_pre = correct_pre + tem

    return correct_pre

def evaluateModel(args, num_classes=186, softmax_enabled=1):
    """
    For a trained model, let's evaluate its performance
    """
    # Define our embeddings model
    model = resnet50(	pretrained=True,
                        num_classes=num_classes,
                        ckpt_path=model_path,
                        embedding_size=args.embedding_size,
                        img_type=args.img_type,
                        softmax_enabled=softmax_enabled	)

    # Put the model on the GPU and in evaluation mode
    if not os.path.exists("/home/io18230/Desktop"):
        model.cuda()
    model.eval()

    print('********************************************\nLoading Wills training embeddings')
    embeddings = np.load(embeddings_file)
    train_embeddings = embeddings['embeddings']
    train_labels = embeddings['labels']
    #na = embeddings['path']


    print('Inferring embeddings and labels of the test folder')
    if args.test_folder_class == 3:
        f_embeddings, f_labels, name = inferEmbedding_from_folder_3(args, model)
    else:
        f_embeddings, f_labels, name = inferEmbedding_from_folder(args, model)
    print('Calculating the KNN acc of the test folder')
    f_accuracy, f_preds = KNNAccuracy(train_embeddings, train_labels, f_embeddings, f_labels, args)


    print(f'Correcting KNN labels')
    correct_pre = knn_correct(name, f_preds)
    dateinfo = str(args.date[0]) + '-' + str(args.date[1]) + '_' + str(args.date[2]) + '-' + str(
        args.date[3])
    makedirs(os.path.join(args.save_path, dateinfo, '1Will'))
    save_path = os.path.join(args.save_path, dateinfo,'1Will', f"folder_embeddings.npz")
    np.savez(save_path, embeddings= f_embeddings, labels_knn= f_preds, labels_folder= f_labels, path = name, labels_correct_knn = correct_pre) #numpy array
    #sys.stdout.write(f"KNN accuracy of the test folder={str(round(f_accuracy, 2))} % \n")
    print(f'Saving csv and npz to {save_path}.\n********************************************\n')
    print(len(name),len(f_preds),len(correct_pre),len(f_labels))
    data1 = pd.DataFrame({'image_path': name, 'pre_label_knn': f_preds, 'corr_label_knn': correct_pre, 'model_label': f_labels})
    woduplicates = set(correct_pre)
    data1.to_csv(args.save_path + '/' + dateinfo +'/1Will/label{}'.format(str(len(woduplicates)))+'.csv')

    # Path to statistics file

    sys.stdout.flush()
    sys.exit(0)



# Main/entry method
if __name__ == '__main__':
    # Collate command line arguments
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--current_fold', type=int, default=0,  ###??
                        help="The current fold we'd like to test on")
    parser.add_argument('--exclude_difficult', type=int, default=0,
                        help='Whether to exclude difficult classes from the loaded dataset')
    parser.add_argument('--img_type', type=str, default="RGB",
                        help="Which image type should we retrieve: [RGB, D, RGBD]")
    parser.add_argument('--split', type=str, default="trainvalidtest",
                        help="Which evaluation mode to use: [trainvalidtest, traintest]")
    parser.add_argument('--dataset', nargs='?', type=str, default='RGBDCows2020',
                        help='Which dataset to use: [RGBDCows2020, OpenSetCows2020]')
    parser.add_argument('--batch_size', nargs='?', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--embedding_size', nargs='?', type=int, default=128) # 128
    parser.add_argument('--save_path', type=str, default='/home/io18230/Desktop/output',
                        help="Where to store the embeddings and statistics")


    parser.add_argument('--test_dir', nargs='?', default="/home/io18230/Desktop/RGBDCows2020f/Identification301_4days/RGB") #RGB2  sub_will
    parser.add_argument('--test_folder_class', type=int, default=3, help ='2 or 3')
    parser.add_argument('--neighbors', type=int, default=5, help='neighbors for KNN')
    parser.add_argument('--date', default=[1, 1, 12, 50], nargs = '+', type = int, help='start month, day, end ')
    args = parser.parse_args()

    if os.path.exists("/home/io18230/Desktop"):  # Home windows machine
        pt = '/home/io18230/Desktop/temp/code/ATI-Pilot-Project-masterfriday/ATI-Pilot-Project-master/src/Identifier/MetricLearning/models'
        model_path = pt + '/best_model_state.pkl'
        embeddings_file = pt + '/train_embeddings.npz'
    else:
        if os.path.exists("/home/will"):
            pt = '/home/will/Desktop'
        else:
            pt = '/work'
        model_path = pt + '/io18230/Projects/ATI-workspace/Embeddings/will_trained_186/best_model_state.pkl'
        embeddings_file = pt + '/io18230/Projects/ATI-workspace/Embeddings/will_trained_186/train_embeddings.npz'
    evaluateModel(args)