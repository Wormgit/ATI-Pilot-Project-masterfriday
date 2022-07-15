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
from utilities.utils_test import inferEmbedding_from_folder, inferEmbedding_from_folder_3, inferEmbeddings, KNNAccuracy, makedirs

"""
File for inferring the embeddings of the test portion of a selected database and
evaluating its classification performance using KNN
"""

def evaluateModel(args,num_classes=172,softmax_enabled=0):
    # Define our embeddings model
    model = resnet50(	pretrained=True,
                        num_classes=num_classes,
                        ckpt_path=args.model_path,
                        embedding_size=args.embedding_size,
                        img_type=args.img_type,
                        softmax_enabled=softmax_enabled	)
    # Put the model on the GPU and in evaluation mode
    if not os.path.exists("/home/io18230/Desktop"):
        model.cuda()
    model.eval()

    # Get the embeddings and labels of the training set and testing set
    if args.load_train_embdings:
        print('Loading training embeddings')
        embeddings = np.load(args.train_embeddings_file)
        train_embeddings = embeddings['embeddings'] # We only need embeddings
        if args.KNN_OR_EMBEDDING:
            train_labels = embeddings['labels']
        #na = embeddings['path'] # no this item if we got from the train on bc4
    else:
        print('Inferring the training embeddings')
        train_dataset = Utilities.selectDataset(args, "train") # Load the relevant datasets
        train_embeddings, train_labels = inferEmbeddings(args, model, train_dataset, "train")

    print('Inferring embeddings and labels of the test folder')
    if args.test_folder_class == 3:
        f_embeddings, f_labels, name = inferEmbedding_from_folder_3(args, model)
    else:
        f_embeddings, f_labels, name = inferEmbedding_from_folder(args, model)

    if args.KNN_OR_EMBEDDING:
        print('Loading testing labels and training labels from wills network')
        embeddingssss = np.load(args.Label_will)
        f_labels = embeddingssss['labels']
    #加个assert name 2 和name的
    #     reader_val = list(csv.reader(open(args.csv_from_will, 'r'), delimiter=','))
    #     del reader_val[0]
    #     tmp = path.tolist()
    #     for items in reader_val:
    #         position = tmp.index(items [1])
    #         labels_embedding[position] = items [2]
    #         labels_embedding = np.array(labels_embedding).astype(float)
    # else:
    #     labels_embedding = embeddings['labels']

        print('Calculating the KNN acc of the test folder')
        f_accuracy, f_preds = KNNAccuracy(train_embeddings, train_labels, f_embeddings, f_labels, args)
        print('Saving the predicted labels')
        data1 = pd.DataFrame({'image_path': name, 'pre_label': f_preds})
        data1.to_csv(args.save_path +'/no_use_label.csv')

    print('\nSaving the embeddings')
    save_path = os.path.join(args.save_path, f"folder_embeddings.npz")
    if args.KNN_OR_EMBEDDING:
        np.savez(save_path, embeddings=f_embeddings, labels= f_preds, path = name) #numpy array
    else:
        np.savez(save_path, embeddings=f_embeddings, path=name)

    # Path to statistics file
    if args.KNN_OR_EMBEDDING:
        sys.stdout.write(f"Folder accuracy={str(f_accuracy)} \n")
    if not args.test_a_folder:   #暂时用不上了
        test_dataset = Utilities.selectDataset(args, "test")
        # assert train_dataset.getNumClasses() == test_dataset.getNumClasses()
        # Load the validation set too if we're supposed to
        if args.split == "trainvalidtest":
            valid_dataset = Utilities.selectDataset(args, "valid")

        print('Do not infer testing and training embeddings')
        test_embeddings, test_labels = inferEmbeddings(args, model, test_dataset, "test")
        #Is there a validation set too
        if args.split == "trainvalidtest":
            valid_embeddings, valid_labels = inferEmbeddings(args, model, valid_dataset, "valid")

            # Combine the training and validation embeddings/labels to help KNN
            train_embeddings = np.concatenate((train_embeddings, valid_embeddings))
            train_labels = np.concatenate((train_labels, valid_labels))

            # Get performance on the validation and testing sets
            valid_accuracy, valid_preds = KNNAccuracy(train_embeddings, train_labels, valid_embeddings, valid_labels,args)

        # Get performance on the testing set
        test_accuracy, test_preds = KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels,args)

        # Is there a validation set too
        if args.split == "trainvalidtest":
            # Store statistics (to produce confusion matrices)
            stats = {	'valid_labels': valid_labels.astype(int).tolist(),
                        'valid_preds': valid_preds.astype(int).tolist(),
                        'test_labels': test_labels.astype(int).tolist(),
                        'test_preds': test_preds.astype(int).tolist()		}

            # Write it out to the console so that subprocess can pick them up and close
            sys.stdout.write(f"Validation accuracy={str(valid_accuracy)}; Testing accuracy={str(test_accuracy)};"
                             f"folder accuracy={str(f_accuracy)} \n")

        elif args.split == "traintest":
            # Store statistics (to produce confusion matrices)
            stats = {	'test_labels': test_labels.astype(int).tolist(),
                        'test_preds': test_preds.astype(int).tolist()		}

            # Write it out to the console so that subprocess can pick them up and close
            sys.stdout.write(f"Testing accuracy={str(test_accuracy)}\n")
        stats_path = os.path.join(args.save_path, f"{args.img_type}_testing_stats.json")
        # Save them to file
        with open(stats_path, 'w') as handle:
            json.dump(stats, handle, indent=4)

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
    parser.add_argument('--embedding_size', nargs='?', type=int, default=128)
    parser.add_argument('--save_path', type=str, default='/home/io18230/Desktop/output',
                        help="Where to store the embeddings and statistics")
    parser.add_argument('--test_a_folder', type=int, default=1)

    parser.add_argument('--test_dir', nargs='?', default="/home/io18230/Desktop/sub_will")
    #parser.add_argument('--test_dir', nargs='?', default="/home/io18230/Desktop/RGBDCows202066/Identification/RGB") #RGBDCows2020/Identification/RGB
    parser.add_argument('--test_folder_class', type=int, default=2,help ='2 or 3')
                                                                                    # I_train  I_train_today
    parser.add_argument('--model_path', nargs='?', default='/home/io18230/Desktop/I_train_today/current_model_state.pkl') # current_model_state best
    parser.add_argument('--class_num', type=int, default=172) # need to fill in one that is the same as the number of trainng folders #will can be every thing.
    parser.add_argument('--load_train_embdings', type=int, default=1) # chose to load the trained one or 0 infer based on the json file.
    parser.add_argument('--train_embeddings_file', default='/home/io18230/Desktop/I_train_today/train_embeddings.npz') #w test_embeddings
    parser.add_argument('--neighbors', type=int, default=5, help='neighbors for KNN')
    parser.add_argument('--KNN_OR_EMBEDDING', type=int, default=0, help='1 KNN, 0 embedding ONLY')
    parser.add_argument('--Label_will', default='/home/io18230/Desktop/rgb2_lab/combined/folder_embeddings.npz')
    args = parser.parse_args()

    evaluateModel(args, num_classes=args.class_num) # 手动算一下class_num