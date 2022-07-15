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
from Utilities.ImageUtils import ImageUtils

"""
File for inferring the embeddings of the test portion of a selected database and
evaluating its classification performance using KNN
"""

def evaluateModel(args):
    """
    For a trained model, let's evaluate its performance
    """

    num_classes = 186
    # Define our embeddings model
    model = resnet50(	pretrained=True,
                        num_classes=num_classes,
                        ckpt_path=model_path,
                        embedding_size=args.embedding_size,
                        img_type=args.img_type,
                        softmax_enabled=softmax_enabled	)

    # Put the model on the GPU and in evaluation mode
    #model.cuda()
    model.eval()

    # Get the embeddings and labels of the training set and testing set
    if load_train_embdings:
        print('Loading training embeddings')
        embeddings = np.load(embeddings_file)
        train_embeddings = embeddings['embeddings']
        train_labels = embeddings['labels']
        #na = embeddings['path']
    else:
        # Load the relevant datasets
        train_dataset = Utilities.selectDataset(args, "train")

        if not args.class_num:
            num_classes = train_dataset.getNumClasses()
        else:
            num_classes = args.class_num
        if args.will_186_net:
            num_classes = 186
        print('Inferring the training embeddings')
        train_embeddings, train_labels = inferEmbeddings(args, model, train_dataset, "train")


    print('Inferring embeddings and labels of the test folder')
    if args.test_folder_class == 3:
        f_embeddings, f_labels, name = inferEmbedding_from_folder_3(args, model)
    else:
        f_embeddings, f_labels, name = inferEmbedding_from_folder(args, model)

    if 'npz' in args.test_label_file:
        print('Loading testing labels and training labels from wills network')
        embeddingssss = np.load(args.test_label_file)
        f_labels = embeddingssss['labels']
        name2 = embeddingssss['path']
        embeddingssssttt =np.load(args.train_label_file)
        train_labels = embeddingssssttt['labels']
        name3 = embeddingssssttt['path']
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
    f_accuracy, f_preds = KNNAccuracy(train_embeddings, train_labels, f_embeddings, f_labels)
    print('Saving the predicted labels')
    data1 = pd.DataFrame({'image_path': name, 'pre_label': f_preds})
    data1.to_csv(args.save_path +'/label.csv')

    print('Saving the embeddings')
    if args.save_embeddings:
        save_path = os.path.join(args.save_path, f"folder_embeddings.npz")
        np.savez(save_path, embeddings=f_embeddings, labels= f_preds, path = name) #numpy array
    # Path to statistics file

    sys.stdout.write(f"Folder accuracy={str(f_accuracy)} \n")
    if not args.test_a_folder:
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
            valid_accuracy, valid_preds = KNNAccuracy(train_embeddings, train_labels, valid_embeddings, valid_labels)

        # Get performance on the testing set
        test_accuracy, test_preds = KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels)

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


# Use KNN to classify the embedding space
def KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels, n_neighbors=5):
    # Define the KNN classifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-4)

    # Give it the embeddings and labels of the training set
    neigh.fit(train_embeddings, train_labels)

    # Total number of testing instances
    total = len(test_labels-1)

    # Get the predictions from KNN
    predictions = neigh.predict(test_embeddings)
    correct = (predictions == test_labels).sum()
    accuracy = (float(correct) / total) * 100
    return accuracy, predictions

# Infer the embeddings for a given dataset
def inferEmbeddings(args, model, dataset, split):
    # Wrap up the dataset in a PyTorch dataset loader
    data_loader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=6, shuffle=False)
    # Embeddings/labels to be stored on the testing set
    outputs_embedding = np.zeros((1, args.embedding_size))
    labels_embedding = np.zeros((1))

    # Iterate through the training/testing portion of the dataset and get their embeddings
    for images, m, n, labels, q in tqdm(data_loader, desc=f"Inferring {split} embeddings"):
        # We don't need the autograd engine
        with torch.no_grad():
            # Put the images on the GPU and express them as PyTorch variables
            # images = Variable(images.cuda())

            # Get the embeddings of this batch of images
            outputs = model(images)
            # m += args.batch_size

            # Express embeddings in numpy form
            embeddings = outputs.data
            embeddings = embeddings.detach().cpu().numpy()

            # Convert labels to readable numpy form
            labels = labels.view(len(labels))
            labels = labels.detach().cpu().numpy()

            # Store testing data on this batch ready to be evaluated
            outputs_embedding = np.concatenate((outputs_embedding, embeddings), axis=0)
            labels_embedding = np.concatenate((labels_embedding, labels), axis=0)

    # If we're supposed to be saving the embeddings and labels to file
    if args.save_embeddings:
        # Construct the save path
        save_path = os.path.join(args.save_path, f"{split}_embeddings.npz")

        # Save the embeddings to a numpy array
        np.savez(save_path, embeddings=outputs_embedding, labels=labels_embedding)

    return outputs_embedding, labels_embedding

def inferEmbedding_from_folder(args, model):
    # Wrap up the dataset in a PyTorch dataset loader
    m = 0
    outputs_embedding = np.zeros((1, args.embedding_size))
    labels_embedding = np.zeros((1))

    i = 1
    list = []
    label = []
    list_name = []
    list_name_tmp = []

    for items in os.listdir(args.test_dir):
        for image in os.listdir(os.path.join(args.test_dir, items)):
            img1 = cv2.imread(os.path.join(args.test_dir, items, image))

            list.append(img1)
            if items.isdigit():
                label.append(int(items))
            else:
                label.append(items)
            list_name_tmp.append(os.path.join(items, image))
            if i % 4 ==0:
                for ittt in list_name_tmp:
                    list_name.append(ittt)
                    list_name_tmp = []
                images = ImageUtils.npToTorch([list[0],list[1],list[2],list[3]], return_tensor=True, resize=(224, 224))
                with torch.no_grad():
                    # Put the images on the GPU and express them as PyTorch variables
                    # images = Variable(images.cuda())
                    # Get the embeddings of this batch of images
                    outputs = model(images)
                    m += args.batch_size

                    # Express embeddings in numpy form
                    embeddings = outputs.data
                    embeddings = embeddings.detach().cpu().numpy()

                    # Convert labels to readable numpy form
                    labels = np.array(label)
                    l = len(labels)
                    #labels = labels.view(len(labels))
                    #labels = labels.detach().cpu().numpy()

                    # Store testing data on this batch ready to be evaluated
                    outputs_embedding = np.concatenate((outputs_embedding, embeddings), axis=0)
                    labels_embedding = np.concatenate((labels_embedding, labels), axis=0)

                list = []
                label = []
            i += 1
        # We don't need the autograd engine
    return np.delete(outputs_embedding, 0, axis = 0), np.delete(labels_embedding, 0), list_name

def inferEmbedding_from_folder_3(args, model):
    # Wrap up the dataset in a PyTorch dataset loader
    m = 0
    outputs_embedding = np.zeros((1, args.embedding_size))
    labels_embedding = np.zeros((1))

    i = 1
    list = []
    label = []
    list_name = []
    list_name_tmp = []

    for items in os.listdir(args.test_dir):
        for sub in os.listdir(os.path.join(args.test_dir,items)):
            for image in os.listdir(os.path.join(args.test_dir, items, sub)):
                img1 = cv2.imread(os.path.join(args.test_dir, items, sub, image))

                list.append(img1)
                label.append(int(items))
                list_name_tmp.append(os.path.join(items,sub, image))
                if i % 4 ==0:
                    for ittt in list_name_tmp:
                        list_name.append(ittt)
                        list_name_tmp = []
                    images = ImageUtils.npToTorch([list[0],list[1],list[2],list[3]], return_tensor=True, resize=(224, 224))
                    with torch.no_grad():
                        # Put the images on the GPU and express them as PyTorch variables
                        # images = Variable(images.cuda())
                        # Get the embeddings of this batch of images
                        outputs = model(images)
                        m += args.batch_size

                        # Express embeddings in numpy form
                        embeddings = outputs.data
                        embeddings = embeddings.detach().cpu().numpy()

                        # Convert labels to readable numpy form
                        labels = np.array(label)
                        l = len(labels)
                        #labels = labels.view(len(labels))
                        #labels = labels.detach().cpu().numpy()

                        # Store testing data on this batch ready to be evaluated
                        outputs_embedding = np.concatenate((outputs_embedding, embeddings), axis=0)
                        labels_embedding = np.concatenate((labels_embedding, labels), axis=0)

                    list = []
                    label = []
                i += 1
        # We don't need the autograd engine
    return np.delete(outputs_embedding, 0, axis = 0), np.delete(labels_embedding, 0), list_name

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
    parser.add_argument('--save_embeddings', type=int, default=1,
                        help="Should we save the embeddings to file")
    parser.add_argument('--save_path', type=str, default='/home/io18230/Desktop/output',
                        help="Where to store the embeddings and statistics")
    parser.add_argument('--test_a_folder', type=int, default=1)

    #parser.add_argument('--test_dir', nargs='?', default="/home/io18230/Desktop/RGB2")
    parser.add_argument('--test_dir', nargs='?', default="/home/io18230/Desktop/RGBDCows2020_/Identification/RGB") #RGBDCows2020/Identification/RGB
    parser.add_argument('--test_folder_class', type=int, default=3,help ='2 or 3')

    parser.add_argument('--will_186_net', type=int, default=0)  # 1 for wills net
    parser.add_argument('--model_path', nargs='?', default='/home/io18230/Desktop/current_model_state.pkl') #w  current_model_state
    parser.add_argument('--class_num', type=int, default=172) # need to fill in one that is the same as the number of trainng folders #will can be every thing.
    parser.add_argument('--load_train_embdings', type=int, default=1) # chose to load the trained one or infer based on the json file.
    parser.add_argument('--train_embeddings_file', default='/home/io18230/Desktop/train_embeddings.npz') #w 22 will_trained_186 test_embeddings
    parser.add_argument('--test_label_file', default='/home/io18230/Desktop/label/folder_embeddings.npz')
    parser.add_argument('--train_label_file', default='/home/io18230/Desktop/label/folder_embeddings.npz')

    args = parser.parse_args()

    if args.load_train_embdings:
        load_train_embdings = 1
    else:
        load_train_embdings = 0

    if args.will_186_net:
        softmax_enabled = 1
        load_train_embdings = 1
        test_a_folder = 1
        model_path = '/home/io18230/Desktop/will_trained_186/best_model_state.pkl'
        embeddings_file ='/home/io18230/Desktop/will_trained_186/train_embeddings.npz'
        print('Using wills training embeddings')
    else:
        print('Using my training embeddings')
        softmax_enabled = 0
        model_path = args.model_path
        embeddings_file = args.train_embeddings_file

    evaluateModel(args)