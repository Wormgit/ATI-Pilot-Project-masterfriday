# Core libraries
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
# PyTorch
import torch
from torch.utils import data
from torch.autograd import Variable
# Local libraries
from Utilities.ImageUtils import ImageUtils

# makedirs, KNNAccuracy, inferEmbedding_from_folder, inferEmbedding_from_folder_3, inferEmbeddings

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

# Use KNN to classify the embedding space
def KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels, args):
    # Define the KNN classifier
    n_neighbors = args.neighbors
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

def inferEmbedding_from_folder(args, model):
    # Wrap up the dataset in a PyTorch dataset loader
    m = 0
    outputs_embedding = np.zeros((1, args.embedding_size))
    labels_embedding = np.zeros((1))

    i = 1
    count_class = 0
    list = []
    label = []
    list_name = []
    list_name_tmp = []

    for items in os.listdir(args.test_dir):
        count_flag = 0
        for image in os.listdir(os.path.join(args.test_dir, items)):
            pos = image.find('2020-') #WILL'S DATA'S FORMAT
            month_ = int(image[pos + 5:pos + 7])
            date_ = int(image[pos + 8:pos + 10])
            #if (args.date[0] <= month_) * (month_ <= args.date[2]) * (args.date[1] <= date_) * (date_ <= args.date[3]): #DATA FILTER
            if 1:
                count_flag = 1
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
                        if not os.path.exists("/home/io18230/Desktop"):
                            images = Variable(images.cuda())
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
                    i = 1
                    break
                i += 1
        if count_flag:
            count_class +=1

        # We don't need the autograd engine
    print(f'number of class is {count_class}')
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
                        if not os.path.exists("/home/io18230/Desktop"):
                            images = Variable(images.cuda())
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
            if not os.path.exists("/home/io18230/Desktop"):
                images = Variable(images.cuda())

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

