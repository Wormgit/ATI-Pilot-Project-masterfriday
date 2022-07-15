#!/usr/bin/env python

# Core libraries
import os
import sys
from PIL import Image
sys.path.append("../")
import cv2
import json
import pickle
import shutil
import random, csv
import numpy as np
from tqdm import tqdm
from numpy import linalg as LA
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from collections import OrderedDict

# Image interpretation
# from lime import lime_image
# from skimage.segmentation import mark_boundaries

# DL Stuff
import torch
import torchvision
from torch.utils import data

# My libraries
from config import cfg
from Utilities.DataUtils import DataUtils
from Utilities.ImageUtils import ImageUtils
from Visualiser.VisualisationManager import VisualisationManager

count = 0
folderLenCtl = 314
OldVsNew = 10


class RGBDCows2020(data.Dataset):
    """
    This Class manages everything to do with the RGBDCows2020 dataset and interfacing with it in PyTorch
    """

    def __init__(self,
                 split_mode="trainvalidtest",
                 fold=0,
                 num_training_days=-1,
                 num_testing_days=-1,
                 split="train",  # train, test or valid
                 img_type="RGB",  # D, RGB or RGBD
                 retrieval_mode="single",  # Retrieve triplets or single images
                 depth_type="normal",  # What type of depth image do we want (normal or binarised)
                 img_size=(224, 224),  # Resolution to get images at
                 augment=False,  # Whether to randomly augment images (only works for training)
                 transform=None,  # Transform images to PyTorch form
                 suppress_info=False,  # Suppress printing some info about this dataset
                 exclude_difficult=False  # Whether to exclude difficult categories
                 ):
        """ Class constructor

        Args:
            split_mode (str, optional): How are we splitting the dataset into training and testing? The default
                "random" uses random splits from the entire data corpus and so may be heavily correlated. The second
                option "day" splits the dataset by day. If using "day" splits, then the number of days to use for
                training and testing need to be specified (num_training_days, num_testing_days)
            fold (int, optional): If we're in "random" split mode, which fold should we use to retrieve train
                test splits
            num_training_days (int, optional): If we're in "day" split mode, how many days should make up the
                training set (there are 31 days total). This will choose num_training_days from the start of the
                ordered list of days
            num_testing_days (int, optional): As for num_training_days but will pick from the end of the list of
                days for the testing set.
            split (str, optional): Does this instance of the dataset want to retrieve training or test data?
        """

        # Initialise superclass
        super(RGBDCows2020, self).__init__()

        # The root directory for the dataset itself
        self.__root = cfg.DATASET.RGBDCOWS2020_LOC
        if split == 'valid':
            self.__root = cfg.DATASET.RGBDCOWS2020_val2

        # Which split mode are we in
        self.__split_mode = split_mode

        # Should we exclude a list of difficult examples from the dataset
        self.__exclude_difficult = exclude_difficult

        # The split we're after (e.g. train/test)
        self.__split = split

        # What image type to retrieve? (D, RGB or RGBD)
        self.__img_type = img_type

        # What retrieval mode are we after, single images or triplets (a, p, n)
        self.__retrieval_mode = retrieval_mode

        # Static list of difficult categories (e.g. all one colour with no markings) to be removed if requested
        if self.__exclude_difficult:
            self.__exclusion_cats = ["054", "069", "073", "173"]
        else:
            self.__exclusion_cats = []

        # If we're in triplet mode, remove animal 182 as it only has two instances and when split into train/valid/test
        # causes issues with finding positives and negatives
        # if self.__retrieval_mode == "triplet":
        #     self.__exclusion_cats.append("182")

        # The fold we're currently using for train/test splits (if in random mode)
        self.__fold = str(fold)

        # Select and load the split file we're supposed to use
        # We're splitting into train/valid/test (single fold for the time being)
        if self.__split_mode == "trainvalidtest":
            self.__splits_filepath = os.path.join(self.__root, "single_train_valid_test_splits.json")
            print(f'Train or val path: {self.__root}')
            assert os.path.exists(self.__splits_filepath)
            with open(self.__splits_filepath) as handle:
                self.__splits = json.load(handle)

        # The folders containing RGB and depth folder datasets
        self.__RGB_dir = os.path.join(self.__root, "RGB")
        if depth_type == "normal":
            self.__D_dir = os.path.join(self.__root, "Depth")
        elif depth_type == "binarised":
            self.__D_dir = os.path.join(self.__root, "Depth_Binarised")
        assert os.path.exists(self.__RGB_dir)
        assert os.path.exists(self.__D_dir)

        # Retrieve the number of classes from both of these
        self.__RGB_folders = DataUtils.allFoldersAtDir(self.__RGB_dir, exclude_list=self.__exclusion_cats)
        # self.__D_folders = DataUtils.allFoldersAtDir(self.__D_dir, exclude_list=self.__exclusion_cats)
        # assert len(self.__RGB_folders) == len(self.__D_folders) #JIng
        self.__num_classes = len(self.__RGB_folders) + len(self.__exclusion_cats)

        # The image size to resize to
        self.__img_size = img_size

        # The complete dictionary of filepaths
        self.__files = {'train': [], 'valid': [], 'test': []}

        # The dictionary of filepaths sorted by ID
        self.__sorted = {'train': {}, 'valid': {}, 'test': {}}

        # Whether to transform images to PyTorch form
        self.__transform = transform

        # For PyTorch, which device to use, GPU or CPU?
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__csv_filepath = 'C_GMM_SELECT_DIS.csv'
        """
        Class setup
        """

        with open(self.__csv_filepath, newline='') as file:
            reader = csv.reader(file, delimiter=";", quotechar='"')
            reader = list(reader)
            del reader[0]
            firstforder = []
            secondforder = []
            for i in range(len(reader)):
                m = reader[i][0].split(',')
                if m[2] != m[3]: # different folder
                    if int(m[2][:3]) <= folderLenCtl and int(m[3][:3]) <= folderLenCtl:
                        firstforder.append(m[2])
                        secondforder.append(m[3])

        self.__first = firstforder
        self.__second = secondforder

        self.__count = 0
        self.__batch_count = 0

        # Iterate through each category
        for current_RGB in (self.__RGB_folders):
            # Make sure we're inspecting the same category
            raw_ID = os.path.basename(current_RGB)
            # assert raw_ID == os.path.basename(current_D)
            tmp_train = {}
            tmp_test = {}
            tmp_val = {}
            # Find all the images within these folders
            for inter in os.listdir(current_RGB):
                RGB_paths = DataUtils.allFilesAtDirWithExt(os.path.join(current_RGB, inter), ".jpg")

                # There may be no validation files that get populated, have an empty array to add for this case
                valid_files = []

                # We're using a train/valid/test file with a single fold
                if self.__split_mode == "trainvalidtest":
                    # Create the long lists of all training and testing filenames for this fold
                    train_files = [{'class_ID': raw_ID, 'sub_ID': inter, 'filename': x} for x in
                                   self.__splits[raw_ID][inter]['train']]
                    valid_files = [{'class_ID': raw_ID, 'sub_ID': inter, 'filename': x} for x in
                                   self.__splits[raw_ID][inter]['valid']]
                    test_files = [{'class_ID': raw_ID, 'sub_ID': inter, 'filename': x} for x in
                                  self.__splits[raw_ID][inter]['test']]

                    self.__sorted['train'][raw_ID] = {}
                    self.__sorted['valid'][raw_ID] = {}
                    self.__sorted['test'][raw_ID] = {}
                    # Create the list of filenames sorted by category for this fold
                    tmp_train[inter] = list(self.__splits[raw_ID][inter]['train'])
                    tmp_test[inter] = list(self.__splits[raw_ID][inter]['test'])
                    tmp_val[inter] = list(self.__splits[raw_ID][inter]['valid'])

                self.__files['train'].extend(train_files)
                self.__files['valid'].extend(valid_files)
                self.__files['test'].extend(test_files)

            self.__sorted['train'][raw_ID] = tmp_train
            self.__sorted['valid'][raw_ID] = tmp_val
            self.__sorted['test'][raw_ID] = tmp_test
            # Populate the total list of files

        # Print some info
        if not suppress_info: self.printStats(extended=False)

    """
    Superclass overriding methods
    """

    def __len__(self):
        """
        Get the number of items for this dataset (depending on the split)
        """
        return len(self.__files[self.__split])

    def __getitem__(self, index):
        """
        Superclass overriding retrieval method for a particular index
        """

        # TODO: add augmentation possiblities


        if self.__batch_count%16 == 0:
            self.__chance = random.randint(1, 10)
        self.__batch_count += 1
        # Extract the anchor filename and the class it belongs to

        # old self extration
        if self.__chance > OldVsNew:

            anchor_filename = self.__files[self.__split][index]['filename']  # get any train file
            sub_id = self.__files[self.__split][index]['sub_ID']
            label_anchor = self.__files[self.__split][index]['class_ID']

            # Keep a copy of the original label
            label_anchor_orig = label_anchor
            sub_id_orig = sub_id
            # Convert to numpy form
            label_anchor = np.array([int(label_anchor)])
            sub_id = np.array([int(sub_id)])
            # Construct the full path to the image based on which type we'd like to retrieve

            img_anchor = self.__fetchImage(os.path.join(label_anchor_orig, sub_id_orig), anchor_filename)

            # Transform the anchor image and corresponding label if we're supposed to
            if self.__transform is not None:
                img_anchor = self.__transform(img_anchor)
            else:
                img_anchor = ImageUtils.npToTorch([img_anchor])[0]
            label_anchor = torch.from_numpy(label_anchor).long()

            # Otherwise we're in triplet mode
            img_pos = self.__retrievePositive(anchor_filename, label_anchor_orig, sub_id_orig)
            img_neg, label_neg, sub_neg, random_mark = self.__retrieveNegative(label_anchor_orig, sub_id_orig)
            label_neg = np.array([int(label_neg)])
            sub_neg = np.array([int(sub_neg)])
            random_mark = np.array([int(random_mark)])

            if self.__transform is not None:
                img_pos = self.__transform(Image.fromarray(img_pos))
                img_neg = self.__transform(Image.fromarray(img_neg))
            else:
                # Convert the positive and negative images
                img_pos, img_neg = ImageUtils.npToTorch([img_pos, img_neg])

                # Convert the negative label
            label_neg = torch.from_numpy(label_neg).long()
                #label_anchor2 = torch.tensor([500]) #test
                #sub_id2 = np.array([500])#test
        else:
            # New
            # if self.__retrieval_mode == "triplet":
            if self.__count >= len(self.__first):
                self.__count = 0
            i_f = self.__count
            label_anchor_orig = self.__first[i_f][0:3]
            sub_id_orig = self.__first[i_f][-1]
            l2 = self.__second[i_f][0:3]
            s2 = self.__second[i_f][-1]

            img_pos, img_pos2 = self.__retrieve2(label_anchor_orig, sub_id_orig, l2, s2)

            self.__count += 1
            # Load a random negative from a different class
            img_neg, label_neg, sub_neg, random_mark = self.__retrieveNegative(label_anchor_orig, sub_id_orig)
            if l2 == label_neg and s2 == sub_neg: # do twice incase get item from the same folder
                img_neg, label_neg, sub_neg, random_mark = self.__retrieveNegative(label_anchor_orig, sub_id_orig)
                if l2 == label_neg and s2 == sub_neg:
                    img_neg, label_neg, sub_neg, random_mark = self.__retrieveNegative(label_anchor_orig, sub_id_orig)

            # Convert label to numpy
            label_neg = np.array([int(label_neg)])
            sub_neg = np.array([int(sub_neg)])
            random_mark = np.array([int(random_mark)])
            sub_id = np.array([int(sub_id_orig), int(s2)])

            # Transform positive and negative into PyTorch friendly form
            img_anchor = img_pos2
            if self.__transform is not None:
                img_anchor = self.__transform(Image.fromarray(img_anchor))
                img_neg = self.__transform(Image.fromarray(img_neg))
                img_pos = self.__transform(Image.fromarray(img_pos))
            else:
                # Convert the positive and negative images
                img_pos, img_neg = ImageUtils.npToTorch([img_pos, img_neg])
                # Convert the negative label
                # positive sub  # neg sub
                img_anchor = ImageUtils.npToTorch([img_anchor])[0]

            label_anchor = torch.from_numpy(np.array([int(label_anchor_orig), int(l2)])).long()
            label_neg = torch.from_numpy(label_neg).long()

        return img_anchor, img_pos, img_neg, label_anchor, label_neg, sub_id, sub_neg, random_mark

    """
    Public methods
    """

    def printStats(self, extended=False):
        """Print statistics about this dataset"""
        print("__RGBDCows2020 Dataset___________________________________________________")
        print(f"Total number of categories: {self.__num_classes}")
        if self.__exclude_difficult:
            print(f"Removed {len(self.__exclusion_cats)} difficult categories: {self.__exclusion_cats}")
        images_str = f"Found {len(self.__files['train'])} training images, "
        images_str += f"{len(self.__files['valid'])} validation images, "
        images_str += f"{len(self.__files['test'])} testing images."
        print(images_str)
        print(f"Current fold: {self.__fold}, current split: {self.__split}")
        print(
            f"Image type: {self.__img_type}, retrieval mode: {self.__retrieval_mode}, split mode: {self.__split_mode}")
        print("_________________________________________________________________________")

        # We want some extended information about this set
        if extended:
            assert self.__sorted['train'].keys() == self.__sorted['test'].keys()
            for k in self.__sorted['train'].keys():
                # Compute the number of images for this class
                total_images = len(self.__sorted['train'][k]) + len(self.__sorted['test'][k])

                # Highlight which classes have fewer instances than the number of folds
                if total_images < len(self.__splits.keys()):
                    print(f"Class {k} has {total_images} images")

    """
    (Effectively) private methods
    """

    def __retrieve2(self, label_anchor, sub, l2, s2):
        # Copy the list of filenames for this category

        filenames = (self.__sorted[self.__split][label_anchor][sub])
        # Pick a random positive
        img_name = random.choice(filenames)
        filenames2 = (self.__sorted[self.__split][l2][s2])
        img_name2 = random.choice(filenames2)
        # Load the image based on the image type we're after
        return self.__fetchImage(label_anchor + '/' + sub, img_name), self.__fetchImage(l2 + '/' + s2, img_name2)

    # Retrieve a random positive from this class that isn't the anchor
    def __retrievePositive(self, anchor_filename, label_anchor, sub):
        # Copy the list of filenames for this category
        filenames = list(self.__sorted[self.__split][label_anchor][sub])  ###
        assert anchor_filename in filenames
        if len(filenames) < 2:
            print(f'{filenames} s folder has only one image')
        # Subtract the anchor path
        filenames.remove(anchor_filename)
        # Pick a random positive
        img_name = random.choice(filenames)
        # Load the image based on the image type we're after
        return self.__fetchImage(label_anchor + '/' + sub, img_name)

    # Retrieve a random negative instance from the current split set
    def __retrieveNegative(self, label_anchor, sub_id):
        # Copy the list of IDs
        # if self.__split !='train':
        #     print('self.__split:{}'.format( self.__split))
        sub_track = list(self.__sorted[self.__split][label_anchor].keys())
        if random.randint(1, 10) >= 7 and len(sub_track) > 1:
            sub_track.remove(sub_id)
            ran_sub = random.choice(random.choice(sub_track))
            category_seltec = label_anchor
            random_mark = 0
        else:
            IDs = list(self.__sorted[self.__split].keys())
            assert label_anchor in IDs
            # Subtract the anchor's ID
            IDs.remove(label_anchor)
            # Randomly select a category
            category_seltec = random.choice(IDs)
            random_neg = list(self.__sorted[self.__split][category_seltec].keys())
            ran_sub = random.choice(random_neg)
            # Randomly select a filename in that category
            random_mark = 1

        # if self.__split !='train':
        #     print(category_seltec, ran_sub)
        img_name = random.choice(self.__sorted[self.__split][category_seltec][ran_sub])
        # if self.__split != 'train':
        #     print('***',img_name)

        # Load the image based on the image type we're after
        return self.__fetchImage(os.path.join(category_seltec, ran_sub),
                                 img_name), category_seltec, ran_sub, random_mark

    # Fetch the specified image based on its type, category and filename
    def __fetchImage(self, category, filename):
        # We just want a standard RGB image
        if self.__img_type == "RGB":
            img_path = os.path.join(self.__RGB_dir, category, filename)
            return ImageUtils.loadImageAtSize(img_path, self.__img_size)


    # Helper function for produceLIME function in predicting on a batch of images
    def __batchPredict(self, batch):
        # Convert to PyTorch
        batch = batch[0, :, :, :].transpose(2, 0, 1)
        batch = torch.from_numpy(batch).float()
        batch = batch[None, :, :, :]

        # Put the batch on the GPU
        batch = batch.to(self.__device)

        # Get outputs on this batch
        with torch.no_grad():
            logits = self.__model(batch)

        # Get normalised probabilities from softmax
        probs = torch.nn.functional.softmax(logits, dim=1)

        return probs.detach().cpu().numpy()

    """
    Getters
    """

    def getNumClasses(self):
        return self.__num_classes

    def getDatasetPath(self):
        return self.__root

    def getSplitsFilepath(self):
        return self.__splits_filepath

    """
    Setters
    """


# Entry method/unit testing method
if __name__ == '__main__':
    """
    Testing or otherwise for the actual class object, training, testing
    """
    # Visualise the images and annotations via the PyTorch datasets object
    # (the same way they're accessed in a training or testing loop)
    # dataset = RGBDCows2020(	split_mode="day",
    # 						num_training_days=1,
    # 						num_testing_days=1,
    # 						retrieval_mode="single",
    # 						augment=False,
    # 						img_type="RGB",
    # 						suppress_info=False	)
    dataset = RGBDCows2020(fold=0, retrieval_mode="single", img_type="RGB")  # triplet
