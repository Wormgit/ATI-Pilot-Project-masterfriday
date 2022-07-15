#!/usr/bin/env python

# Core libraries
import os
import sys
sys.path.append("../")
import cv2
import json
import pickle
import shutil
import random
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

class RGBDCows2020(data.Dataset):
	"""
	This Class manages everything to do with the RGBDCows2020 dataset and interfacing with it in PyTorch
	"""

	def __init__(	self,
					split_mode="trainvalidtest",
					fold=0,
					num_training_days=-1,
					num_testing_days=-1,
					split="train",				# train, test or valid
					img_type="RGB",				# D, RGB or RGBD
					retrieval_mode="single",	# Retrieve triplets or single images
					depth_type="normal",		# What type of depth image do we want (normal or binarised)
					img_size=(224, 224),		# Resolution to get images at
					augment=False,				# Whether to randomly augment images (only works for training)
					transform=False,			# Transform images to PyTorch form
					suppress_info=False,		# Suppress printing some info about this dataset
					exclude_difficult=False 	# Whether to exclude difficult categories
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
		if self.__exclude_difficult: self.__exclusion_cats = ["054", "069", "073", "173"]
		else: self.__exclusion_cats = []

		# If we're in triplet mode, remove animal 182 as it only has two instances and when split into train/valid/test
		# causes issues with finding positives and negatives
		if self.__retrieval_mode == "triplet":
			self.__exclusion_cats.append("182")

		# The fold we're currently using for train/test splits (if in random mode)
		self.__fold = str(fold)

		# We're splitting into train/valid/test (single fold for the time being)
		if self.__split_mode == "trainvalidtest":
			self.__splits_filepath = os.path.join(self.__root, "single_train_valid_test_splits.json")
			assert os.path.exists(self.__splits_filepath)
			with open(self.__splits_filepath) as handle:
				self.__splits = json.load(handle)


		# The folders containing RGB and depth folder datasets
		self.__RGB_dir = os.path.join(self.__root, "RGB")
		if depth_type == "normal": self.__D_dir = os.path.join(self.__root, "Depth")
		elif depth_type == "binarised": self.__D_dir = os.path.join(self.__root, "Depth_Binarised")
		assert os.path.exists(self.__RGB_dir)
		assert os.path.exists(self.__D_dir)

		# Retrieve the number of classes from both of these
		self.__RGB_folders = DataUtils.allFoldersAtDir(self.__RGB_dir, exclude_list=self.__exclusion_cats)
		self.__D_folders = DataUtils.allFoldersAtDir(self.__D_dir, exclude_list=self.__exclusion_cats)
		assert len(self.__RGB_folders) == len(self.__D_folders)
		self.__num_classes = len(self.__RGB_folders) + len(self.__exclusion_cats)

		# The image size to resize to
		self.__img_size = img_size

		# The complete dictionary of filepaths
		self.__files = {'train': [], 'valid':[], 'test': []}

		# The dictionary of filepaths sorted by ID
		self.__sorted = {'train': {}, 'valid':{}, 'test': {}}

		# Whether to transform images to PyTorch form
		self.__transform = transform

		# For PyTorch, which device to use, GPU or CPU?
		self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		"""
		Class setup
		"""

		# Iterate through each category
		for current_RGB, current_D in zip(self.__RGB_folders, self.__D_folders):
			# Make sure we're inspecting the same category
			raw_ID = os.path.basename(current_RGB)
			assert raw_ID == os.path.basename(current_D)

			# Find all the images within these folders
			for inter in os.listdir(current_RGB):
				RGB_paths = DataUtils.allFilesAtDirWithExt(os.path.join(current_RGB,inter), ".jpg")
				D_paths = DataUtils.allFilesAtDirWithExt(os.path.join(current_RGB,inter), ".jpg")
				assert len(RGB_paths) == len(D_paths)

				# There may be no validation files that get populated, have an empty array to add for this case
				valid_files = []

				# We're using a train/valid/test file with a single fold
				if self.__split_mode == "trainvalidtest":
					# Create the long lists of all training and testing filenames for this fold
					train_files = [{'class_ID': raw_ID, 'sub_ID': inter, 'filename': x} for x in self.__splits[raw_ID][inter]['train']]
					valid_files = [{'class_ID': raw_ID, 'sub_ID': inter, 'filename': x} for x in self.__splits[raw_ID][inter]['valid']]
					test_files =  [{'class_ID': raw_ID, 'sub_ID': inter, 'filename': x} for x in self.__splits[raw_ID][inter]['test']]

					self.__sorted['train'][raw_ID] = {}
					self.__sorted['valid'][raw_ID] = {}
					self.__sorted['test'][raw_ID] = {}
					# Create the list of filenames sorted by category for this fold
					self.__sorted['train'][raw_ID][inter] = list(self.__splits[raw_ID][inter]['train'])
					self.__sorted['valid'][raw_ID][inter] = list(self.__splits[raw_ID][inter]['valid'])
					self.__sorted['test'][raw_ID][inter] = list(self.__splits[raw_ID][inter]['test'])

				# We're in day split mode, populate differently

			# Populate the total list of files
			self.__files['train'].extend(train_files)
			self.__files['valid'].extend(valid_files)
			self.__files['test'].extend(test_files)

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

		# Extract the anchor filename and the class it belongs to
		anchor_filename = self.__files[self.__split][index][0]['filename'] # get any train file
		label_anchor = self.__files[self.__split][index]['class_ID']
		print (self.__files[self.__split][1]['filename'])
		#print ('index',index)

		# Keep a copy of the original label
		label_anchor_orig = label_anchor

		# Convert to numpy form
		label_anchor = np.array([int(label_anchor)])

		# Construct the full path to the image based on which type we'd like to retrieve
		img_anchor = self.__fetchImage(label_anchor_orig, anchor_filename)

		# Transform the anchor image and corresponding label if we're supposed to
		if self.__transform:
			img_anchor = ImageUtils.npToTorch([img_anchor])[0]
			label_anchor = torch.from_numpy(label_anchor).long()
		# If we're in single image retrieval mode, stop here and return

		if self.__retrieval_mode == "single": 
			print('single',anchor_filename)
			return img_anchor, label_anchor, anchor_filename

		# Otherwise we're in triplet mode
		elif self.__retrieval_mode == "triplet":
			# If we're retrieving the test or validation set, no need to bother finding a positive and negative
			if self.__split == "test" or self.__split == "valid": 
				return img_anchor, [], [], label_anchor, []

			# Load another random positive from this class
			img_pos = self.__retrievePositive(anchor_filename, label_anchor_orig)

			# Load a random negative from a different class
			img_neg, label_neg = self.__retrieveNegative(label_anchor_orig)

			# Convert label to numpy
			label_neg = np.array([int(label_neg)])

			# Transform positive and negative into PyTorch friendly form
			if self.__transform:
				# Convert the positive and negative images
				img_pos, img_neg = ImageUtils.npToTorch([img_pos, img_neg])

				# Convert the negative label
				label_neg = torch.from_numpy(label_neg).long()

			return img_anchor, img_pos, img_neg, label_anchor, label_neg

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
		print(f"Image type: {self.__img_type}, retrieval mode: {self.__retrieval_mode}, split mode: {self.__split_mode}")
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

	# Visualise the images and labels one by one, transform needs to be false for
	# this function to work
	def visualise(self, shuffle=True):
		# Transform to PyTorch form needs to be false
		assert not self.__transform

		# PyTorch data loader
		trainloader = data.DataLoader(self, batch_size=1, shuffle=shuffle)

		# Visualise differently based on which retrieval mode we're in
		if self.__retrieval_mode == "single":
			for img, label, filename in trainloader:
				# Convert the label to the string format we're used to
				label_str = str(label.numpy()[0][0]).zfill(3)
				print(f"Class is: {label_str} for filename: {filename[0]}")

				# We just have a single image to display
				if self.__img_type == "RGB" or self.__img_type == "D":
					# Convert image from tensor to numpy
					disp_img = img[0].numpy().astype(np.uint8)

					# Display the image
					cv2.imshow("Anchor image", disp_img)

				# We have a RGB and depth image to display
				elif self.__img_type == "RGBD":
					# Convert image from tensor to numpy and extract RGB and depth components
					RGBD_img = img[0].numpy().astype(np.uint8)
					RGB_img = RGBD_img[:,:,:3]
					D_img = np.zeros(RGB_img.shape, dtype=np.uint8)
					D_img[:,:,0] = RGBD_img[:,:,3]
					D_img[:,:,1] = RGBD_img[:,:,3]
					D_img[:,:,2] = RGBD_img[:,:,3]

					# Concatenate the images vertically
					disp_img = np.concatenate((RGB_img, D_img), axis=0)
					cv2.imshow("Anchor RGB and D image", disp_img)

				# Wait for user keypress
				cv2.waitKey(0)

		# Visualise triplets
		elif self.__retrieval_mode == "triplet":
			for img, img_pos, img_neg, label, label_neg in trainloader:
				# Convert the labels to the string format
				str_lab = str(label.numpy()[0][0]).zfill(3)
				str_neg = str(label_neg.numpy()[0][0]).zfill(3)
				print(f"Anchor class: {str_lab}, negative: {str_neg}")

				# We just have RGB or D image to display
				if self.__img_type == "RGB" or self.__img_type == "D":
					# Convert images from tensor to numpy
					disp_anc = img[0].numpy()
					disp_pos = img_pos[0].numpy()
					disp_neg = img_neg[0].numpy()

				# We need to separate RGB and D
				elif self.__img_type == "RGBD":
					# Convert images from tensor to numpy
					RGBD_anc = img[0].numpy()
					RGBD_pos = img_pos[0].numpy()
					RGBD_neg = img_neg[0].numpy()

					# Get the RGB components
					RGB_anc = RGBD_anc[:,:,:3]
					RGB_pos = RGBD_pos[:,:,:3]
					RGB_neg = RGBD_neg[:,:,:3]

					# Create the D components
					D_anc = np.zeros(RGB_anc.shape, dtype=np.uint8)
					D_pos = np.zeros(RGB_pos.shape, dtype=np.uint8)
					D_neg = np.zeros(RGB_neg.shape, dtype=np.uint8)

					# Copy array slices across to each
					D_anc[:,:,0] = RGBD_anc[:,:,3] ; D_anc[:,:,1] = RGBD_anc[:,:,3] ; D_anc[:,:,2] = RGBD_anc[:,:,3]
					D_pos[:,:,0] = RGBD_pos[:,:,3] ; D_pos[:,:,1] = RGBD_pos[:,:,3] ; D_pos[:,:,2] = RGBD_pos[:,:,3]
					D_neg[:,:,0] = RGBD_neg[:,:,3] ; D_neg[:,:,1] = RGBD_neg[:,:,3] ; D_neg[:,:,2] = RGBD_neg[:,:,3]

					# Vertically concatenate RGB and D components
					disp_anc = np.concatenate((RGB_anc, D_anc), axis=0)
					disp_pos = np.concatenate((RGB_pos, D_pos), axis=0)
					disp_neg = np.concatenate((RGB_neg, D_neg), axis=0)

				# Concatenate the images into one image
				disp_img = np.concatenate((disp_anc, disp_pos, disp_neg), axis=1)
				cv2.imshow("Anchor, positive and negative images", disp_img)
				cv2.waitKey(0)


	"""
	(Effectively) private methods
	"""

	# Retrieve a random positive from this class that isn't the anchor
	def __retrievePositive(self, anchor_filename, label_anchor):
		# Copy the list of filenames for this category
		filenames = list(self.__sorted["train"][label_anchor])
		assert anchor_filename in filenames

		# Subtract the anchor path
		filenames.remove(anchor_filename)

		# Pick a random positive
		img_name = random.choice(filenames)

		# Load the image based on the image type we're after
		return self.__fetchImage(label_anchor, img_name)

	# Retrieve a random negative instance from the current split set
	def __retrieveNegative(self, label_anchor):
		# Copy the list of IDs
		IDs = list(self.__sorted["train"].keys())
		assert label_anchor in IDs

		# Subtract the anchor's ID
		IDs.remove(label_anchor)

		# Randomly select a category
		random_category = random.choice(IDs)

		# Randomly select a filename in that category
		img_name = random.choice(self.__sorted[self.__split][random_category])

		# Load the image based on the image type we're after
		return self.__fetchImage(random_category, img_name), random_category

	# Fetch the specified image based on its type, category and filename
	def __fetchImage(self, category, filename):
		# We want a 4-channel RGBD image
		if self.__img_type == "RGBD":
			# Construct the full paths to the RGB and Depth images
			RGB_path = os.path.join(self.__RGB_dir, category, filename)
			D_path = os.path.join(self.__D_dir, category, filename)

			# Load them both as RGB images
			RGB_img = ImageUtils.loadImageAtSize(RGB_path, self.__img_size)
			D_img = ImageUtils.loadImageAtSize(D_path, self.__img_size)

			# flatten D_img to single channel (should currently be 3 equal greyscale channels)
			assert np.array_equal(D_img[:,:,0], D_img[:,:,1])
			assert np.array_equal(D_img[:,:,1], D_img[:,:,2])
			D_img = D_img[:,:,0]

			# Combine into one 4-channel RGBD np array
			RGBD_img = np.concatenate((RGB_img, np.expand_dims(D_img, axis=2)), axis=2)

			return RGBD_img

		# We just want a standard RGB image
		elif self.__img_type == "RGB":
			img_path = os.path.join(self.__RGB_dir, category, filename)
			return ImageUtils.loadImageAtSize(img_path, self.__img_size)

		# We want just the depth image
		elif self.__img_type == "D":
			img_path = os.path.join(self.__D_dir, category, filename)
			return ImageUtils.loadImageAtSize(img_path, self.__img_size)

	# Helper function for produceLIME function in predicting on a batch of images
	def __batchPredict(self, batch):
		# Convert to PyTorch
		batch = batch[0,:,:,:].transpose(2, 0, 1)
		batch = torch.from_numpy(batch).float()
		batch = batch[None,:,:,:]

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
	dataset = RGBDCows2020(fold=0, retrieval_mode="triplet", img_type="RGB")
	#dataset.visualise(shuffle=True)

