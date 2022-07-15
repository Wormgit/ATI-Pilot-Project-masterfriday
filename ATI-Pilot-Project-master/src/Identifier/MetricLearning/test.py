# Core libraries
import os
import sys
import cv2
import json
import torch
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

	# Load the relevant datasets
	train_dataset = Utilities.selectDataset(args, "train")
	test_dataset = Utilities.selectDataset(args, "test")
	assert train_dataset.getNumClasses() == test_dataset.getNumClasses()

	# Load the validation set too if we're supposed to
	if args.split == "trainvalidtest":
		valid_dataset = Utilities.selectDataset(args, "valid")
	if not args.class_num:
		num_classes = train_dataset.getNumClasses()
	else:
		num_classes = args.class_num
	# Define our embeddings model
	model = resnet50(	pretrained=True,
						num_classes=num_classes,
						ckpt_path=args.model_path,
						embedding_size=args.embedding_size,
						img_type=args.img_type,
						softmax_enabled=args.softmax_enabled	)

	# Put the model on the GPU and in evaluation mode
	#model.cuda()
	model.eval()

	# Get the embeddings and labels of the training set and testing set
	if args.load_embdings:
		embeddings = np.load(args.embeddings_file)
		train_embeddings = embeddings['embeddings']
		train_labels = embeddings['labels']
	else:
		train_embeddings, train_labels = inferEmbeddings(args, model, train_dataset, "train")

	if not args.only_train:
		f_embeddings, f_labels, name = inferEmbedding_from_folder(args, model)

		f_accuracy, f_preds = KNNAccuracy(train_embeddings, train_labels, f_embeddings, f_labels)
		data1 = pd.DataFrame({'image_path': name, 'pre_label': f_preds})
		data1.to_csv(args.save_path +'/label.csv')

		if args.save_embeddings:
			# Construct the save path
			save_path = os.path.join(args.save_path, f"folder_embeddings.npz")
			# Save the embeddings to a numpy array
			np.savez(save_path, embeddings=f_embeddings, labels= f_preds, path = name)
			sys.stdout.write(f"folder accuracy={str(f_accuracy)} \n")

	if args.use_valid:
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
			sys.stdout.write(f"Validation accuracy={str(valid_accuracy)}; Testing accuracy={str(test_accuracy)}\n")

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

def kGridSearch(args):
	""" Perform a grid search for k nearest neighbours

	Example command:
	python test.py --model_path="D:\Work\results\CEiA\SRTL\05\rep_0\triplet_cnn_open_cows_best_x1.pkl" --mode="gridsearch" --dataset="OpenSetCows2020" --save_path="output"

	"""

	# Which fold / repitition to use
	args.repeat_num = 0

	# What is the ratio of unknown classes
	args.unknown_ratio = 0.5

	# Load the relevant datasets
	train_dataset = Utilities.selectDataset(args, "train")
	test_dataset = Utilities.selectDataset(args, "valid")
	assert train_dataset.getNumClasses() == test_dataset.getNumClasses()

	# Define our embeddings model
	model = resnet50(	pretrained=True,
						num_classes=train_dataset.getNumClasses(),
						ckpt_path=args.model_path,
						embedding_size=args.embedding_size,
						img_type=args.img_type,
						softmax_enabled=args.softmax_enabled	)

	# Put the model on the GPU and in evaluation mode
	model.cuda()
	model.eval()

	# Get the embeddings and labels of the training set and testing set
	train_embeddings, train_labels = inferEmbeddings(args, model, train_dataset, "train")
	test_embeddings, test_labels = inferEmbeddings(args, model, test_dataset, "valid")

	# Dict to store results
	results = {}

	# Iterate from 1 to the number of testing instances
	for k in tqdm(range(1, len(test_dataset))):
		# Classify the test set
		accuracy, _ = KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels, n_neighbors=k)

		# Save the accuracy for this k
		results[k] = accuracy

		# print(f"Accuracy = {accuracy} for {k}-NN classification")

	# Save them to file
	with open(os.path.join(args.save_path, f"k_grid_search.json"), 'w') as handle:
		json.dump(results, handle, indent=4)

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

	# How many were correct?
	correct = (predictions == test_labels).sum()

	# Compute accuracy
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
	for images, _, _, labels, _, _,_,_ in tqdm(data_loader, desc=f"Inferring {split} embeddings"):
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
			label.append(int(items))
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

# Main/entry method
if __name__ == '__main__':
	# Collate command line arguments
	parser = argparse.ArgumentParser(description='Params')

	# Required arguments
	parser.add_argument('--model_path', nargs='?', default='/home/io18230/Desktop/will_trained_186/best_model_state.pkl', #w 22
						help='Path to the saved model to load weights from')
	parser.add_argument('--embeddings_file', nargs='?', default='/home/io18230/Desktop/22/train_embeddings.npz', #w 22
						help='Path to the saved model to load weights from')
	parser.add_argument('--save_path', type=str, default='/home/io18230/Desktop',
						help="Where to store the embeddings and statistics")
	parser.add_argument('--split', type=str, default="trainvalidtest",
						help="Which evaluation mode to use: [trainvalidtest, traintest]")
	parser.add_argument('--mode', type=str, default="evaluate",
						help="Which mode are we in: [evaluate, gridsearch]")
	parser.add_argument('--dataset', nargs='?', type=str, default='RGBDCows2020',
						help='Which dataset to use: [RGBDCows2020, OpenSetCows2020]')
	parser.add_argument('--exclude_difficult', type=int, default=0,
						help='Whether to exclude difficult classes from the loaded dataset')
	parser.add_argument('--batch_size', nargs='?', type=int, default=16,
						help='Batch size for inference')
	parser.add_argument('--embedding_size', nargs='?', type=int, default=128,
						help='Size of the dense layer for inference')
	parser.add_argument('--current_fold', type=int, default=0,
						help="The current fold we'd like to test on")
	parser.add_argument('--repeat_num', type=int, default=-1,
						help="The repeat number we're on")
	parser.add_argument('--unknown_ratio', type=float, default=-1.0,
						help="The current ratio of unknown classes")
	parser.add_argument('--save_embeddings', type=int, default=1,
						help="Should we save the embeddings to file")
	parser.add_argument('--img_type', type=str, default="RGB",
						help="Which image type should we retrieve: [RGB, D, RGBD]")
	parser.add_argument('--softmax_enabled', type=int, default=0,
						help="Whether softmax was enabled when training the model")
	parser.add_argument('--test_dir', nargs='?',
						default="/home/io18230/Desktop/RGBDCows2020xx/Identification/RGB")
	parser.add_argument('--use_valid', type=int, default=1)
	parser.add_argument('--class_num', type=int, default=0)
	parser.add_argument('--load_embdings', type=int, default=0)
	parser.add_argument('--only_train', type=int, default=1)



	# Parse them
	args = parser.parse_args()

	if args.mode == "evaluate":
		evaluateModel(args)
	elif args.mode == "gridsearch":
		kGridSearch(args)
	else:
		print(f"Mode not recognised, exiting.")
		sys.exit(1)