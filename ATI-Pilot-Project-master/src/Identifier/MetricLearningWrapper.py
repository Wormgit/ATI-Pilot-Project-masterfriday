# Core libraries
import os
import sys
sys.path.append("../")
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# PyTorch
import torch
from torch.autograd import Variable
import argparse
# My libraries
from config import cfg
from Identifier.MetricLearning.models.embeddings import resnet50
from Utilities.ImageUtils import ImageUtils

class MetricLearningWrapper():
	"""
	This class wraps up the metric learning identifcation method for use in the full pipeline
	"""

	def __init__(self):
		""" Class constructor """

		# Base directory to find weights, embeddings, etc.
		if args.base_dir !='0':
			self.__base_dir = args.base_dir
		else:
			self.__base_dir = cfg.ID.ML_WEIGHTS_DIR

		# The maximum input image size
		self.__img_size = cfg.ID.IMG_SIZE

		# The size of the batch when processing multiple images
		#self.__batch_size = cfg.ID.BATCH_SIZE

		# The full path to find the weights for the network
		self.__weights_path = os.path.join(self.__base_dir, cfg.ID.ML_WEIGHTS_FILE)
		assert os.path.exists(self.__weights_path)

		# Create the model and load weights at the specified path
		self.__model = resnet50(ckpt_path=self.__weights_path, softmax_enabled=args.softmax_enabled,num_classes =args.num_classes)

		# Put the model on the GPU and in evaluation mode
		# self.__model.cuda()
		# self.__model.eval()

		# Path to find the embeddings used as the "training" set in k-NN
		self.__embeddings_path0 = os.path.join(self.__base_dir, cfg.ID.ML_EMBED_0)
		self.__embeddings_path1 = os.path.join(self.__base_dir, cfg.ID.ML_EMBED_1)
		assert os.path.exists(self.__embeddings_path0)
		assert os.path.exists(self.__embeddings_path1)

		# Load the embeddings into memory
		file0 = np.load(self.__embeddings_path0)
		file1 = np.load(self.__embeddings_path1)

		# Extract the embeddings and corresponding labels
		embeddings = np.concatenate((file0['embeddings'][1:], file1['embeddings'][1:]))
		labels = np.concatenate((file0['labels'][1:], file1['labels'][1:]))

		# Create the KNN-based classifier
		self.__classifier = KNeighborsClassifier(n_neighbors=cfg.ID.K, n_jobs=-4)

		# Fit these values to the classifier
		self.__classifier.fit(embeddings, labels)

		# Device for tensors
		self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	"""
	Public methods
	"""

	def predictBatch(self, images, visualise=False):
		""" Predict on a batch (list) of images """

		# We don't need the autograd engine
		with torch.no_grad():
			# Rezie and transform the images into PyTorch form
			images = ImageUtils.npToTorch(images, return_tensor=True, resize=self.__img_size)

			# Put the images on the GPU and express them as a PyTorch variable
			#images = images.to(self.__device)
			m = (images.size())
			# Get the predictions on this batch
			outputs = self.__model(images)

			# Express the embeddings in numpy form
			embeddings = outputs.data
			embeddings = embeddings.detach().cpu().numpy()

			# Classify the output embeddings using k-NN
			predictions = self.__classifier.predict(embeddings).astype(int)

			# Visualise the predictions if we're supposed to
			if visualise:
				# Convert the images back to numpy
				np_images = images.detach().cpu().numpy().astype(np.uint8)

				# Iterate through each image and display it
				for i in range(np_images.shape[0]):
					# Extract the image
					disp_img = np_images[i,:,:,:]

					# Transpose it back to HWC from CWH
					disp_img = disp_img.transpose(1, 2, 0)

					cv2.imshow(f"Prediction = {predictions[i]}", disp_img)
					print(f"Prediction = {predictions[i]}")
					cv2.waitKey(0)

		return predictions

# Entry method/unit testing method
if __name__ == '__main__':
	# Create an instance and test on a batch of images

	parser = argparse.ArgumentParser(description='Params')
	parser.add_argument('--base_dir', nargs='?', default='/home/io18230/Desktop/w')
	parser.add_argument('--test_dir', nargs='?',
						default="/home/io18230/Desktop/RGBDCows2020/Identification/RGB")
	parser.add_argument('--save_path', type=str, default='/home/io18230/Desktop',
						help="Where to store the embeddings and statistics")
	parser.add_argument('--softmax_enabled', type=int, default=0)
	parser.add_argument('--num_classes', type=int, default=9)
	args = parser.parse_args()

	identifier = MetricLearningWrapper()
	i = 0
	list = []
	label = []
	correct = 0
	total = 0
	for items in os.listdir(args.test_dir):
		if items == '003':
			print('end:003')
			break
		for image in os.listdir(os.path.join(args.test_dir, items)):
			print(items, image)
			img1 = cv2.imread(os.path.join(args.test_dir, items, image))
			i +=1
			list.append(img1)
			label.append(int(items))
			if i == 4:
				p=identifier.predictBatch([list[0],list[1],list[2],list[3]], visualise=0)
				correct += (p == np.array(label)).sum()
				total += 4
				list = []
				label = []
				i = 0

	accuracy = (float(correct) / total) * 100
	print(correct,total,accuracy)




	# img1 = cv2.imread(os.path.join(base_dir, "001/r_000010_00_4Wct_0533_0212_30.75.jpg")) # Class 114
	# #img1 = cv2.imread(os.path.join(base_dir, "001/r_000010_00_4Wct_0533_0212_30.75.jpg")) # Class 114
	# img0 = cv2.imread(os.path.join(base_dir, "005/r_000015_00_0Tct_0377_0224_10.31.jpg")) # Class 96
	# identifier.predictBatch([img0, img1, img0, img1, img0, img1, img0, img1], visualise=True)


