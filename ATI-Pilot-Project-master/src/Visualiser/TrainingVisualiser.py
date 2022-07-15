# Core libraries
import os
import sys
sys.path.append(os.path.abspath("../"))
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

# My libraries
from Utilities.DataUtils import DataUtils

class TrainingVisualiser(object):
	# Class constructor
	def __init__(self):
		pass

	"""
	Public methods
	"""

	"""
	(Effectively) private methods
	"""

	"""
	Staticmethods
	"""

	@staticmethod
	def visualiseTrainingGraph(epochs, train_loss, val_loss, train_acc, val_acc):
		fig, ax1 = plt.subplots()
		colour1 = 'tab:blue'
		ax1.set_xlabel('Epochs')
		ax1.set_ylabel('Accuracy', color=colour1)
		ax1.set_xlim((0, np.max(epochs)))
		ax1.set_ylim((0.,1.))
		ax1.plot(epochs, val_acc, color=colour1)

		ax2 = ax1.twinx()
		colour2 = 'tab:red'
		ax2.set_ylabel('Loss', color=colour2)
		ax2.set_ylim((0., 3.))
		ax2.plot(epochs, val_loss, color=colour2)

		plt.tight_layout()
		plt.show()
		# plt.savefig()

	@staticmethod
	def visualiseFromPickleFile(file_path):
		print(f"Loading data from file: {file_path}")

		# Load the data
		with open(file_path, 'rb') as handle:
			data = pickle.load(handle)

		# Add epoch data
		epochs = np.arange(0, 100)

		# Extract arrays from this
		train_loss = np.array(data['loss']['train'])
		val_loss = np.array(data['loss']['val'])
		train_acc = np.array(data['acc']['train'])
		val_acc = np.array(data['acc']['val'])

		# Render a graph from this
		graph = TrainingVisualiser.visualiseTrainingGraph(epochs, train_loss, val_loss, train_acc, val_acc)

	@staticmethod
	def visualiseFromNPZFile(file_path, acc_path, acc_path2, acc_path3, acc_path_train):
		print(f"Loading data from file: {file_path}")

		Train_epoch = 100
		Train_step  = 20
		epoch_validation_gap = 4

		if os.path.exists(file_path):
			with np.load(file_path) as data:
				train_loss = data['losses_mean']
				train_steps = data['loss_steps']
				mmval_acc = data['accuracies']/100    # original code we do not use it
				val_steps = data['accuracy_steps']
				train_size = int(len(train_steps) /Train_epoch)


				Epoch_trainloss = []
				#count_train_size = 0
				for i in range(Train_epoch):
					fir = train_size * i
					las = train_size*(i+1) - 1
					ave = np.mean(train_loss[fir:las])
					Epoch_trainloss.append(ave)

				Epoch_train_steps = np.arange(Train_step * train_size, np.max(train_steps) + Train_step * train_size, Train_step * train_size)
				Epoch_train_steps = Epoch_train_steps[:len(Epoch_trainloss)]

		if os.path.exists(root_dir+"/valLoss_All_mean.npz"):
			with np.load(root_dir+"/valLoss_All_mean.npz") as data:
				valLoss = data['valLoss_All_mean']

			if val_steps.shape[0] == 0:
				step_size = round(float(np.max(train_steps)) /valLoss.shape[0])
				val_steps = np.arange(0, np.max(train_steps), step_size)

		if os.path.exists(acc_path):
			with np.load(acc_path) as bb:
				val_acc1 = bb['acc_test_folder']/100
				val_acc_ari1 = bb['ARI_test_folder']
				val_acc2 = bb['acc_test_knn']/100
				val_acc_ari2 = bb['ARI_test_knn']
				if len(val_acc2) > len(val_acc1):
					val_acc = val_acc2
					val_acc_ari =val_acc_ari2
				else:
					val_acc = val_acc1
					val_acc_ari = val_acc_ari1

		if os.path.exists(acc_path2):
			with np.load(acc_path2) as bb:
				val_acc1 = bb['acc_test_folder']/100
				val_acc_ari1 = bb['ARI_test_folder']
				val_acc2 = bb['acc_test_knn']/100
				val_acc_ari2 = bb['ARI_test_knn']
				if len(val_acc2) > len(val_acc1):
					val_acc22 = val_acc2
					val_acc_ari22 =val_acc_ari2
				else:
					val_acc22 = val_acc1
					val_acc_ari22 = val_acc_ari1

		if os.path.exists(acc_path3):
			with np.load(acc_path3) as bb:
				val_acc1 = bb['acc_test_folder']/100
				val_acc_ari1 = bb['ARI_test_folder']
				val_acc2 = bb['acc_test_knn']/100
				val_acc_ari2 = bb['ARI_test_knn']
				if len(val_acc2) > len(val_acc1):
					val_acc3 = val_acc2
					val_acc_ari3 =val_acc_ari2
				else:
					val_acc3 = val_acc1
					val_acc_ari3 = val_acc_ari1

		if os.path.exists(acc_path_train):
			with np.load(acc_path_train) as aa:
				t_acc1 = aa['acc_test_folder']/100
				t_acc_ari1 = aa['ARI_test_folder']
				t_acc2 = aa['acc_test_knn']/100
				t_acc_ari2 = aa['ARI_test_knn']
				if len(t_acc2) > len(t_acc1):
					t_acc = t_acc2
					t_acc_ari =t_acc_ari2
				else:
					t_acc = t_acc1
					t_acc_ari = t_acc_ari1

			best_epoch_acc = np.argmax(val_acc)*epoch_validation_gap
			best_epoch_ari = np.argmax(val_acc_ari)*epoch_validation_gap
			print(f"Best accuracy = {np.max(val_acc)}, Epoch{best_epoch_acc}")
			print(f"Best ARI = {np.max(val_acc_ari)}, Epoch {best_epoch_ari}")
			print(val_acc)

		#max_steps = max(np.max(val_steps), np.max(train_steps))
		max_steps = np.max(train_steps)

		#plot accuracy
		fig, ax1 = plt.subplots()
		colour1 = 'tab:blue'
		ax1.set_xlabel('Steps')
		ax1.set_ylabel('Accuracy', color=colour1)
		ax1.set_xlim((0, max_steps*1.35))
		ax1.set_ylim((0.,1.))


		if os.path.exists(acc_path):
			#ax1.plot(val_steps, val_acc, color=colour1, label='Test Acc', linestyle='-')
			ax1.plot(val_steps, val_acc_ari, color='tab:cyan', label='Test ARI 20')  #tab:cyan orange
			ax1.plot(val_steps, val_acc_ari3, color='tab:gray', label='Test ARI ??')
		if os.path.exists(acc_path2):
			ax1.plot(val_steps, val_acc_ari22, color='tab:olive', label='Test ARI 50')

		ax1.plot(val_steps, t_acc_ari, color=colour1, label='Train Acc 130')
		#plt.scatter(val_steps[np.argmax(val_acc)], 1.02, markerfacecolor='none')
		if os.path.exists(acc_path):
			pass
			#ax1.text(val_steps[np.argmax(val_acc)], 1.02, f'Best ACC Epoch {best_epoch_acc}', fontsize=10)
			#ax1.text(val_steps[np.argmax(val_acc_ari)], 0.7, f'Best ARI Epoch {best_epoch_ari}', fontsize=10)
			#plt.plot(val_steps[np.argmax(val_acc)], np.max(val_acc), 'o', color ='r',markersize=12,markerfacecolor='none')
			#plt.plot(val_steps[np.argmax(val_acc_ari)], np.max(val_acc_ari), 'o', color ='r', markersize=12, markerfacecolor='none')
		plt.legend(loc="upper right")

		#plot loss
		ax2 = ax1.twinx()
		colour2 = 'tab:red'
		ax2.set_ylabel('Loss', color='tab:orange')
		ax2.set_ylim((0., 5.))
		#ax2.plot(train_steps, train_loss, color='tab:green', label='Train loss')
		ax2.plot(Epoch_train_steps, Epoch_trainloss, color='tab:orange', label='Train loss')
		if os.path.exists("/home/io18230/Desktop/valLoss_All_mean.npz"):
			ax2.plot(val_steps, valLoss[:], color=colour2, label='Val loss')
		plt.legend(loc="lower right")

		plt.tight_layout()
		plt.show()

	@staticmethod
	def visualiseKGridSearch():
		file_path = "D:\\Work\\ATI-Pilot-Project\\src\\Identifier\\MetricLearning\\output\\k_grid_search.json"

		with open(file_path, 'r') as handle:
			data = json.load(handle)

		X = np.array(list(data.keys())).astype(int)
		Y = np.array([data[x] for x in data.keys()])

		plt.figure()
		plt.plot(X, Y)
		plt.ylabel("Accuracy (%)")
		plt.xlabel("k")
		plt.xlim((0,np.max(X)))
		plt.tight_layout()
		# plt.show()
		plt.savefig("k-grid-search.pdf")

# Entry method/unit testing method
if __name__ == '__main__':
	# Root dir
	# root_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\RGBDCows2020\\Results\\ClosedSet\\RGBD"
	# files = DataUtils.allFilesAtDirWithExt(root_dir, ".pkl")
	# for file in files:
	# 	if "_data_logs.pkl" in file:
	# 		TrainingVisualiser.visualiseFromPickleFile(file)

	# root_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\RGBDCows2020\\Results\\LatentSpace"
	# img_type = "RGBD"
	# fold = 9
	# file_path = os.path.join(root_dir, img_type, f"fold_{fold}", "logs.npz")
	# TrainingVisualiser.visualiseFromNPZFile(file_path)

	#TrainingVisualiser.visualiseKGridSearch()
	root_dir = '/home/io18230/Desktop/tmmmpid/'
	file_path = os.path.join(root_dir, "logs.npz")
	acc_path3 = os.path.join(root_dir, "acc.npz")
	acc_path = os.path.join(root_dir, "acc20.npz")
	acc_path_train = os.path.join(root_dir, "acc_train.npz")
	acc_path2 = os.path.join(root_dir, "acc50.npz")


	TrainingVisualiser.visualiseFromNPZFile(file_path, acc_path, acc_path2, acc_path3, acc_path_train)