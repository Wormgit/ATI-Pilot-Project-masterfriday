# Core libraries
import os
import sys
sys.path.append(os.path.abspath("../"))
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from textwrap import fill


# My libraries

class TrainingVisualiser(object):
	# Class constructor
	def __init__(self):
		pass

	@staticmethod
	def visualiseFromNPZFile(log_path, acc_path):
		print(f"Loading data from file: {log_path}")

		epoch_validation_gap = 2
		show_val_acc = 1
		Train_epoch = 61

		y_axi_loss = 10
		m_frontsize = 16
		line = 2.5
		bias = 1 #if find epoch number, set it to zero.
		if os.path.exists(root_dir+"/valLoss_1st.npz"):
			with np.load(root_dir+"/valLoss_1st.npz") as data:
				valLoss_stp = data['valLoss_1stStepN']
				valLoss_1st = data['valLoss_1st_single']
			with np.load(root_dir+"/TrainLoss_1st.npz") as data:
				TlLoss_stp = data['TrainLoss_1stStepN']
				TlLoss_1st = data['TrainLoss_1st_single']
			bias = 0

		if os.path.exists(root_dir+"/valLoss_All_mean.npz"):
			with np.load(root_dir+"/valLoss_All_mean.npz") as data:
				valLoss = data['valLoss_All_mean']

		Train_epoch= len(valLoss) # 可能val出的更慢一点
		print(f'Results of {Train_epoch} epochs')

		if os.path.exists(log_path):
			with np.load(log_path) as data:
				train_loss = data['losses_mean']
				train_steps = data['loss_steps']
				#mmval_acc = data['accuracies']/100    # original code we do not use it
				val_steps = data['accuracy_steps']

				train_size = int(len(train_steps) /Train_epoch)
				Train_step = train_steps[1] - train_steps[0]
				Epoch_trainloss = []
				for i in range(Train_epoch):
					fir = train_size * i
					las = train_size*(i+1) - 1
					ave = np.mean(train_loss[fir:las])
					Epoch_trainloss.append(ave)

				Epoch_train_steps = np.arange(Train_step * train_size, np.max(train_steps) + Train_step * train_size, Train_step * train_size)
				Epoch_train_steps = Epoch_train_steps[:len(Epoch_trainloss)]


			if val_steps.shape[0] == 0:
				step_size = round(float(np.max(train_steps)) /valLoss.shape[0])
				val_steps = np.arange(0, np.max(train_steps), step_size)
				val_steps = Epoch_train_steps

		if os.path.exists(root_dir+"/acc1st.npz"):
			with np.load(root_dir+"/acc1st.npz") as bb:
				val_accf = bb['acc_test_folder'] / 100
				val_acc_arif = bb['ARI_test_folder']
				vastep = [0,1,10,100]
		if os.path.exists(acc_path):
			with np.load(acc_path) as bb:
				val_acc = bb['acc_test_folder']/100
				val_acc_ari = bb['ARI_test_folder']

			best_epoch_acc = np.argmax(val_acc)*epoch_validation_gap
			best_epoch_ari = np.argmax(val_acc_ari)*epoch_validation_gap
			print(f"Best accuracy       = {np.max(val_acc)}, @ Epoch{best_epoch_acc}")
			print(f"Best ARI            = {np.max(val_acc_ari)},@ use Epoch {best_epoch_ari} directly start from 0")
			# for i in range (len(val_acc)):
			# 	print(i,val_acc[i])
			# 	if i >30:
			# 		break

		max_steps = max(np.max(val_steps), np.max(train_steps))
		#max_steps = 63000 #np.max(train_steps)


		#plot left side
		with plt.rc_context(
				{'ytick.color': 'tab:blue'}):
			plt.rcParams['font.size'] = 14
			#plot accuracy
			fig, ax1 = plt.subplots()
			fig.set_figheight(6)
			fig.set_figwidth(8)
			colour1 = 'tab:blue'
			ax1.set_xlabel('Steps',fontsize =m_frontsize)
			ax1.set_ylabel('Accuracy or ARI', color=colour1,fontsize =m_frontsize)
			ax1.set_xlim((-max_steps/50, max_steps))
			ax1.set_ylim((0.,1.))
			######log
			#ax1.set_xscale('symlog')  #symlog  logit log
			#ax1.set_xlim((0, max_steps*1.35))

		# ax1.set_xticklabels([ '0', '1', '10', '100', '10000','20000', '30000', '40000', '50000', '60000', '70000', '80000',])

		if os.path.exists(acc_path):
			#ax1.plot(val_steps, val_acc, color=colour1, label='Test Acc', linestyle='-')
			step = val_steps
			if bias:
				gap = step[1] - step[0]
				step =step-gap
			if len(val_steps) > len(val_acc_ari):
				step = step[:len(val_acc_ari)]

			if os.path.exists(root_dir + "/acc1st.npz"):
				step = vastep + step.tolist()
				val_acc = val_accf.tolist() + val_acc.tolist()
				val_acc_ari = val_acc_arif.tolist() + val_acc_ari.tolist()

			if show_val_acc:
				ax1.plot(step, val_acc, color='tab:blue', label='val acc',linestyle='dashed', linewidth= line)
				print(f'length of acc      : {len(step)}, {step[11]- step[10]} steps per epoch')
			# if os.path.exists(root_dir+"/valLoss_1st.npz"):
			# 	step = valLoss_stp.tolist() +step.tolist()
			# 	val_acc_ari = valLoss_1st.tolist() +val_acc_ari.tolist()
			ax1.plot(step, val_acc_ari, color='tab:cyan', label='val ARI', linewidth= line)  #tab:cyan orange  olive   tab:gray colour1

		# plot text
		marksize=90
		#plt.scatter(val_steps[np.argmax(val_acc)], 1.02, markerfacecolor='none')
		if os.path.exists(acc_path):
			x_best = np.argmax(val_acc_ari)
			ax1.text(step[x_best], np.max(val_acc_ari)+0.07, f'{round(np.max(val_acc_ari),2)}', fontsize=m_frontsize)
			#plt.plot(val_steps[x_best], np.max(val_acc_ari), 'o', color='tab:cyan', markersize=10, markerfacecolor='none')
			plt.scatter(step[x_best], np.max(val_acc_ari), color='k',s = marksize)

		with plt.rc_context(
				{'ytick.color': 'tab:orange'}):
			plt.rcParams['font.size'] = 14
			plt.legend(loc="upper right")
			#plot loss
			ax2 = ax1.twinx()
			colour2 = 'tab:red'
			ax2.set_ylabel('Reciprocal Triplet Loss', color='tab:orange',fontsize =m_frontsize)
			ax2.set_ylim((0., y_axi_loss))

			steps = val_steps
			if bias:
				gap = steps[1] - steps[0]
				steps =steps-gap
			if len(val_steps) > len(valLoss):
				steps = steps[:len(valLoss)]
			if os.path.exists(root_dir+"/valLoss_All_mean.npz"):
				if os.path.exists(root_dir + "/valLoss_All_mean.npz"):
					if os.path.exists(root_dir + "/valLoss_1st.npz"):
						steps = valLoss_stp.tolist() + steps.tolist()
						valLoss = valLoss_1st.tolist() + valLoss.tolist()
				ax2.plot(steps, valLoss[:], color=colour2, label='val loss',linewidth= line)
				print(f'length of valloss  : {len(steps)}, {steps[11] - steps[10]} steps per epoch')


			if bias:
				gap = Epoch_train_steps[1] - Epoch_train_steps[0]
				Epoch_train_steps =Epoch_train_steps-gap



			if os.path.exists(root_dir + "/valLoss_1st.npz"):
				Epoch_train_steps = TlLoss_stp.tolist() + Epoch_train_steps.tolist()
				Epoch_trainloss = TlLoss_1st.tolist() + Epoch_trainloss
			else:
				Epoch_train_steps = Epoch_train_steps.tolist()
				Epoch_trainloss = Epoch_trainloss
			ax2.plot(Epoch_train_steps, Epoch_trainloss, color='tab:orange', label='train loss',linewidth= line)
			print(f'length of trainloss: {len(Epoch_train_steps)}, {Epoch_train_steps[11] - Epoch_train_steps[10]} steps per epoch')
			assert Epoch_train_steps[11] - Epoch_train_steps[10] == steps[11] - steps[10]

			if os.path.exists(acc_path):
				if os.path.exists(log_path):
					ax2.text(Epoch_train_steps[x_best], Epoch_trainloss[x_best] + 0.3,	 f'{round(Epoch_trainloss[x_best], 2)}',
							 fontsize=m_frontsize)
					plt.scatter(Epoch_train_steps[x_best], Epoch_trainloss[x_best], color='k',s = marksize)

					ax2.text(steps[x_best], valLoss[x_best]+0.6, f'{round(valLoss[x_best], 2)}', fontsize=m_frontsize)
					plt.scatter(steps[x_best], valLoss[x_best], color='k',s = marksize)

			plt.legend(loc="lower right",borderaxespad=0.3)
		plt.tight_layout()
		#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.1)
		plt.savefig(root_dir + '/train_id.png')
		plt.show()


# Entry method/unit testing method
if __name__ == '__main__':

	root_dir = '/home/io18230/Desktop/' #/cv4animal/2GMM/lr5(didnotuse it)'#   lr5    lr3old
	# valloss: valLoss_All_mean.npz needs to be in the directory.
	train_loss_path = os.path.join(root_dir, "logs.npz") # train loss
	acc_ARI_VAL = os.path.join(root_dir, "acc.npz")
	TrainingVisualiser.visualiseFromNPZFile(train_loss_path, acc_ARI_VAL)