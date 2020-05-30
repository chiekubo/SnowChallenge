import os,sys
import argparse
import cv2
import random
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torchvision

import mymodels
from mypath import Path
from dataloaders import make_data_loader




class Testor(object):
	def __init__(self, args):
		self.args = args
		path = Path()

		# image to patch
		n_patch = int(16*12) #192
		eval_per = int(n_patch/args.batch_size) # 

		# path
		self.event_dir = path.event
		self.model_dir = os.path.join(self.event_dir, 'run', 'checkpoint.pth.tar')

		test_list = args.test_list
		with open(test_list) as f:
			self.lines = f.readlines()

		# range
		self.ids  = [i.split(' ')[0] for i in self.lines]
		self.mins = [float(i.split(' ')[1]) for i in self.lines]
		self.maxs = [float(i.split(' ')[2]) for i in self.lines]

		# Define Dataloader
		kwargs = {'num_workers': args.workers, 'pin_memory': True}
		self.test_loader = make_data_loader(args, **kwargs)

		# Define network
		self.model = mymodels.resnet152()

		# Binary classification
		num_ftrs = self.model.fc.in_features
		self.model.fc = nn.Linear(num_ftrs, 2) # len(class_names) = 2
		del self.model.maxpool

		# Resuming checkpoint
		state_dict = torch.load(self.model_dir)
		print('=>load pre-trained model')
		best_acc = state_dict['best_acc'] # accuracy
		print('=>model accuracy: %0.3f' %  best_acc)
		self.model.load_state_dict(state_dict['state_dict'])

		# Using cuda
		if args.cuda:
			self.model = self.model.cuda()


	def testing(self):
		self.acc_list = []
		patch_pred = []
		self.model.eval()
		tbar = tqdm(self.test_loader, desc='\r')

		for i, sample in enumerate(tbar):
			image = sample['patch']
			# image.size()==torch.Size([32, 3, 128, 128])
			if self.args.cuda:
				image = image.cuda()
			with torch.no_grad():
				output = self.model(image)
				_, pred = torch.max(output, 1)
				pred = list(pred.cpu().numpy())
				patch_pred += pred

				if (i+1)%6==0:
					assert len(patch_pred)==192
					acc = sum(patch_pred)/192.0
					self.acc_list.append(acc)
					patch_pred = []


		assert len(self.acc_list)==len(self.mins)
		assert len(self.acc_list)==len(self.maxs)
		#assert len(self.acc_list)==300*3 # 300 test images for each of no, it, ge

		print('%d images in test set' % len(self.acc_list))

		self.pred_list = []
		for i in range(len(self.acc_list)):
			if not self.acc_list[i]<self.mins[i] and not self.acc_list[i]>self.maxs[i]:
				self.pred_list.append(1)
			else:
				self.pred_list.append(0)

		print('no snow image dectect accuracy: {}'.format(sum(self.pred_list[:300])/300.0))
		print('it snow image dectect accuracy: {}'.format(sum(self.pred_list[300:600])/300.0))
		print('ge snow image dectect accuracy: {}'.format(sum(self.pred_list[600:900])/300.0))
		print('general dectect accuracy: {}'.format(1.0*(sum(self.pred_list))/len(self.pred_list)))


	def saving(self):
		output_fn = 'test_result.csv'
		output_dir = os.path.join(self.event_dir, output_fn)
		assert len(self.ids)==len(self.acc_list)

		path_list = self.ids
		acc_list = self.acc_list
		range_list = ['('+str(i.split(' ')[1])+','+str(i.split(' ')[2])+')'  for i in self.lines]
		decision_list = self.pred_list

		df = pd.DataFrame({'image_path': path_list, 'coverage': acc_list, 'range':range_list, 'decision':decision_list})
		df.to_csv('test_result.csv', index=False)

		if os.path.isfile(output_dir):
			print('output has been saved in format csv')





def main():
	parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")

	# dataset option
	parser.add_argument('--dataset', type=str, default='harmo_test',
						help='path to test image folder')
	parser.add_argument('--workers', type=int, default=1, metavar='N', 
						help='dataloader threads')

    # test option
	parser.add_argument('--batch-size', type=int, default=32, metavar='N', 
						help='input batch size for testing (default: 64)')

	# cuda, seed and logging
	parser.add_argument('--no-cuda', action='store_true', default=False, 
						help='disables CUDA training')

	# checking point
	parser.add_argument('--resume', type=str, default=None,
						help='put the path to resuming file if needed')

	args = parser.parse_args()

	# cuda option
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	args.test_list = 'test_list.txt'

	# batch size option
	if args.batch_size is None:
		args.batch_size = 32

	testor = Testor(args)
	testor.testing()
	testor.saving()


if __name__ == "__main__":
	main()