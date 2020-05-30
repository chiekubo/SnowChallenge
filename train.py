import os,sys
import argparse
import cv2
import random
import numpy as np
from tqdm import tqdm
import json

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

from mypath import Path
from dataloaders import make_data_loader

import mymodels



class Trainer(object):
	def __init__(self, args):
		self.args = args
		self.path = Path()

		# Saver
		self.saver = Saver(args, self.path.event) # Define Saver

		# Define Dataloader
		kwargs = {'num_workers': args.workers, 'pin_memory': True}
		self.train_loader, self.val_loader = make_data_loader(args, **kwargs)

		# Define network
		self.model = mymodels.resnet152(pretrained=True)
		# self.model = mymodels.resnet50(pretrained=True)
		# if args.pretrained:
		# 	self.model = mymodels.resnet101(pretrained=True)
		# else:
		# 	self.model = mymodels.resnet101()

		# Binary classification
		num_ftrs = self.model.fc.in_features
		self.model.fc = nn.Linear(num_ftrs, 2) # len(class_names) = 2
		del self.model.maxpool

		# Resuming checkpoint
		if args.resume is not None: # path to resuming file
			if not os.path.isfile(args.resume):
				raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
			state_dict = torch.load(args.resume)
			print('=>load pre-trained model from')
			print(args.resume)
			best_acc = state_dict['best_acc'] # accuracy
			print('=>model top 1 accuracy: %0.3f' %  best_acc)
			self.model.load_state_dict(state_dict['state_dict'])

		# Using cuda
		if args.cuda:
			self.model = self.model.cuda()

		# define loss function (criterion) and optimizer
		self.criterion = nn.CrossEntropyLoss().cuda()#交叉熵损失函数
		self.optimizer = torch.optim.SGD(self.model.parameters(),
										args.lr,
										momentum=args.momentum,
										weight_decay=args.weight_decay)

		# Define lr scheduler
		self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

		# Record
		self.best_acc = 0.0


	# training
	def training(self, epoch):
		train_itr_loss = 0.0
		self.model.train() # train model
		tbar = tqdm(self.train_loader)
		n_img_tr = len(self.train_loader.dataset) # number of traindata 19688

		for i, sample in enumerate(tbar):
			image, target = sample['image'], sample['label'] 
			# image.size()==torch.Size([32, 3, 128, 128])
			# target==tensor([0, 1,...,0]), lenth==38

			if self.args.cuda:
				image, target = image.cuda(), target.cuda()
			self.optimizer.zero_grad()
			output = self.model(image)
			loss = self.criterion(output, target) # loss
			loss.backward()
			self.optimizer.step()
			train_itr_loss += loss.item() # single_loss*batch_size
			tbar.set_description('Train loss: %.3f' % (train_itr_loss / (i + 1)))
		self.scheduler.step()
		train_loss = train_itr_loss/n_img_tr
		print('Train:')
		print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
		print('Loss: %.3f' % train_loss)


	def validation(self, epoch):
		val_itr_loss = 0.0
		itr_acc = 0.0
		self.model.eval()
		tbar = tqdm(self.val_loader, desc='\r')
		n_img_va = len(self.val_loader.dataset) # number of val data
		for i, sample in enumerate(tbar):
			image, target = sample['image'], sample['label']
			if self.args.cuda:
				image, target = image.cuda(), target.cuda()
			with torch.no_grad():
				output = self.model(image)
				_, pred = torch.max(output, 1)
			loss = self.criterion(output, target)
			val_itr_loss += loss.item()
			tbar.set_description('Val loss: %.3f' % (val_itr_loss / (i + 1)))
			itr_acc += torch.sum(pred==target.data)
		val_loss = val_itr_loss
		acc = float(itr_acc)/n_img_va
		print('Validation:')
		print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
		print('Current accuracy: %.3f' % acc)
		print('Best accuracy: %.3f' % self.best_acc)
		print('Loss: %.3f' % val_loss)

		# best model
		current_acc = acc # accuracy
		if current_acc > self.best_acc:
			self.best_acc = current_acc
			self.saver.save_args()
			self.saver.save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': self.model.state_dict(),
				'optimizer': self.optimizer.state_dict(),
				'best_acc': self.best_acc})


# saver
class Saver(object):
	def __init__(self, args, event_dir):
		self.args = args
		self.experiment_dir = os.path.join(event_dir, 'run') # folder path
		if not os.path.exists(self.experiment_dir):
			os.makedirs(self.experiment_dir)

	# save model
	def save_checkpoint(self, state, name='checkpoint.pth.tar'):
		filename = os.path.join(self.experiment_dir, name)
		torch.save(state, filename)

	# save all args configuration
	def save_args(self, filename='args.txt'):
		cfgfile = os.path.join(self.experiment_dir, filename)
		with open(cfgfile, 'w') as f:
			json.dump(self.args.__dict__, f, indent=2)




def main():
	parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")

	# model option
	parser.add_argument('--pretrained', dest='pretrained', action='store_true',
						help='use pre-trained model')

	# dataset option
	parser.add_argument('--dataset', type=str, default='harmo_patch',
						choices=['harmo_patch', 'harmo_image'],
						help='dataset name (default: harmo_patch)')
	parser.add_argument('--workers', type=int, default=1, metavar='N', 
						help='dataloader threads')

    # training option
	parser.add_argument('--epochs', type=int, default=60, metavar='N',
						help='number of epochs to train (default: 90)')
	parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
						help='start epochs (default:0)')
	parser.add_argument('--batch-size', type=int, default=32, metavar='N', 
						help='input batch size for training (default: auto)')
	parser.add_argument('--test-batch-size', type=int, default=None, metavar='N', 
						help='input batch size for testing (default: 64)')
	parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
						metavar='LR', help='initial learning rate', dest='lr')
	parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
						help='momentum (default: 0.9)')
	parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
						metavar='W', help='weight decay (default: 1e-4)',
						dest='weight_decay')

	# cuda, seed and logging
	parser.add_argument('--no-cuda', action='store_true', default=False, 
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')

	# checking point
	parser.add_argument('--resume', type=str, default=None,
						help='put the path to resuming file if needed')

	# evaluation option
	parser.add_argument('--eval-interval', type=int, default=1, # validation frequency
						help='evaluuation interval (default: 1)')

	# path
	parser.add_argument('--save-path', type=str, default=None,
						help='path to saver')

	args = parser.parse_args()

	# cuda option
	args.cuda = not args.no_cuda and torch.cuda.is_available()

	# batch size option
	if args.batch_size is None:
		args.batch_size = 32
	if args.test_batch_size is None:
		args.test_batch_size = args.batch_size

	print(args)

	random.seed(args.seed)
	torch.manual_seed(args.seed)
	trainer = Trainer(args)
	print('Starting Epoch:', trainer.args.start_epoch)
	print('Total Epoches:', trainer.args.epochs)
	for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
		trainer.training(epoch)
		if epoch % args.eval_interval==0:
			trainer.validation(epoch)

if __name__ == "__main__":
	main()