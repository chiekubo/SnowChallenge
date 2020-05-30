import os, sys
from PIL import Image
import numpy as np
import random
import dataloaders.custom_transforms as tr
import cv2

import torch
import os, sys
from torchvision import transforms
from torch.utils.data import Dataset

sys.path.append('../')
from mypath import Path


class harmo_patch(Dataset):
	def __init__(self, split='train'):
		super().__init__()

		assert split in ['train', 'val']
		self.split = split

		path = Path()
		self._base_dir = path.patch

		self.split_fn = os.path.join(self._base_dir, self.split+'.txt')

		self.im_ids = []
		self.im_lbs = [] # label:0/1
		with open(self.split_fn, 'r') as f:
			lines = f.read().splitlines()

		for line in lines:
			im_id, im_lb = line.split(' ')
			self.im_ids.append(im_id)
			self.im_lbs.append(im_lb)
		assert len(self.im_ids)==len(self.im_lbs)


	def __getitem__(self, index):
		_image_dir = os.path.join(self._base_dir, self.im_ids[index])
		_image = Image.open(_image_dir)
		if self.split == "train":
			_image = self.transform_tr(_image) # transformed dataset
		elif self.split == 'val':
			_image = self.transform_val(_image)

		_label = int(self.im_lbs[index])
		return {'image':_image, 'label':_label}


	def __len__(self):
		return len(self.im_ids)


	# data transformation
	def transform_tr(self, sample):
		composed_transforms = transforms.Compose([
			# transforms.ColorJitter(brightness=(-1,1),contrast=(-1, 1),saturation=(-0.3, 0.3), hue=(-0.3, 0.3)),
			# transforms.ColorJitter(brightness=0.4, contrast=0.4,saturation=0.4),
			tr.RandomHorizontalFlip(),
			tr.GaussianNoise(),
			tr.RandomGaussianBlur(),
			tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			tr.PatchToTensor()])


		return composed_transforms(sample)

	def transform_val(self, sample):
		composed_transforms = transforms.Compose([
			tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			tr.PatchToTensor()])

		return composed_transforms(sample)


	def __str__(self):
		return 'harmo_patch(split=' + str(self.split) + ')'


class harmo_test(Dataset):
	def __init__(self):
		super().__init__()
		with open('test_list.txt') as f:
			test_list = f.readlines()

		self.image_ids = [i.split(' ')[0] for i in test_list]

		path = Path()
		self._base_dir = path.harmo
		self.all_patch_list = []
		for image_id in self.image_ids:
			image = cv2.imread(os.path.join(self._base_dir, image_id))
			#print(os.path.join(self._base_dir, image_id))
			#assert os.path.exists(os.path.join(self._base_dir, image_id))
			image_pad = self.pad_image(image)
			patch_list = self.image_to_patch(image_pad)
			self.all_patch_list += patch_list
		assert len(self.all_patch_list)==len(self.image_ids)*16*12

	def __getitem__(self, index):
		_patch = self.all_patch_list[index]
		_patch = self.transform_val(_patch)
		return {'patch':_patch}


	def __len__(self):
		return len(self.all_patch_list)


	def pad_image(self, image_array): #(768*1024*3)->(832*1088*3)
		image_pad = np.zeros((832, 1088, 3), np.uint8)
		image_pad[32:800, 32:1056, :] = image_array
		return image_pad # paded image


	def image_to_patch(self, image_pad):
		patch_list = []
		w, h = 16, 12
		for i in range(w):
			for j in range(h):
				patch = np.zeros((128, 128, 3), np.uint8)
				patch = image_pad[j*64:(j+2)*64, i*64:(i+2)*64, :]
				# patch = image_pad[32+h*64-32:32+(h+1)*64+32, 32+w*64-32:32+(w+1)*64+32, :]
				patch_list.append(patch)
		assert len(patch_list)==w*h
		return patch_list


	def transform_val(self, sample):
		composed_transforms = transforms.Compose([
			tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			tr.PatchToTensor()])

		return composed_transforms(sample)


	def __str__(self):
		return 'test_image'


class harmo_image(Dataset):
	def __init__(self, args):
		super().__init__()
		path = Path()
		self._base_dir = path.detect
		self.all_patch_list = []
		self.image_ids = args.image_ids
		for image_id in self.image_ids:
			image = cv2.imread(os.path.join(self._base_dir, image_id))
			image_pad = self.pad_image(image)
			patch_list = self.image_to_patch(image_pad)
			self.all_patch_list += patch_list
		assert len(self.all_patch_list)==len(self.image_ids)*16*12

	def __getitem__(self, index):
		_patch = self.all_patch_list[index]
		_patch = self.transform_val(_patch)
		return {'patch':_patch}


	def __len__(self):
		return len(self.all_patch_list)


	def pad_image(self, image_array): #(768*1024*3)->(832*1088*3)
		image_pad = np.zeros((832, 1088, 3), np.uint8)
		image_pad[32:800, 32:1056, :] = image_array
		return image_pad # paded image


	def image_to_patch(self, image_pad):
		patch_list = []
		w, h = 16, 12
		for i in range(w):
			for j in range(h):
				patch = np.zeros((128, 128, 3), np.uint8)
				patch = image_pad[j*64:(j+2)*64, i*64:(i+2)*64, :]
				# patch = image_pad[32+h*64-32:32+(h+1)*64+32, 32+w*64-32:32+(w+1)*64+32, :]
				patch_list.append(patch)
		assert len(patch_list)==w*h
		return patch_list


	def transform_val(self, sample):
		composed_transforms = transforms.Compose([
			tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			tr.PatchToTensor()])

		return composed_transforms(sample)


	def __str__(self):
		return 'test_image'




