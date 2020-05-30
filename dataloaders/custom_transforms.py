import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from torchvision import transforms

class Normalize(object):
	"""Normalize a tensor image with mean and standard deviation.
	Args:
		mean (tuple): means for each channel.
		std (tuple): standard deviations for each channel.
	"""
	def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
		self.mean = mean
		self.std = std

	def __call__(self, img):
		img = np.array(img).astype(np.float32)
		img /= 255.0
		img -= self.mean
		img /= self.std

		return img


class PatchToTensor(object):
	"""Convert ndarrays in sample to Tensors."""
	def __call__(self, img):
		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		img = np.array(img).astype(np.float32).transpose((2, 0, 1)) # (3,128,128)
		img = torch.from_numpy(img).float()
		return img


class RandomHorizontalFlip(object):
	def __call__(self, img):
		if random.random() < 0.5:
			img = img.transpose(Image.FLIP_LEFT_RIGHT)
		return img


class RandomGaussianBlur(object):
	def __call__(self, img):
		if random.random() < 0.5:
			img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
		return img


class GaussianNoise(object):
	def __call__(self, img, mean=0, std=5):
		if random.random() < 0.3:
			img_np = np.asarray(img)
			#print(img_np.shape)
			noise = np.random.normal(loc=mean, scale=std, size=np.shape(img_np))
			img = Image.fromarray(np.uint8(np.add(img_np, noise)))
		return img

