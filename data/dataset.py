import numpy as np
import os
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

class ImageDataset():
	def __init__(self,datapath,input_transform=None):
		self.datapath = datapath
		self.input_transform = input_transform
	def shuffle(self,seed=None):
		if seed:
			permutation = np.random.RandomState(seed=seed).permutation(len(self.images_paths))
		else:
			permutation = np.random.permutation(len(self.images_paths))
		permutation = np.random.permutation(len(self.images_paths))
		self.images_paths = self.images_paths[permutation]
		self.targets = self.targets[permutation]
	def __getitem__(self,index):
		x = Image.open(os.path.join(self.path,self.images_paths[index])).convert('RGB')
		# print(x.size)
		y = self.targets[index].type(torch.LongTensor)
		if self.input_transform:
			x = self.input_transform(x)
		# print(x.size())
		return x,y
	def chop_dataset(self,fraction=0.5):
		self.shuffle()
		self.images_paths = self.images_paths[0:int(len(self)*fraction)]
		self.targets = self.targets[0:len(self)]
	def __len__(self):
		return len(self.images_paths)
	def nb_classes(self):
		return int(max(self.targets)+1)
	def plot_random_images(self,plot_size=(3,3)):
		figure = plt.figure()
		nb_images = np.prod(plot_size)
		self.shuffle()
		images_to_plot = [self[i] for i in range(nb_images)]
		counter = 1
		for i in range(plot_size[0]):
			for j in range(plot_size[1]):
				subplot = figure.add_subplot(plot_size[0],plot_size[1],counter)
				subplot.imshow(images_to_plot[counter-1][0])
				plt.title(list(self.label2idx.keys())[list(self.label2idx.values()).index(images_to_plot[counter-1][1])])
				counter += 1
		plt.show()
	def get_image(self,index):
		return Image.open(os.path.join(self.path,self.images_paths[index])).convert('RGB')

		



