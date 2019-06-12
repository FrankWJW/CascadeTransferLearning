import numpy as np
import os
from .dataset import ImageDataset
import shutil
import torch

#default_data_path = '/ssd/esm1g14/CascadeTransferLearning/dtd/images'
default_data_path = 'D:/git/CascadeTransferLearning/data'
class DTD(ImageDataset):
	'Describable textures dataset'
	def __init__(self,datapath=default_data_path,images_folder=['train1.txt','val1.txt'],input_transform=None):
		super().__init__(datapath=datapath,input_transform=input_transform)
		self.path = self.datapath
		if isinstance(images_folder,list):
			self.data_file = [os.path.join(datapath,'../labels',image_folder) for image_folder in images_folder]
		else:
			self.data_file = [os.path.join(datapath,'../labels',images_folder)]
		self.prepare_data()
		# self.shuffle()
	def prepare_data(self):
		label_names = os.listdir(self.path)
		label_names.sort()
		self.label2idx = {label : i for i,label in enumerate(label_names)}
		self.images_paths = [np.loadtxt(data_file,dtype=str) for data_file in self.data_file]
		self.images_paths = np.concatenate(self.images_paths)
		targets = [self.label2idx[image.split('/')[0]] for image in self.images_paths]
		self.targets = torch.Tensor(targets)