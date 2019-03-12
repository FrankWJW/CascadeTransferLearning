import numpy as np
import os
from .dataset import ImageDataset
import shutil
import torch

default_data_path = '/ssd/esm1g14/CascadeTransferLearning/fgvc-aircraft-2013b/data/'
class FGVC(ImageDataset):
	def __init__(self,datapath=default_data_path,images_folder='images_variant_trainval.txt',input_transform=None):
		super().__init__(datapath=datapath,input_transform=input_transform)
		self.path = os.path.join(self.datapath,'images')
		self.images_file = images_folder
		self.prepare_data()
		self.shuffle()
	def prepare_data(self):
		labels = open(os.path.join(self.datapath,'variants.txt'),'r').readlines()
		labels = [label.strip() for label in labels]
		self.label2idx = {label : i for i,label in enumerate(labels)}
		file_data = open(os.path.join(self.datapath, self.images_file),'r').readlines()
		self.images_paths = np.asarray([x.split(' ')[0]+'.jpg' for x in file_data])
		self.targets = torch.from_numpy(np.asarray([self.label2idx[x.split(' ')[1].strip()] if len(x.split(' ')) == 2 else self.label2idx[' '.join(x.split(' ')[1::]).strip()] for x in file_data]))