import numpy as np
import os
from .dataset import ImageDataset
import shutil
import torch
from scipy import io

# default_data_path = '/ssd/esm1g14/CascadeTransferLearning/102flowers/'
default_data_path = 'E:/git/CascadeTransferLearning/data/flowers'
class Flowers102(ImageDataset):
	def __init__(self,datapath=default_data_path,images_folder='train-val',input_transform=None):
		super().__init__(datapath=datapath,input_transform=input_transform)
		self.path = os.path.join(self.datapath,'images')
		self.images_file = images_folder
		self.prepare_data()
		self.shuffle()
	def prepare_data(self):
		labels = io.loadmat(os.path.join(self.datapath,'imagelabels.mat'))['labels'][0]
		images_idxs = io.loadmat(os.path.join(self.datapath,'setid.mat'))
		if self.images_file == 'train-val':
			images_idxs = np.concatenate((images_idxs['trnid'][0].flatten(),images_idxs['valid'][0].flatten()))
		elif self.images_file == 'train':
			images_idxs = images_idxs['trnid'][0].flatten()
		elif self.images_file == 'test':
			images_idxs = images_idxs['tstid'][0].flatten()
		elif self.images_file == 'validation':
			images_idxs = images_idxs['valid'][0].flatten()

		self.targets = torch.from_numpy(labels[images_idxs-1]-1)
		self.images_paths = np.asarray(['image_'+str(i).zfill(5)+'.jpg' for i in images_idxs])