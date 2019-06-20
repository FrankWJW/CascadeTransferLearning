import numpy as np
import os
from .dataset import ImageDataset
import shutil
import torch

# default_data_path = '/ssd/esm1g14/CascadeTransferLearning/Calltech-101/'
default_data_path = 'E:/git/CascadeTransferLearning/data/'
# default_data_path = 'CascadeTransferLearning/data'
default_test_images = [20,30]
class Calltech101(ImageDataset):
	def __init__(self,datapath=default_data_path,images_folder='images',input_transform=None,cross_validation_index=None):
		super().__init__(datapath=datapath,input_transform=input_transform)
		self.path = os.path.join(self.datapath,images_folder)
		self.prepare_data()
		# self.shuffle(seed=7)
		# self.cross_validation_index = cross_validation_index
	def prepare_data(self):
		label_names = os.listdir(self.path)
		label_names.sort()
		self.label2idx = {label : i for i,label in enumerate(label_names)}
		images_paths = []
		targets = []
		for folder_label in label_names:
			current_images = os.listdir(os.path.join(self.path,folder_label))
			images_paths += [os.path.join(folder_label,current_image) for current_image in current_images]
			targets += [len(current_images)*[self.label2idx[folder_label]]]
		self.images_paths = np.asarray(images_paths)
		self.targets = torch.from_numpy(np.concatenate(targets,axis=0))
	def split_train_test(self,test_indexes=default_test_images,train_indexes=None):
		raw_images_path = os.path.join(self.datapath,'images')
		label_names = os.listdir(raw_images_path)
		train_path = os.path.join(self.datapath,'train')
		test_path = os.path.join(self.datapath,'test')
		if os.path.isdir(train_path):
			raise ValueError('DATASETS ALREADY SPLITTED')
		os.mkdir(train_path)
		os.mkdir(test_path)
		
		for folder_label in label_names:
			raw_data_path = os.path.join(raw_images_path,folder_label)
			label_train_path = os.path.join(train_path,folder_label)
			label_test_path = os.path.join(test_path,folder_label)
			os.mkdir(label_train_path)
			os.mkdir(label_test_path)
			images = os.listdir(raw_data_path)
			test_images = []
			train_images = []
			for current_image in images:
				image_index = int(current_image.split('_')[1].split('.')[0])
				if image_index in test_indexes:
					test_images += [current_image]
				elif train_indexes == None:
					train_images += [current_image]
				elif image_index in train_indexes:
					train_indexes += [current_image]
			for test_image in test_images:
				shutil.copy(os.path.join(raw_data_path,test_image),os.path.join(label_test_path,test_image))
			for train_image in train_images:
				shutil.copy(os.path.join(raw_data_path,train_image),os.path.join(label_train_path,train_image))
	def random_split_train_test(self,training_samples_per_class=31,test_samples_per_class=None):
		raw_images_path = os.path.join(self.datapath,'images')
		label_names = os.listdir(raw_images_path)
		train_path = os.path.join(self.datapath,'train')
		test_path = os.path.join(self.datapath,'test')
		if os.path.isdir(train_path):
			raise ValueError('DATASETS ALREADY SPLITTED')
		os.mkdir(train_path)
		os.mkdir(test_path)
		
		for folder_label in label_names:
			raw_data_path = os.path.join(raw_images_path,folder_label)
			label_train_path = os.path.join(train_path,folder_label)
			label_test_path = os.path.join(test_path,folder_label)
			os.mkdir(label_train_path)
			os.mkdir(label_test_path)
			images = os.listdir(raw_data_path)
			np.random.shuffle(images)
			train_images = images[0:training_samples_per_class]
			if test_samples_per_class == None:
				test_images = images[training_samples_per_class::]
			for test_image in test_images:
				shutil.copy(os.path.join(raw_data_path,test_image),os.path.join(label_test_path,test_image))
			for train_image in train_images:
				shutil.copy(os.path.join(raw_data_path,train_image),os.path.join(label_train_path,train_image))
	def prune_dataset(self,indexes):
		self.images_paths = self.images_paths[indexes]
		self.targets = self.targets[indexes]

	

