import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from torch.autograd import Variable
import sys
import json
from models import cnns,train,cascade
from torchvision import transforms
from sklearn.model_selection import RepeatedStratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='cars')
parser.add_argument('--saving_folder', type=str, default='')
parser.add_argument('--batch_size', type=int,default=64)
parser.add_argument('--lr',type=float,default=0.01)
parser.add_argument('--momentum',type=float,default=0.9)
parser.add_argument('--nesterov', action='store_false',default=True)
parser.add_argument('--weight_decay', type=float,default=10.e-3)
parser.add_argument('--lr_schedule', type=str, default='10,70')
parser.add_argument('--nb_epochs' , type=int,default=10)
parser.add_argument('--verbose', action='store_true',default=False)
parser.add_argument('--save_best',action='store_true',default=False)
parser.add_argument('--tuning_epochs',type=int,default=40)
parser.add_argument('--fine_tune',action='store_true',default=False)
parser.add_argument('--depth',type=int,default=-1)
parser.add_argument('--cascade',action='store_true',default=False)
parser.add_argument('--starting_nb_epochs',type=int,default=5)
parser.add_argument('--epochs_step',type=int,default=0)
parser.add_argument('--starting_stage',type=int,default=0)
parser.add_argument('--data_augmentation',action='store_true',default=False)
parser.add_argument('--model',type=str,default='resnet18')
parser.add_argument('--fraction_test_set', type=float,default=1.)
parser.add_argument('--back_stages', type=int, default=2)
parser.add_argument('--cross_validation_split',type=int,default=0)
parser.add_argument('--seed',type=int,default=7)
parser.add_argument('--devices',type=int,default=1)
args = parser.parse_args()


if args.data_augmentation:
    training_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
else:
    training_transform, test_transform = 2*[transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])]

if args.dataset == 'cars':
    from data import cars
    stratified_crossvalidation = RepeatedStratifiedKFold(n_splits=3,n_repeats=5,random_state=args.seed)
    data = cars.Calltech101(images_folder='images',input_transform=training_transform)
    data.shuffle(seed=args.seed)
    data_size = len(data)
    indexes = np.arange(0,data_size,1,dtype=int)
    targets = np.asarray(data.targets)
    kfolds = list(stratified_crossvalidation.split(indexes,targets))
    train_dataset = cars.Calltech101(images_folder='images',input_transform=training_transform)
    train_dataset.shuffle(seed=args.seed)
    train_dataset.prune_dataset(indexes=kfolds[args.cross_validation_split][1])
    train_dataset.shuffle(seed=args.seed)

    training_loader = DataLoader(dataset=train_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=2,
                                 drop_last=True,
                                 shuffle=True,
                                 pin_memory=True)

    random_training_sample = next(iter(training_loader))[0]