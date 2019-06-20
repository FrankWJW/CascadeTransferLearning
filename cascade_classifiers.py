import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from torch.autograd import Variable
import sys
import json
from models import cnns,test,cascade
from torchvision import transforms
from sklearn.model_selection import RepeatedStratifiedKFold
from collections import namedtuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--saving_folder', type=str, default='E:/git/CascadeTransferLearning/test/car')
parser.add_argument('--seed',type=int,default=7)
parser.add_argument('--devices',type=int,default=1)
args = parser.parse_args()

run_info = json.load(open(os.path.join(args.saving_folder,'run_info.json'),'r'))
args = namedtuple('Struct', run_info.keys())(*run_info.values())

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
    test_dataset = cars.Calltech101(images_folder='images',input_transform=test_transform)
    train_dataset.shuffle(seed=args.seed)
    test_dataset.prune_dataset(indexes=kfolds[args.cross_validation_split][0])
    validation_dataset = None
    metric = 'mean_class_acc'
    del data
    # validation_dataset = cars.Calltech101(images_folder='validation',input_transform=input_transform)
elif args.dataset == 'flowers':
    from data import flowers
    train_dataset = flowers.Flowers102(input_transform=training_transform)
    test_dataset = flowers.Flowers102(images_folder='test',input_transform=test_transform)
    validation_dataset = None
    metric = 'mean_class_acc'
    # validation_dataset = flowers.Flowers102(images_folder='validation',input_transform=test_transform)
elif args.dataset == 'texture':
    from data import texture
    # nb_files = 1-10
    train_dataset = texture.DTD(images_folder=['train'+str(args.cross_validation_split)+'.txt','val'+str(args.cross_validation_split)+'.txt'],input_transform=training_transform)
    test_dataset = texture.DTD(images_folder='test'+str(args.cross_validation_split)+'.txt',input_transform=test_transform)
    metric = 'acc'
    # validation_dataset = texture.DTD(images_folder='val'+str(args.cross_validation_split)+'.txt',input_transform=test_transform)
    validation_dataset = None
# test_dataset.chop_dataset(fraction=0.05)
testing_loader = DataLoader(dataset=test_dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=0,
                            drop_last=True,
                            pin_memory=True)
 
if args.model[0] == 'r':
  encoder = cnns.ResNetEncoder(resnet_version=args.model)
  model =  cnns.TransferLearning(encoder=encoder,nb_classes=train_dataset.nb_classes())
else:
  encoder = cnns.VGGEncoder(vgg_version=args.model)
  model =  cnns.TransferLearning(encoder=encoder,nb_classes=train_dataset.nb_classes(),max_pool=True,convs_compression=True)

test_sample = next(iter(testing_loader))[0].cuda()
model = model.cuda()
criterion = nn.CrossEntropyLoss()
cascade_trainer = cascade.CascadeLearning(None,None,starting_stage=args.starting_stage)
images_to_plot = {}
tester = test.Tester(testing_loader,criterion,verbose=args.verbose,mean_per_class_metric= metric == 'mean_class_acc')
for sub_model,stage in cascade_trainer.yield_models(model,back_stages=args.back_stages,x_sample=test_sample):
  # sub_model.load_state_dict(torch.load(os.path.join(args.saving_folder,str(stage),'modelbest_.pth.tar')))
  sub_model.load_state_dict(torch.load(os.path.join(args.saving_folder, str(stage), 'model.pth.tar')))
  sub_model.train(False)
  correct_idxs = tester.get_high_confidence_images(sub_model)
  # sorted_idxs = np.argsort(np.abs(correct_idxs[:,1]-0.9))
  # correct_idxs = correct_idxs[sorted_idxs]
  # correct_idxs = correct_idxs[-10::]
  images_to_plot[stage] = correct_idxs

nb_stages = len(images_to_plot.keys())
fig, axeslist = plt.subplots(ncols=nb_stages, nrows=10)
counter = 0
for stage,idxs_to_plot in images_to_plot.items():
  for idx,_ in idxs_to_plot:
    axeslist.ravel()[counter].imshow(test_dataset.get_image(int(idx)))
    axeslist.ravel()[counter].grid(False)
    axeslist.ravel()[counter].axis('off')
    counter += 1
# plt.tight_layout()
# plt.axis('off')
plt.savefig(os.path.join(args.saving_folder,'images.jpg'))


