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
parser.add_argument('--saving_folder', type=str, default='E:/git/CascadeTransferLearning/')
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

if not os.path.isdir(args.saving_folder) and len(str(args.saving_folder)) != 0:
  os.mkdir(args.saving_folder)
  with open(os.path.join(args.saving_folder,'run_info.json'), 'w') as fp:
      json.dump(vars(args), fp)

elif len(str(args.saving_folder)) == 0:
    args.saving_folder = None
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
# test_dataset.chop_dataset(fraction=args.fraction_test_set)
training_loader = DataLoader(dataset=train_dataset,
                             batch_size=args.batch_size,
                             num_workers=0,
                             drop_last=True,
                             shuffle=True,
                             pin_memory=True)
testing_loader = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=0,
                            drop_last=True,
                            pin_memory=True)
if validation_dataset != None:
    validation_loader = DataLoader(dataset=validation_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=0,
                                drop_last=True,
                                pin_memory=True)
else:
    validation_loader=None

if args.model[0] == 'r':
  encoder = cnns.ResNetEncoder(resnet_version=args.model)
  model =  cnns.TransferLearning(encoder=encoder,nb_classes=train_dataset.nb_classes())
else:
  encoder = cnns.VGGEncoder(vgg_version=args.model)
  model =  cnns.TransferLearning(encoder=encoder,nb_classes=train_dataset.nb_classes(),max_pool=True,convs_compression=True)

random_training_sample = next(iter(training_loader))[0]
print('BATCH SIZE')
print(random_training_sample.shape)
print('LEN TRAINING')
print(len(train_dataset))
print('LEN TEST')
print(len(test_dataset))
device = torch.device('cuda:0')
model.set_encoder_stage(args.depth)
model.build_classifier(random_training_sample)
model = model.cuda()
if args.devices > 1:
  model = nn.DataParallel(model,device_ids=[x for x in range(args.devices)])
  # model.to(device)
# if not args.cascade:
#     model.freeze_encoder()
optimizer = cascade.CascadeOptimizer(torch.optim.SGD,{'lr' : args.lr,'momentum' : args.momentum, 'nesterov' : bool(args.nesterov), 'weight_decay' : args.weight_decay})
# optimizer = torch.optim.SGD(filter(lambdna p: p.requires_grad, model.parameters()),lr=args.lr,momentum=args.momentum,nesterov=bool(args.nesterov),weight_decay=args.weight_decay)
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
criterion = nn.CrossEntropyLoss()
trainer = train.Trainer(model,training_loader,optimizer.plain_optimizer(model,learning_rate_reduction=1),criterion,validation_loader=validation_loader,test_loader=testing_loader,verbose=args.verbose,
                 saving_folder=args.saving_folder,save_best=bool(args.save_best),mean_per_class_metric= metric == 'mean_class_acc',devices=[x for x in range(args.devices)])
# trainer = train.Trainer(model,training_loader,optimizer.plain_optimizer(model,learning_rate_reduction=1),criterion,validation_loader=validation_loader,test_loader=testing_loader,verbose=args.verbose,
#                  saving_folder=args.saving_folder,save_best=bool(args.save_best),mean_per_class_metric= metric == 'mean_class_acc',devices='cpu')

if bool(args.cascade):
    cascade_trainer = cascade.CascadeLearning(trainer,optimizer,starting_nb_epochs=args.starting_nb_epochs,epochs_step=args.epochs_step,starting_stage=args.starting_stage)
    cascade_trainer(back_stages=args.back_stages)
else:
    trainer(args.nb_epochs,drop_learning_rate=[int(item) for item in args.lr_schedule.split(',')])
if args.fine_tune:
    print('TUNING')
    trainer.saving_folder = args.saving_folder
    model.freeze_encoder(defreeze=True)
    trainer.optimizer = optimizer.plain_optimizer(model,learning_rate_reduction=10)
    trainer.reset_history()
    trainer(args.tuning_epochs,drop_learning_rate=[int(item) for item in args.lr_schedule.split(',')][1::],name='_tuning')
if args.saving_folder != None:
  with open(os.path.join(args.saving_folder,'run_info.json'), 'w') as fp:
      json.dump(vars(args), fp)