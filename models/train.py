import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable
from tqdm import tqdm
import _pickle as pickle
from sklearn.metrics import f1_score
import numpy as np
import os
import json


class Trainer():
    def __init__(self,model,training_loader,optimizer,criterion,validation_loader=None,test_loader=None,verbose=False,
                 saving_folder=None,save_best=False,mean_per_class_metric=True,memory=True,devices=[0]):
        self.model = model
        self.verbose = verbose
        self.training_loader = training_loader
        self.test_loader = test_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = {'test_acc'  : [],
                        'train_acc'  : [],
                        'test_loss' : [],
                        'train_loss' : [],
                        'val_loss' : [],
                        'val_acc' : []}
        self.saving_folder = saving_folder
        self.save_best = save_best
        self.mean_per_class_metric = mean_per_class_metric
        self.nb_classes = self.training_loader.dataset.nb_classes()
        self.devices = devices
        # self.devices = 'cpu'
        self.model = model.cuda()
        if len(self.devices) > 1:
            self.model = torch.nn.parallel.replicate(self.model, self.devices)
    def __call__(self,nb_epochs,drop_learning_rate=[],name=''):
        print(('TRAINING MODEL WITH EPOCHS %d')%(nb_epochs))
        best_acc = 0.
        starting_epoch = len(self.history['test_loss'])
        for epoch in range(starting_epoch,nb_epochs):
            if epoch in drop_learning_rate:
                for g in self.optimizer.param_groups:
                    g['lr'] = g['lr']*0.1
            print(('EPOCH : %d')%(epoch))
            train_loss,train_acc = self.train_epoch()
            if self.test_loader:
                test_loss,test_acc = self.test_epoch(val=False)
            if self.validation_loader:
                val_loss,val_acc = self.test_epoch(val=True)
            else:
                val_loss,val_acc = test_loss,test_acc
            self.history['train_loss'] += [train_loss]
            self.history['train_acc'] += [train_acc]
            self.history['test_loss'] += [test_loss]
            self.history['test_acc'] += [test_acc]
            self.history['val_loss'] += [val_loss]
            self.history['val_acc'] += [val_acc]


            self.print_last_epoch()
            if self.saving_folder:
                self.save_history(name=name)
                self.save_model(name=name)
            if self.save_best and best_acc < self.history['val_acc'][-1]:
                self.save_model('best_')
                best_acc = self.history['val_acc'][-1]
                print(('BEST TESTING ACC : %.4f')%(self.history['test_acc'][-1]))
        # self.history['training_memory'] = self.estimate_training_memory()
        # print('TRAINING MEMORY')
        # print(self.history['training_memory'])
        # print(('TOTAL MB : %.4f')%(sum(list(self.history['training_memory'].values()))))
    def train_epoch(self):
        total = 0.
        correct = 0.
        running_loss = 0.
        if self.verbose:
            training_loader = tqdm(self.training_loader)
        else:
            training_loader = self.training_loader
        for i,(x,y) in enumerate(training_loader):
            batch_loss,batch_correct = self.train_batch(x,y)
            # if int(np.mod(i,10)) == 0: self.check_gradients() 
            total += x.size(0)
            correct = correct + batch_correct
            running_loss = 0.99 * running_loss + 0.01 * batch_loss.data.item()
            running_losses_dict = {'loss': running_loss}
            if self.verbose:
                training_loader.set_postfix(running_losses_dict)
        correct = correct/total
        return running_loss,correct
    def train_batch(self,x,y):
        self.optimizer.zero_grad()
        y = y.cuda().view(-1)
        x = x.cuda()
        # y = y.view(-1)
        # if self.devices == 'cpu':
        #     out = self.model(x)

        if len(self.devices) == 1:
            out = self.model(x)
        else:
            x = torch.nn.parallel.scatter(x, self.devices)
            out = torch.nn.parallel.parallel_apply(self.model, x)
            out = torch.nn.parallel.gather(out,self.devices[0])
        loss = self.criterion(out, y)
        corrects = (torch.max(out, 1)[1] == y.data).sum().item()
        loss.backward()
        self.optimizer.step()
        return loss,corrects

    def test_epoch(self,val=False):
        if val:
            test_loader = self.validation_loader
        else:
            test_loader = self.test_loader
        if self.mean_per_class_metric:
            correct = self.nb_classes*[0]
            total = self.nb_classes*[0]
        else:
            correct = 0
            total = 0
        running_loss = 0.
        for x,y  in test_loader:
            batch_loss,batch_corrects = self.test_batch(x,y)
            running_loss = 0.99 * running_loss + 0.01 * batch_loss.data.item()
            if self.mean_per_class_metric:
                total = [total[label] + int(sum(y.data == label)) for label in range(self.nb_classes)]
                correct = [i + j for i,j in zip(correct,batch_corrects)]
            else:
                total += x.size(0)
                correct += batch_corrects
        if self.mean_per_class_metric:
            correct = [float(i)/j if j != 0 else np.nan for i,j in zip(correct,total)]
            correct = np.nanmean(correct)
        else:
            correct = correct/total
        return running_loss,correct
              
    def test_batch(self,x,y):
        x = x.cuda()
        y = y.cuda().view(-1)
        out = self.model(x)
        losses = self.criterion(out, y)
        predicts = torch.max(out.data, 1)[1]
        if self.mean_per_class_metric:
            corrects = self.nb_classes*[0]
            for predict,label in zip(predicts,y.data):
                if predict == label:
                    corrects[label] += 1

        else:
            corrects = (predicts == y.data).sum().item()
        
        return losses,np.asarray(corrects)
    def save_history(self,name=''):
        print(('SAVING HISTORY AT %s')%(os.path.join(self.saving_folder, 'history' + name + '.json')))
        with open(os.path.join(self.saving_folder, 'history' + name + '.json'),'w') as fp:
            json.dump(self.history,fp)
    def save_model(self,name=''):
        print(('SAVING MODEL AT %s')%(os.path.join(self.saving_folder,'model' + name + '.pth.tar')))
        torch.save(self.model.state_dict(),os.path.join(self.saving_folder,'model' + name + '.pth.tar'))
    def print_last_epoch(self):
        print(('TRAINING ACC : %.4f')%(self.history['train_acc'][-1]))
        print(('TESTING ACC : %.4f')%(self.history['test_acc'][-1]))
        print(('VAL ACC : %.4f')%(self.history['val_acc'][-1]))
    def reset_history(self):
        self.history = {'test_acc'  : [],
                        'train_acc'  : [],
                        'test_loss' : [],
                        'train_loss' : [],
                        'val_loss' : [],
                        'val_acc' : []}
    def check_gradients(self):
        for stage in range(self.model.encoder.nb_stages):
            print(('GRADIENT AT STAGE %d')%(stage))
            print(self.model.get_gradients(stage))
        print('CLASSIFIER GRADIENT %d')
        print(self.model.get_gradients(-1))

    def plot_history(self):
        plots_dir = os.path.join(self.saving_folder,'plots')
        if not os.path.isdir(plots_dir):
            os.mkdir(plots_dir)
        for metric,curve in self.history.items():
            plt.figure()
            plt.plot(np.asarray(curve))
            plt.title(metric)
            plt.savefig(os.path.join(plots_dir,metric+'.jpg'))
    def estimate_training_memory(self):
        model_size_mb = sum([4*np.asarray(p.size()).prod() for module in self.model.modules() for p in module.parameters() if p.requires_grad])/1.e6
        x = Variable(next(iter(self.training_loader))[0],volatile=True).cuda()
        input_cost_mb = 4*np.asarray(x.size()).prod()/1.e6
        bytes = []
        for module in self.model.modules():
            x = module(x)
            if any([p.requires_grad for p in list(module.parameters())]):
                bytes += [4*np.asarray(x.size()).prod()]
        propagation_cost_mb = 2*sum(bytes)/1.e6
        return {'model' : model_size_mb, 'input' : input_cost_mb, 'propagation' : propagation_cost_mb}