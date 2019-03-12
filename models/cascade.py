import torch
import torchvision
from torch.autograd import Variable
from tqdm import tqdm
import _pickle as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import numpy as np
import os
from datetime import datetime
import json

class CascadeLearning():
    def __init__(self,training_obj,optimizer,starting_nb_epochs=10,epochs_step=1,starting_stage=0):
        self.train = training_obj
        if self.train != None:
            self.nb_stages = self.train.model.encoder.nb_stages
            self.saving_folder = self.train.saving_folder
        self.starting_nb_epochs = starting_nb_epochs
        self.epochs_step = epochs_step
        self.starting_stage = starting_stage
        self.optimizer = optimizer
    def __call__(self,back_stages=2):
        #DISCONENCT RESIDUALS WHEN TRAINING
        #FREEZE LAYERS
        x_sample = next(iter(self.train.training_loader))[0].cuda()
        epochs = self.starting_nb_epochs
        histories = {}
        for stage in range(self.starting_stage,self.nb_stages):
            if self.saving_folder and not os.path.isdir(os.path.join(self.saving_folder,str(stage))):
                os.mkdir(os.path.join(self.saving_folder,str(stage)))
            print(('TRAINING CASCADE LAYER %d')%(stage))
            self.train.model.set_encoder_stage(stage=stage)
            self.train.model.disable_early_layers()
            self.train.model.build_classifier(x_sample,back_stages=back_stages)
            self.train.model = self.train.model.cuda()
            self.train.optimizer = self.optimizer(self.train.model,stage=stage,stage_layer_only=True)
            # self.train.optimizer = self.optimizer.plain_optimizer(self.train.model)
            if self.saving_folder is not None:
                self.train.saving_folder = os.path.join(self.saving_folder,str(stage))
            t0 = datetime.now()
            self.train(nb_epochs=epochs,drop_learning_rate=[epochs//2,3*epochs//4])
            self.train.optimizer.zero_grad()
            histories[stage] = self.train.history
            histories[stage]['time'] = [(datetime.now() - t0).seconds]
            histories[stage]['nb_parameters'] = [sum(p.numel() for p in self.train.model.parameters() if p.requires_grad)]
            self.train.reset_history()
            epochs += self.epochs_step
        with open(os.path.join(self.saving_folder,'history.json'), 'w') as fp:
            json.dump(histories, fp)

    def yield_models(self,model,back_stages,x_sample):
        epochs = self.starting_nb_epochs
        for stage in range(self.starting_stage,model.nb_stages):
            print(('YIELDING CASCADE LAYER %d')%(stage))
            model.set_encoder_stage(stage=stage)
            model.build_classifier(x_sample,back_stages=back_stages)
            yield model,stage
        # self.plot_cascade_learning_curves(histories)
    def plot_cascade_learning_curves(self,histories,metrics_to_ignore=['time','nb_parameters']):
        if not os.path.isdir(os.path.join(self.saving_folder,'plots')):
            os.mkdir(os.path.join(self.saving_folder,'plots'))
        plot_folder = os.path.join(self.saving_folder,'plots')
        global_history = {key : [] for key in histories[self.starting_stage].keys() if not(key in metrics_to_ignore)}
        vertical_lines_coordinates = []
        for stage in range(self.nb_stages-1):
            for metric in global_history.keys():
                global_history[metric] += [histories[stage][metric]]
            vertical_lines_coordinates += [len(histories[stage][metric])]

        for key,values in global_history.items():
            plt.figure()
            plt.plot(np.concatenate(values,axis=0))
            plt.ylabel(key)
            plt.xlabel('epochs')
            last_line_x = 0
            for i in vertical_lines_coordinates:
                plt.axvline(x=i+last_line_x)
                last_line_x += i
            plt.savefig(os.path.join(plot_folder,key+'.jpg'))
            plt.close()

class CascadeOptimizer():
    def __init__(self,optimizer,optimizer_params):
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
    def __call__(self,model,stage,stage_layer_only=False):
        parameters_group = []
        if not stage_layer_only:
            for i in range(stage):
                block_params = model.encoder.get_block_at_stage(i).parameters()
                parameters_group += [{'params' : filter(lambda p: p.requires_grad, block_params), 'lr' : self.optimizer_params['lr']/10**(stage), 'weight_decay' : self.optimizer_params['weight_decay']}]
        parameters_group += [{'params' : filter(lambda p: p.requires_grad, model.encoder.get_block_at_stage(stage).parameters()), 'lr' : self.optimizer_params['lr'], 'weight_decay' : self.optimizer_params['weight_decay']}]
        parameters_group += [{'params' : model.last_compression.parameters(), 'lr' : 10*self.optimizer_params['lr'], 'weight_decay' : self.optimizer_params['weight_decay']}]
        parameters_group += [{'params' : model.classifier.parameters(), 'lr' : 10*self.optimizer_params['lr'], 'weight_decay' : self.optimizer_params['weight_decay']}]
        self.last_optimizer = self.optimizer(parameters_group,**self.optimizer_params)
        return self.last_optimizer
    def plain_optimizer(self,model,learning_rate_reduction=100):
        optimizer_params = self.optimizer_params
        optimizer_params['lr'] /= learning_rate_reduction
        return self.optimizer(filter(lambda p: p.requires_grad, model.parameters()),**self.optimizer_params)





