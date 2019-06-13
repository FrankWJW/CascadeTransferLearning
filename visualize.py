import _pickle as pickle
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import json
import argparse
import seaborn as sb
import pandas as pd
matplotlib.style.use('seaborn')

# PERFORMANCE VS MEMORY PLOT
sb.set()
class ColourYielder():
    def __init__(self,colours='bgrcmyk'):
        self.colours = colours
        self.generator = iter(self.colours)
    def next(self):
        colour = next(self.generator,None)
        if colour == None:
            self.reset()
            colour = next(self.generator,None)
        return colour
    def reset(self):
        self.generator = iter(self.colours)
    def create_colour_dict(self,indexes):
        return {index : color for index,color in zip(indexes,self.colours[0:len(indexes)])}
class Renderer():
    def __init__(self,folder,datasets=['calltech','flowers','texture'],colour_yielder=ColourYielder(),history_name='history.json',cascade=False,metrics_to_ignore=['time','nb_parameters'],tuning_history=False):
        self.folder = folder
        self.datasets = datasets
        self.cascade = cascade
        self.history_dict = {}
        self.metrics_to_ignore = metrics_to_ignore
        self.colour_yielder = colour_yielder
        if self.folder != None:
            for dataset in datasets:
                self.history_dict[dataset] = {}
                path = os.path.join(self.folder,dataset)
                self.history_dict[dataset] = {int(run_index) : json.load(open(os.path.join(path,run_index,history_name),'r')) for run_index in os.listdir(path)}
                if tuning_history:
                    tuning_histories = {int(run_index) : json.load(open(os.path.join(path,run_index,'history_tuning.json'),'r')) for run_index in os.listdir(path)}
                    for run_index in self.history_dict[dataset].keys():
                        for metric in self.history_dict[dataset][run_index].keys():
                            self.history_dict[dataset][run_index][metric] = np.concatenate((self.history_dict[dataset][run_index][metric],tuning_histories[run_index][metric]))
            if cascade:
                for dataset,run_info in self.history_dict.items():
                    for run_index,histories in run_info.items():
                        stages = [int(stage) for stage in histories.keys()]
                        global_history = {key : [] for key in histories[str(stages[0])].keys() if not(key in self.metrics_to_ignore)}
                        starting_stage = min(stages)
                        last_stage = max(stages)
                        for stage in range(int(starting_stage),int(last_stage)+1):
                            for metric in global_history.keys():
                                global_history[metric] += [histories[str(stage)][metric]]
                        self.history_dict[dataset][run_index] = {key : history for key,history in global_history.items()}
    def plot_learning_curves(self):
        plt.figure()
        if self.cascade:
            for dataset in self.datasets:
                to_plot = []
                for run_index,history in self.history_dict[dataset].items():
                    to_plot += [np.concatenate(history['test_acc'],axis=0)]
                to_plot = np.mean(to_plot,axis=0)
                plt.plot(to_plot,label=dataset,linewidth=2.0)
        else:
            for dataset in self.datasets:
                to_plot = []
                for run_index,history in self.history_dict[dataset].items():
                    to_plot += [history['test_acc']]
                to_plot = np.mean(to_plot,axis=0)
                plt.plot(to_plot,label=dataset,linewidth=2.0)
        plt.legend(loc=0)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')


    def print_mean_acc(self):
        for dataset in self.datasets:
            print(('\n%s')%(dataset))
            test_accs = []
            for run_index,history in self.history_dict[dataset].items():
                if self.cascade:
                    test_acc = np.concatenate(history['test_acc'],axis=0)
                else:
                    test_acc = history['test_acc']
                # print(('RUN %d')%(run_index))
                # print(('TEST ACC %.3f')%(test_acc[-1]))
                # print(('MAX AT EPOCH %.3f')%(np.argmax(history['test_acc'])))
                test_accs += [test_acc[-1]]
            print(('MEAN TEST ACC %.3f')%(np.mean(test_accs)))
            print(('STD TEST ACC %.3f')%(np.std(test_accs)))
            print(('BEST TEST ACC %.3f')%(np.max(test_accs)))    
    def plot_back_stages_tuning_curves(self):
        # sb.set(style="whitegri`d")
        matplotlib.rc('xtick', labelsize=15) 
        matplotlib.rc('ytick', labelsize=15) 
        for dataset in self.datasets:
            plt.figure()
            plt.title(dataset.capitalize(),fontsize=15)
            plt.xlabel('Number of residuals',fontsize=15)
            plt.ylabel('Accuracy',fontsize=15)
            labels = []
            to_plot = []
            for run_index,history in self.history_dict[dataset].items():
                # plt.plot(history['test_acc'],label=str(run_index))
                labels += [int(run_index)]
                to_plot += [[x[-1] for x in history['test_acc']]]

            # to_plot = np.concatenate(to_plot,axis=1)
            tmp = np.argsort(labels)
            labels = np.asarray(labels)[tmp]
            to_plot = np.asarray(to_plot)[tmp]
            color_dict = self.colour_yielder.create_colour_dict(range(0,len(to_plot[0])))
            legend_trace = []
            for label,values in zip(labels,to_plot):
                for i,stage_val in enumerate(values):
                    if i not in legend_trace:
                        plt.scatter(label,stage_val,color=[color_dict[i]],label=i)
                        legend_trace += [i]
                    else:
                        plt.scatter(label,stage_val,color=[color_dict[i]])
                self.colour_yielder.reset()
            plt.boxplot(x=to_plot.T,labels=labels,positions=labels)
            # plt.ylim()
            # plt.legend(loc=2,ncol=9)
    def plot_starting_stage_tuning_curves(self):
        # plt.rcParams.update({'font.size': 50})
        matplotlib.rc('xtick', labelsize=15) 
        matplotlib.rc('ytick', labelsize=15) 
        for dataset in self.datasets:
            fig = plt.figure()
            plt.title(dataset.capitalize(),fontsize=15)
            plt.xlabel('Starting stage',fontsize=15)
            plt.ylabel('Accuracy',fontsize=15)
            labels = []
            to_plot = []
            for run_index,history in self.history_dict[dataset].items():
                # plt.plot(history['test_acc'],label=str(run_index))
                labels += [int(run_index)]
                to_plot += [[x[-1] for x in history['test_acc']]]

            tmp = np.argsort(labels)
            labels = np.asarray(labels)[tmp]
            to_plot = np.asarray(to_plot)[tmp]
            # sb.boxplot(x=list(to_plot.values()))
            # print(labels)
            # print([x[-1] for x in to_plot.tolist()])
            color_dict = self.colour_yielder.create_colour_dict(labels)
            legend_trace = []
            for label,values in zip(labels,to_plot):
                for i,stage_val in enumerate(values):
                    if label+i not in legend_trace:
                        plt.scatter(label,stage_val,color=[color_dict[label+i]],label=label+i)
                        legend_trace += [label+i]
                    else:
                        plt.scatter(label,stage_val,color=[color_dict[label+i]])
                self.colour_yielder.reset()
            plt.boxplot(x=to_plot.T,labels=labels,positions=labels)
            plt.legend(loc=0,fontsize=12)
            # for axe in fig.get_axes():
            #     axe.set_fontsize(18)


    def plot_memory_complexity(self):
        plt.figure()
        for algorithm,models in MEMORY_DICT.items():
            for model,values in models.items():
                plt.scatter(len(values[1])*[values[0]],values[1],c=MEMORY_COLOURS[algorithm],marker=MEMORY_DOTS[model],label=algorithm + ' ' + model)
        plt.xlabel('Training Memory (MB)')
        plt.ylabel('Accuracy')
        plt.legend(loc=0)
          
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str,default='E:\git\CascadeTransferLearning\data')
parser.add_argument('--history_name',type=str,default='history.json')
parser.add_argument('--datasets',type=list,default=['calltech','flowers','texture'])
parser.add_argument('--cascade',action='store_true',default=False)
parser.add_argument('--plot_starting_stage',action='store_true',default=False)
parser.add_argument('--plot_back_stages',action='store_true',default=False)
parser.add_argument('--plot_learning_curves',action='store_true',default=False)
parser.add_argument('--tuning_history',action='store_true',default=False)
parser.add_argument('--plot_memory_complexity',action='store_true',default=False)
args = parser.parse_args()
colours =  [(0.,0.,0.),(192./255,192./255,192./255)]+sb.color_palette('Paired')
MEMORY_DICT = {'FT' : {'ResNet-34' : (912,(0.92,0.93,0.70)), 'ResNet-50' : (1213,(0.92,0.93,0.72)), 'ResNet-101' : (2108,(0.92,0.93,0.72))},
               'CTL' : {'ResNet-34' : (723,(0.90,0.89,0.67)), 'ResNet-50' : (735,(0.92,0.91,0.71)), 'ResNet-101' : (750,(0.92,0.91,0.71))}}
MEMORY_COLOURS = {'FT' : 'r','CTL' : 'b'}
MEMORY_DOTS = {'ResNet-34' : '^','ResNet-50' : 'o','ResNet-101' : 'x'}
# [(255, 255, 255),(192, 192, 192),(128, 128, 128),(0,0,0),(255,0,0),(128,0,0),(255,255,0),(128,128,0),
           # (0,255,0),(0,128,0),(0,255,255),(0,128,128),(0,0,255),(0,0,128),(255,0,255),(128,0,128)]
colours = ColourYielder(colours=colours)
renderer = Renderer(folder=args.folder if len(args.folder) > 0 else None,history_name=args.history_name,datasets=args.datasets,cascade=args.cascade,colour_yielder=colours,tuning_history=args.tuning_history)
# renderer.print_mean_acc()
if args.plot_starting_stage:
    renderer.plot_starting_stage_tuning_curves()
    plt.show()
elif args.plot_back_stages:
    renderer.plot_back_stages_tuning_curves()
    plt.show()
elif args.plot_learning_curves:
    renderer.plot_learning_curves()
    plt.show()
elif args.plot_memory_complexity:
    renderer.plot_memory_complexity()
    plt.show()
else:
    renderer.print_mean_acc()