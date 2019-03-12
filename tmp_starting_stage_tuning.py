
import _pickle as pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import json

base_folder = './Run3_cascade_tune_starting_stage_18'
datasets = ['flowers','calltech']
nb_stages = 9
starting_stages = [2,3,4,5,6,7,8]
metrics_to_ignore = ['time','nb_parameters']
nb_epochs_per_stage = 30
histories = {}
for dataset in datasets:
    histories[dataset] = {}
    for starting_stage in starting_stages:
        histories[dataset][starting_stage] = {}
        for stage in range(starting_stage,nb_stages):
            with open(os.path.join(base_folder,dataset,str(starting_stage),str(stage),'history.json'), 'r') as fp:
                histories[dataset][starting_stage][stage] = json.load(fp)
for dataset,starting_stage_history in histories.items():
    for starting_stage,stage_history in starting_stage_history.items():
        global_history = {key : [] for key in stage_history[starting_stage].keys() if not(key in metrics_to_ignore)}
        for stage in range(starting_stage,nb_stages):
            for metric in global_history.keys():
                global_history[metric] += [np.asarray(stage_history[stage][metric]).flatten()]
        histories[dataset][starting_stage] = {metric : np.concatenate(values,axis=0) for metric,values in global_history.items()}

for dataset,starting_stages_history in histories.items():
    plt.figure()
    for starting_stage,starting_stage_history in starting_stages_history.items():
        test = starting_stage_history['test_acc']
        test = test[test< 1]
        epochs = np.arange(0,len(test),1)+ starting_stage*nb_epochs_per_stage
        plt.plot(epochs,test,label=str(starting_stage))
        print('STARTING STAGE BEST')
        print(('TEST ACC : %.4f')%(np.max(test)))
    plt.title(dataset)
    plt.xlabel('EPOCHS')
    plt.ylabel('ACC')
    plt.legend(loc=0)
plt.show()
    
