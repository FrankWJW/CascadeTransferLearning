import json
import numpy as np
import os
import matplotlib.pyplot as plt

#MAGNITUDE PLOTS WITH RESPECT TO LR
#SEE HOW LAYERS ARE PRUNED BY THE ALGORITHM. SINCE THERE IS MORE WEIGHT DECAY, IT SHOULD BE SETTING TO ZEROS A LOT OF ACTIVATIONS, CHECK DIFFERENCE.
#INFLUENCE OF EARLY LAYERS FEATURES ON LATER CLASSIFIERS
#TEST REDUCING THE DIMENSIONALITY USING A CONV OR RELATIVE AVERAGE POOLING
#INFLUENCE OF STARTING THE ALGORITHM EARLIER OR LATER, SAME AS NUMBER OF BACK STAGES
##STORE CLASSIFIER AND USE IT AT N STAGE WHERE ACCARACY IS GREATER
##base_folders = ['Run1_cascade_back_stages_flowers'] + ['Run3_end_end_flowers'] #KEY RUNS ['Run8_end_end_cars'] + ['Run4_cascade_back_stages_cars']
## AS A TRANSFER LEARNING SELECTION FEATURE ALGORITHM

base_folders = ['Run3_end_end_flowers'] + ['Run1_cascade_50/flowers/']
nb_stages = 2*[16]
starting_stages = [8,8]
metrics_to_ignore = ['time','nb_parameters']
histories = {}
for starting_stage,last_stage,base_folder in zip(starting_stages,nb_stages,base_folders):
    histories[base_folder] = {}
    for stage in range(starting_stage,last_stage):
        with open(os.path.join(base_folder,str(stage),'history.json'), 'r') as fp:
            histories[base_folder][str(stage)] = json.load(fp)
for starting_stage,last_stage,(model,history) in zip(starting_stages,nb_stages,histories.items()):
    global_history = {key : [] for key in history[str(starting_stage)].keys() if not(key in metrics_to_ignore)}
    for stage in range(starting_stage,last_stage):
        for metric in global_history.keys():
            global_history[metric] += [history[str(stage)][metric]]
    histories[model] = global_history

for model,history in histories.items():
    test_acc = np.concatenate(history['test_acc'],axis=0)
    test_acc = test_acc[test_acc < 1]
    print(np.max(test_acc))
    plt.plot(test_acc,label=model)
    plt.legend(loc=0)
    print(model)
    print(('TEST ACC : %.4f')%(np.max(test_acc)))
plt.show()
    
