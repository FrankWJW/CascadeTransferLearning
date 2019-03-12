import numpy as np
from sklearn.preprocessing import Normalizer
from tqdm import tqdm
def gather_feature_vectors(model,dataset,verbose=True,cpu=False):
    fvs = []
    labels = []
    if verbose:
        print(('LOADING %d SAMPLES')%(len(dataset)))
        dataset = tqdm(dataset)
    for x,y in dataset:
        if not cpu:
            x = x.cuda()
        fvs += [model(x.unsqueeze(0)).view(-1).cpu().numpy()]
        labels += [y.numpy()]
    return np.asarray(fvs),np.asarray(labels)

def normalize_fvs(train_fvs,test_fvs):
    # normalizer = Normalizer()
    # normalizer = normalizer.fit(train_fvs)
    # return normalizer.transform(train_fvs),normalizer.transform(test_fvs)
    mean = np.mean(train_fvs,axis=0)
    std = np.std(train_fvs,axis=0) +1.-4
    train_fvs = np.asarray([(fv - mean)/std for fv in train_fvs])
    test_fvs = np.asarray([(fv - mean)/std for fv in test_fvs])
    return train_fvs,test_fvs
