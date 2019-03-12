import torch
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import _pickle as pickle
from sklearn.metrics import f1_score,confusion_matrix
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

class Tester():
    def __init__(self,test_loader,criterion,verbose=False,mean_per_class_metric=False):
        self.verbose = verbose
        self.test_loader = test_loader
        self.criterion = criterion
        self.nb_classes = self.test_loader.dataset.nb_classes()
        self.mean_per_class_metric = mean_per_class_metric
    def test(self,model):
        model = model.cuda()
        if self.mean_per_class_metric:
            correct = self.nb_classes*[0]
            total = self.nb_classes*[0]
        else:
            correct = 0
            total = 0
        running_loss = 0.
        for x,y  in self.test_loader:
            batch_loss,batch_corrects = self.test_batch(model,x,y)
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
        return {'loss' : running_loss, 'acc' : correct}
              
    def test_batch(self,model,x,y):
        x = x.cuda()
        y = y.cuda().view(-1)
        out = model(x)
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
    def get_high_confidence_images(self,model):
        model = model.cuda()
        corrects = []
        incorrects = []
        for i,(x,y) in enumerate(self.test_loader.dataset):
            x,y = x.cuda().unsqueeze(0), y.cuda()
            out = model(x)[0]
            out = F.softmax(out)
            if torch.argmax(out) == y:
              corrects += [[int(i),float(torch.max(out).cpu().detach().numpy()),int(y.cpu().numpy())]]
            else:
              incorrects += [[int(i),float(torch.max(out).cpu().detach().numpy()),int(y.cpu().numpy())]]
        return corrects,incorrects

    def plot_confusion_matrix(self,cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, [value for value in classes.values()], rotation=90)
        plt.yticks(tick_marks, [value for value in classes.values()])

        fmt = '.1f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        np.set_printoptions(precision=2)
        plt.savefig(os.path.join(self.saving_folder, 'confusion_matrix.png'),bbox_inches='tight')