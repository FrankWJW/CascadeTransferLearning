import torch
from torch import nn
import torchvision.models as models
from torch.nn.init import kaiming_normal_

class MLP(nn.Module):
    def __init__(self,input_size,output_size,hidden_units):#LEN(HIDDENUNITS) == LAYERS
        super(MLP, self).__init__()
        self.linear_layers = [nn.Sequential(*[nn.Linear(input_size,hidden_units[0]),nn.Dropout(0.5),nn.ReLU()])]
        self.linear_layers += [nn.Sequential(*[nn.Linear(in_features,out_features),nn.Dropout(0.5),nn.ReLU()]) for in_features,out_features in zip(hidden_units[0:-1],hidden_units[1::])]
        self.linear_layers += [nn.Linear(hidden_units[-1],output_size)]
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.init_weights()
    def init_weights(self):##CHECK INIT WEIGHTS COS OF SEQUENTIAL
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_normal(m.weight.data)
            elif isinstance(m, nn.ModuleList):
                for mm in m:
                    if isinstance(mm,nn.Sequential):
                        for mmm in mm:
                            if isinstance(mmm,nn.Linear):
                                kaiming_normal(mmm.weight.data)
            elif isinstance(m,nn.Sequential):
                for mm in m:
                    if isinstance(mm,nn.Linear):
                        kaiming_normal(mm.weight.data)
    def forward(self,x):
        x = x.view(x.size(0),-1)
        for linear_layer in self.linear_layers:
            x = linear_layer(x)
        return x


class TransferLearning(nn.Module):
    def __init__(self,encoder,nb_classes,max_pool=False,convs_compression=False):
        super().__init__()
        self.encoder = encoder
        self.nb_classes = nb_classes
        self.stage = -1
        self.nb_stages = self.encoder.nb_stages
        self.last_stages_features = self.encoder.last_stages_features
        if not max_pool:
            self.pool = nn.AvgPool2d(kernel_size=7,stride=3)
        else:
            self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.convs_compression = convs_compression
    def freeze_encoder(self,defreeze=False):
        for layer in self.encoder.parameters():
            layer.requires_grad = defreeze
    def get_gradients(self,stage):
        def mean_gradient(module):
            gradients = 0
            for m in module.parameters():
                if hasattr(m.grad,'data'):
                    grad_avg = m.grad.data.cpu().numpy().mean()
                    if grad_avg != None:
                        gradients += grad_avg
            return gradients
        if stage == 0 and hasattr(self.encoder,'conv1'):
            layer = self.encoder.conv1
        elif stage == -1 :
            layer = self.classifier
        else:
            layer = self.get_block_at_stage(stage)
        return mean_gradient(layer)
    def set_encoder_stage(self,stage):
        self.stage = stage
    def get_block_at_stage(self,stage):
        return self.encoder.get_block_at_stage(stage)
    def build_classifier(self,x,back_stages=0):
        self.back_stages = back_stages
        x = self.encoder(x,stage=self.stage,back_stages=self.back_stages)
        if self.back_stages != 0:
            # x = [self.last_compression(xx).view(xx.size(0),-1) for xx in x]
            last_compression = []
            if self.convs_compression:
                self.convs_compre = []
            for xx in x:
                if self.convs_compression:
                    conv = nn.Conv2d(xx.size(1),16,kernel_size=1)
                    kaiming_normal_(conv.weight.data)
                    self.convs_compre += [nn.Sequential(conv,nn.ReLU(inplace=True)).cuda()]
                    xx = self.convs_compre[-1](xx)
                xx = self.pool(xx)
                xx = xx.view(xx.size(0),-1)

                last_compression += [nn.Sequential(nn.Linear(xx.size(1),self.last_stages_features),nn.ReLU(inplace=True))]
                self.last_compression = nn.ModuleList(last_compression)
                [kaiming_normal_(layer[0].weight.data) for layer in self.last_compression]
                self.classifier = nn.Linear(self.last_stages_features,self.nb_classes)
                kaiming_normal_(self.classifier.weight.data)
            # x = torch.cat(x,1)
        else:
            # x = self.pool(x)
            self.last_compression = self.pool
            x = self.last_compression(x)
            x = x.view(x.size(0),-1)
            self.classifier = nn.Linear(x.size(1),self.nb_classes)
            kaiming_normal_(self.classifier.weight.data)

        # self.last_compression = nn.MaxPool2d(kernel_size=7,stride=1)
        # for m in self.last_compression.modules():
        #     if isinstance(m, nn.Conv2d):
        #         kaiming_normal_(m.weight.data)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # x = self.last_compression(x)
        # x = x.view(x.size(0),-1)

        # self.classifier = MLP(int(x.size(1)),self.nb_classes,[int(x.size(1)/10),int(x.size(1)/10)]).cuda()
        # self.classifier.init_weights()
        
        # self.classifier = nn.Linear((self.back_stages+1)*self.last_stages_features,self.nb_classes).cuda()
        
        # for m in self.classifier.modules():
        #     if isinstance(m, nn.Linear):
        #             kaiming_normal(m.weight.data)
        #     elif isinstance(m,nn.Sequential):
        #         for mm in m:
        #             if isinstance(mm,nn.Linear):
        #                 kaiming_normal(mm.weight.data)
    def forward(self,x):
        x = self.encoder(x,stage=self.stage,back_stages=self.back_stages)
        if self.back_stages != 0:
            if self.convs_compression:
                x = [conv(xx) for xx,conv in zip(x,self.convs_compre)]
            x = [self.pool(xx).view(xx.size(0),-1) for xx in x]
            x = sum([layer(xx) for layer,xx in zip(self.last_compression,x)])
            # x = torch.cat([layer(xx) for layer,xx in zip(self.last_compression,x)],1) 
        else:
            x = self.last_compression(x)
            x = x.view(x.size(0),-1)
        # x = x.view(x.size(0),-1)
        return self.classifier(x)
    def disable_early_layers(self):
        self.encoder.disable_early_layers(stage=self.stage)


class ResNetEncoder(nn.Module):
    def __init__(self,resnet_version='resnet18',avg_pool=False,full_model=False):
        super().__init__()
        if resnet_version == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.last_stages_features = 512
        elif resnet_version == 'resnet152':
            self.model = models.resnet152(pretrained=True)
            self.last_stages_features = 2048
        elif resnet_version == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            self.last_stages_features = 2048
        elif resnet_version == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.last_stages_features = 2048
        elif resnet_version == 'resnet34':
            self.model = models.resnet34(pretrained=True)
            self.last_stages_features = 512
        self.model_stages = list(self.model.layer1)+list(self.model.layer2)+list(self.model.layer3)+list(self.model.layer4)
        self.conv1 = nn.Sequential(self.model.conv1,self.model.bn1,self.model.relu,self.model.maxpool)
        self.nb_stages = len(self.model_stages) +1
        self.avg_pool = avg_pool
        if avg_pool:
            self.pool = nn.AvgPool2d(kernel_size=7,stride=1)
        self.full_model = full_model
    def forward(self,x,stage=-1,back_stages=0):
        if not self.full_model:
            x = self.conv1(x)
            if stage != 0:
                main_x = self.get_output_at_stage(x,stage=stage)
                if back_stages > 0 and stage - back_stages >= 0:
                    sec_x = [self.get_output_at_stage(x,stage=stage-i) if stage-i != 0 else x for i in range(2,back_stages+2)]
                    x = [main_x]+sec_x
                else:
                    x = main_x
            if self.avg_pool:
                x = self.pool(x)
        else:
            x = self.conv1(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            if self.avg_pool:
                x = self.pool(x)
        return x
    def get_output_at_stage(self,x,stage):
            stage_counter = 1
            for layer in self.model_stages:
                if stage_counter == stage:
                    return layer(x)
                else:
                    x = layer(x)
                stage_counter += 1
            return x
    def get_block_at_stage(self,stage):
        if stage == 0:
            return self.conv1
        else:
            stage_counter = 1
            for resblock in self.model_stages:
                if stage_counter == stage:
                    return resblock
                stage_counter += 1
            return resblock
    def disable_early_layers(self,stage):
        if stage != 0:
            for m in self.conv1.parameters():
                m.requires_grad = False
        stage_counter = 1
        if stage_counter < stage:
            for layer in self.model_stages:
                for param in layer.parameters():
                    param.requires_grad = False
                stage_counter += 1
                if stage_counter == stage:
                    break

class VGGEncoder(nn.Module):
    def __init__(self,vgg_version='vgg19'):
        super().__init__()

        cfg = { 'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}
        if vgg_version == 'vgg19':
            self.model = models.vgg19(pretrained=True)
            cfg = cfg['E']
            self.model_stages_idxs = [1,3,6,8,11,13,15,17,20,22,24,26,29,31,33,35]
            self.last_stages_features = 4096
        elif vgg_version == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            cfg = cfg['D']
            self.model_stages_idxs = [1,3,6,8,11,13,15,18,20,22,25,27,29]
            self.last_stages_features = 4096
        elif vgg_version == 'vgg13':
            self.model = models.vgg13(pretrained=True)
            cfg = cfg['B']
            self.model_stages_idxs = [1,3,6,8,11,13,16,18,21,23]
            self.last_stages_features = 4096
        self.nb_stages = len(self.model_stages_idxs)

        
    def forward(self,x,stage=-1,back_stages=0):
        main_x = self.get_output_at_stage(x,stage=stage)
        if back_stages > 0 and stage - back_stages >= 0:
            sec_x = [self.get_output_at_stage(x,stage=stage-i) if stage-i != 0 else x for i in range(1,back_stages+2)]
            x = [main_x]+sec_x
        else:
            x = main_x

        return x
    def get_output_at_stage(self,x,stage):
        nb_features = self.model_stages_idxs[stage]
        feature_counter = 0
        for layer in self.model.features:
            if feature_counter == nb_features:
                return layer(x)
            else:
                x = layer(x)
            feature_counter += 1
        return x
    def get_block_at_stage(self,stage):
        nb_features = self.model_stages_idxs[stage]-1
        feature_counter = 0
        for layer in self.model.features:
            if feature_counter == nb_features:
                return layer
            feature_counter += 1
    def disable_early_layers(self,stage):
        nb_features = self.model_stages_idxs[stage-1]
        feature_counter = 0
        for layer in self.model.features:
            for param in layer.parameters():
                param.requires_grad = False
            feature_counter += 1
            if feature_counter == nb_features:
                break





