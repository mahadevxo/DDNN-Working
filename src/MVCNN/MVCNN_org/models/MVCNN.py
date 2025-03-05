import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).to(device)
std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).to(device)

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().float(), :]
    return x.view(xsize)


class SVCNN(Model):

    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='alexnet'):
        super(SVCNN, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).to(self.device)
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).to(self.device)

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,40)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,40)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048,40)
        else:
            if self.cnn_name == 'alexnet':
                self.net_features = models.alexnet(pretrained=self.pretraining).features
                self.net_classifier = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_features = models.vgg11(pretrained=self.pretraining).features
                self.net_classifier = models.vgg11(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_features = models.vgg16(pretrained=self.pretraining).features
                self.net_classifier = models.vgg16(pretrained=self.pretraining).classifier
            
            self.net_classifier._modules['6'] = nn.Linear(4096,40)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_features(x)
            return self.net_classifier(y.view(y.shape[0],-1))


class MVCNN(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='alexnet', num_views=12):
        super(MVCNN, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).to(self.device)
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).to(self.device)

        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_features = nn.Sequential(*list(model.net.children())[:-1])
            self.net_classifier = model.net.fc
        else:
            self.net_features = model.net_features
            self.net_classifier = model.net_classifier

    def forward(self, x):
        y = self.net_features(x)
        
        batch_size = int(x.shape[0]/self.num_views) #Batch Size
        y = y.view(batch_size, self.num_views, y.shape[-3], y.shape[-2], y.shape[-1]) #Reshaping
        y_max = torch.max(y, dim=1)[0] #Maxpooling
        y_flat = y_max.view(y_max.shape[0], -1) #Flatten Tensor
        output = self.net_classifier(y_flat) #Classifier
        return output