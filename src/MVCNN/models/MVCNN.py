import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device)
std = torch.FloatTensor([0.229, 0.224, 0.225]).to(device)

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class SVCNN(Model):

    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='vgg11', device=None):
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
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.FloatTensor([0.229, 0.224, 0.225]).to(device)

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining).to(device=self.device)
                self.net.fc = nn.Linear(512,40).to(self.device)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining).to(self.device)
                self.net.fc = nn.Linear(512,40).to(self.device)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining).to(self.device)
                self.net.fc = nn.Linear(2048,40).to(self.device)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features.to(self.device)
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier.to(self.device)
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features.to(self.device)
                self.net_2 = models.vgg11(pretrained=self.pretraining).classifier.to(self.device)
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features.to(self.device)
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier.to(self.device)
            
            self.net_2._modules['6'] = nn.Linear(4096,40)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        y = self.net_1(x)
        y = y.view(y.size(0), -1)
        assert y.size(1) == self.net_2[0].in_features, f"Expected {self.net_2[0].in_features}, got {y.size(1)}"
        return self.net_2(y)


class MVCNN(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=12, device = None):
        super(MVCNN, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.FloatTensor([0.229, 0.224, 0.225]).to(device)
        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1]).to(self.device)
            self.net_2 = model.net.fc.to(self.device)
        else:
            self.net_1 = model.net_1.to(self.device)
            self.net_2 = model.net_2.to(self.device)

    def forward(self, x):
        y = self.net_1(x)  # Process input through net_1
        y = y.view(y.size(0), -1)  # Flatten the tensor
        print(f"Debug: Output shape after net_1: {y.shape}")  # Debugging output

        # Dynamically adjust net_2 input features if mismatch occurs
        expected_features = self.net_2[0].in_features
        if y.size(1) != expected_features:
            print(f"Warning: Adjusting net_2 input features from {expected_features} to {y.size(1)}")
            self.net_2[0] = nn.Linear(y.size(1), self.net_2[0].out_features).to(self.device)

        return self.net_2(y)  # Pass reshaped tensor to net_2

