import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
# from .Model import Model

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


# class SVCNN(Model):
class SVCNN(nn.Module):

    def __init__(self, name, nclasses=33, pretraining=True, cnn_name='vgg11', device=None):
        # sourcery skip: low-code-quality
        super(SVCNN, self).__init__()

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.FloatTensor([0.229, 0.224, 0.225]).to(device)

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                weights = models.ResNet18_Weights.DEFAULT if self.pretraining else None
                self.net = models.resnet18(weights=weights).to(device=self.device)
                self.net.fc = nn.Linear(512,33).to(self.device)
            elif self.cnn_name == 'resnet34':
                weights = models.ResNet34_Weights.DEFAULT if self.pretraining else None
                self.net = models.resnet34(weights=weights).to(self.device)
                self.net.fc = nn.Linear(512,33).to(self.device)
            elif self.cnn_name == 'resnet50':
                weights = models.ResNet50_Weights.DEFAULT if self.pretraining else None
                self.net = models.resnet50(weights=weights).to(self.device)
                self.net.fc = nn.Linear(2048,33).to(self.device)
        else:
            if self.cnn_name == 'alexnet':
                weights = models.AlexNet_Weights.DEFAULT if self.pretraining else None
                self.net_1 = models.alexnet(weights=weights).features.to(self.device)
                self.net_2 = models.alexnet(weights=weights).classifier.to(self.device)
            elif self.cnn_name == 'vgg11':
                weights = models.VGG11_Weights.DEFAULT if self.pretraining else None
                self.net_1 = models.vgg11(weights=weights).features.to(self.device)
                self.net_2 = models.vgg11(weights=weights).classifier.to(self.device)
            elif self.cnn_name == 'vgg16':
                weights = models.VGG16_Weights.DEFAULT if self.pretraining else None
                self.net_1 = models.vgg16(weights=weights).features.to(self.device)
                self.net_2 = models.vgg16(weights=weights).classifier.to(self.device)
            
            # Modify the classifier to handle the correct input size
            self.net_2._modules['6'] = nn.Linear(4096,33) 
            '''
            MY MODELNET DATASET HAS ONLY 33 MODELS
            '''

    @torch.jit.ignore # type: ignore
    def class_names(self):
        return ['airplane','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','door','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel', 'person','piano',
                         'plant','radio','range_hood','sink','stairs',
                         'stool','tent','toilet','tv_stand','vase','wardrobe','xbox']

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        
        y = self.net_1(x)
        
        # Apply adaptive pooling to get a fixed size output regardless of input dimensions
        y = nn.functional.adaptive_avg_pool2d(y, (7, 7))
        y = y.view(y.size(0), -1)
        
        return self.net_2(y)


# class MVCNN(Model):
class MVCNN(nn.Module):

    def __init__(self, name, model, nclasses=33, cnn_name='vgg11', num_views=12, device = None):
        super(MVCNN, self).__init__()

        self.nclasses = nclasses
        self.num_views = num_views
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.FloatTensor([0.229, 0.224, 0.225]).to(device)
        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            # exclude ResNet's avgpool and fc from features
            children = list(model.net.children())
            self.net_1 = nn.Sequential(*children[:-2]).to(self.device)
            self.pool = children[-2]  # this is AdaptiveAvgPool2d((1,1))
            self.net_2 = model.net.fc.to(self.device)
        else:
            self.net_1 = model.net_1.to(self.device)
            self.net_2 = model.net_2.to(self.device)

    def forward(self, x):
        # multi‐view case
        if x.dim() == 5:
            N, V, C, H, W = x.shape
            x = x.view(N * V, C, H, W)
            y = self.net_1(x)                   # (N*V, F, h, w)
            if not self.use_resnet:
                # only spatial‐pool per‐view in non‐ResNet branch
                y = F.adaptive_avg_pool2d(y, (7, 7))

            # now reshape and view‐pool across V
            Fdim, Hdim, Wdim = y.size(1), y.size(2), y.size(3)
            y = y.view(N, V, Fdim, Hdim, Wdim)
            y, _ = torch.max(y, dim=1)         # (N, F, H, W)

            if self.use_resnet:
                # apply ResNet’s avgpool **after** view‐pooling
                y = self.pool(y)               # (N, F, 1, 1)

            y = y.view(N, -1)                   # flatten to (N, F*H*W) or (N, F)
            return self.net_2(y)

        # single‐view case (unchanged)
        else:
            y = self.net_1(x)
            if not self.use_resnet:
                y = F.adaptive_avg_pool2d(y, (7, 7))
            y = y.view(y.size(0), -1)
            return self.net_2(y)