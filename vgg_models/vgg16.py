# VGG16 cats_dogs
from torchvision.models import vgg16 as vgg16
import torch.nn as nn
class Model:
    def __init__(self):
        self.model = self.create_model()
    
    def create_model(self):
        model = vgg16(pretrained=False)
        model.classifier[len(model.classifier)-1] = nn.Linear(4096, 2)
        return model