# prune and save

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as models
from numpy import arange

def prune_model(model, sparsity):
    prunable_layers = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d))
    ]
    
    for _, module in prunable_layers:
        prune.l1_unstructured(module, name="weight", amount=sparsity)
        prune.remove(module, name="weight")
    
    return model

for sparsity in arange(0.0, 1.1, 0.1):
    print(f"Pruning at sparsity {sparsity:.1f}")
    model = models.vgg11(pretrained=True)
    model = prune_model(model, sparsity)
    torch.save(model.state_dict(), f"vgg11_{sparsity:.1f}.pth")
    print(f"Saved vgg11_{sparsity:.1f}.pth")