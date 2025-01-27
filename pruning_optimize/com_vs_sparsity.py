import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision.models import vgg11
import numpy
import time
import matplotlib.pyplot as plt

computation_time = []

def prune_model(model, sparsity):
    prunable_layers = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    
    print(f"Pruning {len(prunable_layers)} layers with sparsity {sparsity}")
    
    params = []
    
    for _, module in prunable_layers:
        params.append(module, 'weight')
    
    prune.global_unstructured(
        params,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )
    
    for _, module in prunable_layers:
        prune.remove(module, 'weight')
        
    return model

def computation_time(model):
    model = model.cuda()
    model.eval()
    
    images = torch.randn(100, 3, 244, 244)
    
    start_time = time.time()
    
    for image in images:
        image = image.unsqueeze(0).cuda()
        _ = model(image)
    
    end_time = time.time()
    
    return end_time - start_time

for sparsity in numpy.linspace(0.0, 1.0, 0.1):
    model = vgg11(pretrained=True)
    model = prune_model(model, sparsity)
    time_taken = computation_time(model)
    computation_time.append(time_taken)

print("Pruning Complete")

plt.plot(numpy.linspace(0.0, 1.0, 0.1), computation_time)
plt.xlabel("Sparsity")
plt.ylabel("Computation Time")
plt.title("Computation Time vs Sparsity")
plt.savefig("computation_time_vs_sparsity.png")