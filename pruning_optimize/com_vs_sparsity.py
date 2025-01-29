import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision.models import vgg11
import numpy
import time
import matplotlib.pyplot as plt

computation_time_results = []
sparsityies = []
def prune_model(model, sparsity):
    prunable_layers = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d))
    ]
    
    for _, module in prunable_layers:
        prune.l1_unstructured(module, name="weight", amount=sparsity)
        prune.remove(module, name="weight")
    
    return model

def compute_time(model):
    model = model.cuda()
    model.eval()
    images = torch.randn(100, 3, 224, 224)
    start_time = time.time()
    for image in images:
        image = image.unsqueeze(0).cuda()
        _ = model(image)
    end_time = time.time()
    return end_time - start_time


print("Imported all libaries")

for sparsity in numpy.arange(0.0, 1.01, 0.01):
    print("Starting with")
    print(f"Sparsity: {sparsity}")
    model = vgg11(pretrained=True)
    print("Model loaded")
    model = prune_model(model, sparsity)
    print("Model pruned")
    time_taken = compute_time(model)
    print(f"Time taken: {time_taken}")
    computation_time_results.append(time_taken)
    sparsityies.append(sparsity)
    

with open("computation_time_results.csv", "w") as f:
    for I in range(len(sparsityies)):
        f.write(f"{sparsityies[I]},{computation_time_results[I]}\n")

plt.plot(numpy.arange(0.0, 1.01, 0.01), computation_time_results)
plt.xlabel("Sparsity")
plt.ylabel("Computation Time")
plt.title("Computation Time vs Sparsity")
plt.savefig("computation_time_vs_sparsity.png")