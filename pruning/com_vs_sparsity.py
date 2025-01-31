import torch
# import torch.nn as nn
# import torch.nn.utils.prune as prune
from torchvision.models import vgg11
import time
import matplotlib.pyplot as plt
import os
import re

computation_time_results = {}


def get_pruned_models():
    path = './vgg11_pruned/'
    prune_models = []
    prune_models.extend(path+file for file in os.listdir(path))
    print(prune_models)
    return prune_models

# def prune_model(model, sparsity):
#     prunable_layers = [
#         (name, module) for name, module in model.named_modules()
#         if isinstance(module, (nn.Conv2d))
#     ]
    
#     for _, module in prunable_layers:
#         prune.l1_unstructured(module, name="weight", amount=sparsity)
#         prune.remove(module, name="weight")
    
#     return model

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


print("Imported all libraries")

pruned_models = get_pruned_models()
for model_path in pruned_models:
    print(f"Starting with {model_path}")
    model = vgg11(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    time_taken = compute_time(model)
    print(f"Time taken: {time_taken}")
    #vgg11_0.7.pth
    if match := re.search(r'_(\d+\.\d+)\.pth', model_path):
        sparsity = float(match[1])
    else:
        print(f"Warning: Could not extract sparsity from {model_path}")
        continue
    computation_time_results[sparsity] = time_taken

computation_time_results = dict(sorted(computation_time_results.items()))

# for sparsity in numpy.arange(0.0, 1.01, 0.01):
#     print("Starting with")
#     print(f"Sparsity: {sparsity}")
#     model = vgg11(pretrained=True)
#     print("Model loaded")
#     model = prune_model(model, sparsity)
#     print("Model pruned")
#     time_taken = compute_time(model)
#     print(f"Time taken: {time_taken}")
#     computation_time_results.append(time_taken)
#     sparsityies.append(sparsity)


#plot
plt.plot(computation_time_results.keys(), computation_time_results.values())
plt.xlabel('Sparsity')
plt.ylabel('Computation Time')
plt.title('Computation Time vs Sparsity')
plt.savefig('computation_time_vs_sparsity.png')

#save results
with open('computation_time_results.txt', 'w') as file:
    file.write(str(computation_time_results))
    file.close()
print("Results saved")