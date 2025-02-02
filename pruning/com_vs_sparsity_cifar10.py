import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy

# Define a pruning function
def prune_model_individual(model, amount):
    for _, module in model.named_modules():
        # Prune only Conv2d and Linear layers
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            if module.bias is not None:
                prune.l1_unstructured(module, name='bias', amount=amount)
    return model

# Load the dataset (using CIFAR-10 as an example)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
except Exception as e:
    print("Error loading datasets. Ensure you have internet access and sufficient storage.")
    raise e

test_loader = DataLoader(torch.utils.data.Subset(test_dataset, range(100)), batch_size=32, shuffle=False)

# Define evaluation functions
def evaluate_computation_time(model):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            _ = model(images)
    end_time = time.time()
    return end_time - start_time


def evaluate_model_accuracy(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (100 * (correct / total))

def computation_time_accuracy(model):
    model.eval()
    start_time = time.time()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    end_time = time.time()
    return end_time - start_time, (100 * (correct / total))

# Sparsity levels
sparsity_levels = numpy.arange(0.0, 1.1, 0.1)
computation_times = []
accuracies = []

for sparsity in sparsity_levels:
    print(f"Loading Model for {sparsity:.2f} sparsity")
    model = models.alexnet(pretrained=True)
    print("Pruning Model")
    model = prune_model_individual(model, amount=sparsity)
    print("Evaluating Computation Time and Accuracy")
    comp_time, accuracy = computation_time_accuracy(model)
    computation_times.append(comp_time)
    accuracies.append(accuracy)
    print(f"Sparsity: {sparsity:.1f}, Computation Time: {comp_time:.4f} seconds, Accuracy: {accuracy:.3f}%")

# Plot the graphs
plt.figure(figsize=(14, 6))

# Plot 1: Sparsity vs Computation Time
# plt.subplot(1, 2, 1)
plt.plot(sparsity_levels, computation_times, marker='o')
plt.title("Sparsity vs Computation Time")
plt.xlabel("Sparsity Level")
plt.ylabel("Computation Time (seconds)")
plt.grid(True)

# Plot 2: Sparsity vs Accuracy
# plt.subplot(1, 2, 2)
# plt.plot(sparsity_levels, accuracies, marker='o')
# plt.title("Sparsity vs Accuracy")
# plt.xlabel("Sparsity Level")
# plt.ylabel("Accuracy (%)")
# plt.grid(True)

plt.tight_layout()
plt.savefig("pruning_results_jetson.png")