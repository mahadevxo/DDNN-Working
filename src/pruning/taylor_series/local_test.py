import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import re
import gc
import time
import os
import random


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
def create_results_file(filename=f'pruning_results_{time.time()}.csv'):
    with open(filename, 'w') as f:
        f.write("Pruning Amount, Final Accuracy, Time, Memory\n")
    return filename

def append_result(filename, pruning_amount, accuracy, compute_time, model_size):
    with open(filename, 'a') as f:
        f.write(f"{pruning_amount}%, {accuracy}%,{compute_time}, {model_size}\n")

def get_images(folder_path, num_samples=300):
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])
        data_dataset = datasets.ImageFolder(folder_path, transform=transform)
        indices = random.sample(range(len(data_dataset)), min(num_samples, len(data_dataset)))
        data_dataset = Subset(data_dataset, indices)
        
        return DataLoader(data_dataset, batch_size=32, shuffle=True, num_workers=1, pin_memory=True)

def test_model(model, device, test_loader):
    model.eval()
    model.to(device)
    correct, total, compute_time = 0, 0, 0
    tries = 3
    for _ in range(tries):
        start_time = time.time()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        compute_time += time.time() - start_time
    compute_time /= tries
    accuracy = 100 * correct / total
    model_size = os.path.getsize(model.state_dict()) / (1024 * 1024)  # Size in MB
    return accuracy, compute_time, model_size

def main():
    test_loader = get_images('./imagenet-mini/val')
    model_paths = sorted([f for f in os.listdir() if f.startswith("pruned_AlexNet") and f.endswith(".pth")])

    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    results_filename = create_results_file()
    for model_path in model_paths:
        print(f"Loading model from {model_path}")
        model = torch.load(model_path, map_location=device)
        accuracy, compute_time, model_size = test_model(model, device, test_loader)
        match = re.search(r'pruned_AlexNet_(\d+\.\d+)', model_path)
        pruning_amount = match[1] if match else "Unknown"
        append_result(results_filename, pruning_amount, accuracy, compute_time, model_size)
        print(f"Model: {model_path}, Accuracy: {accuracy:.2f}%, Time: {compute_time:.2f}s, Size: {model_size:.2f}MB")
if __name__ == "__main__":
    main()