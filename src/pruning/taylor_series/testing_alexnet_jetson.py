import torch
from torchvision import models as models
import numpy as np
import gc
import time
import glob
from torchvision import transforms, datasets
import random
from torch.utils.data import DataLoader, Subset


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
        
def create_results_file(filename='pruning_results_AlexNet_Jetson.csv'):
    with open(filename, 'w') as f:
        f.write("Pruning Amount, Final Accuracy, Time, Memory\n")
    return filename


def append_result(filename, pruning_amount, accuracy, compute_time, model_size):
    with open(filename, 'a') as f:
        f.write(f"{pruning_amount}%, {accuracy}%,{compute_time}, {model_size}\n")

def get_images(folder_path='imagenet-mini/train', num_samples=200):
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

def test(model, test_loader, device, final_test=True):
    model.eval()
    model.to(device)
    correct_top1 = 0
    total = 0
    compute_time = 0
    accuracies = []
    computation_times = []
    
    tries = 3 if final_test else 1
    for _ in range(tries):        
        with torch.inference_mode():
            for images, labels in test_loader:
                try:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    t1 = time.time()
                    outputs = model(images)
                    t2 = time.time()
                    compute_time += t2 - t1
                    
                    # Top-1 accuracy
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct_top1 += (predicted == labels).sum().item()
                    
                finally:
                    # Free up memory
                    del images, labels
                    if 'outputs' in locals(): 
                        del outputs
                    if 'predicted' in locals(): 
                        del predicted
                    clear_memory()
        
        accuracies.append(100.0 * correct_top1 / total if total > 0 else 0)
        computation_times.append(compute_time)
        clear_memory()
    del test_loader
    clear_memory()
    
    return [np.mean(accuracies), np.mean(computation_times)]

def get_model_size(model):
    total_size = sum(
        param.nelement() * param.element_size() for param in model.parameters()
    )
    # Convert to MB
    return total_size / (1024 ** 2)


def main():
    filename = create_results_file()
    model_paths = glob.glob("AlexNetPruned/*")
    model_paths.sort()
    print("Found models:")
    for model_path in model_paths:
        print(model_path)
    print("========================================"*3)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a test dataset
    test_loader = get_images()

    for model_path in model_paths:
        pruning_amount = model_path.split('_')[-1].replace('.pth', '')
        print(f"Testing model with pruning amount {pruning_amount}%")
        
        # Load the pruned model
        model = torch.load(model_path, map_location=device)
        
        # Get the size of the model
        model_size = get_model_size(model)
        
        # Test the model
        accuracy, compute_time = test(model, test_loader, device, final_test=True)
        
        # Append results to file
        append_result(filename, model_path.split('_')[-1].split('.')[0], accuracy, compute_time, model_size)
        print(f"Model size: {model_size:.2f} MB")
        print(f"Final accuracy: {accuracy:.2f}%")
        print(f"Compute time: {compute_time:.5f} seconds")
        clear_memory()
        del model
        clear_memory()
        print("Memory cleared.")
        print("========================================"*3)
        
if __name__ == "__main__":
    main()