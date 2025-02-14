import torch
from torchvision import models
from torch.utils.data import DataLoader, Subset
import time
import random
from torch.optim import SGD
import torch.nn as nn
from torchvision import datasets, transforms
from ComprehensiveVGGPruner import ComprehensiveVGGPruner
import numpy


def replace_layers(features, layer_idx, replace_indices, new_layers):
    """Replaces specific layers in a feature list with new layers.

    This function iterates through the provided feature list and replaces layers at specified indices with new layers from a separate list.

    Args:
        features: The original list of layers.
        layer_idx (int): Index of the layer group being modified.
        replace_indices: A list of indices indicating which layers in `features` should be replaced.
        new_layers: A list of new layers to be inserted.

    Returns:
        torch.nn.Sequential: A new Sequential model with the replaced layers.
    """
    new_features = []
    for i, layer in enumerate(features):
        if i in replace_indices:
            new_features.append(new_layers[replace_indices.index(i)])
        else:
            new_features.append(layer)
    return torch.nn.Sequential(*new_features)

def get_random_images(n=500):
    transform = transforms.Compose([
    transforms.Resize((255,255)),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])
    data_folder = 'imagenet-mini/val'
    data_dataset = datasets.ImageFolder(data_folder, transform=transform)
    num_samples = n
    indices = random.sample(range(len(data_dataset)), num_samples)
    subset_dataset = Subset(data_dataset, indices)
    return DataLoader(subset_dataset, batch_size=32, shuffle=True)



def prune_vgg16_comprehensive(model_org, prune_percentage):
    # Load a pretrained VGG16 model
    model = model_org

    # Create pruner and perform pruning
    pruner = ComprehensiveVGGPruner(model, prune_percentage)
    return pruner.prune_all_layers()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_parameter_reduction(original_model, pruned_model):
    if not hasattr(original_model, 'parameters') or not hasattr(pruned_model, 'parameters'):
        raise ValueError("Inputs must be valid PyTorch models")

    original_params = sum(p.numel() for p in original_model.parameters() if p.requires_grad)
    pruned_params = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)

    if original_params == 0:
        raise ValueError("Original model has no trainable parameters")

    return ((original_params - pruned_params) / original_params) * 100

def calculate_flops(model, input_size=(1, 3, 224, 224)):
    """Calculates the floating point operations (FLOPs) for a given model.

    This function iterates through the model's layers, registers forward hooks to count operations for Conv2d and Linear layers, performs a forward pass, and accumulates the total FLOPs.

    Args:
        model: The PyTorch model.
        input_size (tuple, optional): The input size for the model. Defaults to (1, 3, 224, 224).

    Returns:
        int: The total number of FLOPs.
    """
    model.eval()
    # Create custom forward hooks that don't rely on total_ops attribute
    def count_conv2d(m, x, y):
        x = x[0]
        cin = m.in_channels
        cout = m.out_channels
        kh, kw = m.kernel_size
        batch_size = x.size()[0]
        out_h = y.size(2)
        out_w = y.size(3)
        
        # ops: multiply-add is counted as 1 operation
        kernel_ops = kh * kw * cin
        bias_ops = 1 if m.bias is not None else 0
        ops_per_element = kernel_ops + bias_ops
        
        # total ops
        total_ops = batch_size * cout * out_h * out_w * ops_per_element
        return total_ops
    
    def count_linear(m, x, y):
        x = x[0]
        total_ops = x.size(0) * m.in_features * m.out_features
        if m.bias is not None:
            total_ops += x.size(0) * m.out_features
        return total_ops

    # Register hooks
    hooks = []
    total_ops = 0
    
    def register_hooks(module):
        if isinstance(module, torch.nn.Conv2d):
            hook = module.register_forward_hook(
                lambda m, x, y: setattr(m, 'total_ops', count_conv2d(m, x, y)))
            hooks.append(hook)
        elif isinstance(module, torch.nn.Linear):
            hook = module.register_forward_hook(
                lambda m, x, y: setattr(m, 'total_ops', count_linear(m, x, y)))
            hooks.append(hook)
    
    model.apply(register_hooks)
    
    # Perform forward pass
    input = torch.randn(input_size).to('mps')
    model(input)
    
    # Sum up the total operations
    for module in model.modules():
        if hasattr(module, 'total_ops'):
            total_ops += module.total_ops
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return total_ops

def fine_tuning(model):
    """Fine-tunes a given model using SGD optimizer and CrossEntropyLoss.

    This function performs fine-tuning on the provided model for a fixed number of epochs using a subset of ImageNet data. It utilizes Stochastic Gradient Descent (SGD) for optimization and CrossEntropyLoss as the loss function.

    Args:
        model: The PyTorch model to fine-tune.

    Returns:
        torch.nn.Module: The fine-tuned model.
    """
    model = model.to('mps')
    model.train()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for _ in range(3):
        running_loss = 0.0
        for data in get_random_images(n=300):
            inputs, labels = data
            inputs = inputs.to('mps')
            labels = labels.to('mps')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
    print("Finished fine-tuning!")
    return model

def comp_accuracy(model):
    model.eval()
    model = model.to('mps')
    correct = 0
    total = 0
    comp_time = 0
    with torch.no_grad():
        for data in get_random_images(n=1000):
            images, labels = data
            images = images.to('mps')
            labels = labels.to('mps')
            t1 = time.time()
            outputs = model(images)
            t2 = time.time()
            comp_time += (t2 - t1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return [(correct / total)*100 , comp_time]


def dict_to_csv(comp_times, accuracies, g_flops, sparsities):
    with open('vgg16_pruning_results.csv', 'w') as f:
        f.write("Sparsity,Accuracy,Comp Time,GFLOPs\n")
        for sparsity in sparsities:
            f.write(f"{sparsity},{accuracies[sparsity]},{comp_times[sparsity]},{g_flops[sparsity]}\n")


def main():
    """Main function to perform pruning, fine-tuning, and evaluation of VGG16 models.

    This function loads a pre-trained VGG16 model, iteratively prunes it with increasing sparsities, fine-tunes each pruned model, and evaluates its accuracy, computation time, and FLOPs. The results are then saved to a CSV file.
    """
    # Load a pretrained VGG16 model
    model_org = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model_org = model_org.to('mps')
    
    comp_times = {}
    accuracies = {}
    g_flops = {}
    sparsities = numpy.arange(0.0, 1.1, 0.1)
    org_accuracy = None
    for sparsity in sparsities:
        print(f"Pruning {model_org.__class__.__name__} with sparsity {sparsity:.2f}")
        pruned_vgg = prune_vgg16_comprehensive(model_org, prune_percentage=sparsity)
        pruned_vgg = fine_tuning(pruned_vgg)
        # torch.save(pruned_vgg, f'pruned_models/pruned_vgg_{sparsity:.2f}.pth')
        accuracy, comp_time = comp_accuracy(pruned_vgg)
        flops = calculate_flops(pruned_vgg)
        flops = flops / 1e9
        if sparsity == 0.0:
            org_accuracy = accuracy
            print(org_accuracy)
        # param_reduction = calculate_parameter_reduction(model_org, pruned_vgg)
        # param_org, param_pruned = count_parameters(model_org), count_parameters(pruned_vgg)
        # print(f"Original Parameters: {param_org}, Pruned Parameters: {param_pruned}")
        # # param_reduction = ((param_org - param_pruned) / param_org) * 100
        model_size = sum(param.numel() * param.element_size() for param in pruned_vgg.parameters()) / (1024 * 1024)
        print(f"Sparsity: {sparsity:.2f}, Accuracy Reduction: {org_accuracy - accuracy:.2f}%, Model Size: {model_size:.2f} MB, FLOPs: {flops:.2f} GFLOPs, Comp Time: {comp_time:.6f} s")
        
        comp_times[sparsity] = comp_time
        accuracies[sparsity] = accuracy
        g_flops[sparsity] = flops
    dict_to_csv(comp_times, accuracies, g_flops, list(sparsities))

if __name__ == '__main__':
    main()