from heapq import nsmallest
from operator import itemgetter
import torch.optim as optim
import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import time
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
class Pruning:
    def __init__(self, model):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
    
    def _replace_layers(self, model, i, indices, layers):
        return layers[indices.index(i)] if i in indices else model[i]
    
    def _get_next_conv(self, model, layer_index):
        next_conv = None
        offset = 1
        modules = list(model.features)
        while (layer_index + offset) < len(modules):
            candidate = modules[layer_index + offset]
            if isinstance(candidate, torch.nn.Conv2d):
                next_conv = candidate
                break
            offset += 1
        return next_conv
    
    def _get_next_conv_offset(self, model, layer_index):
        offset = 1
        modules = list(model.features)
        while (layer_index + offset) < len(modules):
            candidate = modules[layer_index + offset]
            if isinstance(candidate, torch.nn.Conv2d):
                break
            offset += 1
        return offset
    
    def _create_new_conv(self, conv, in_channels=None, out_channels=None):
        in_channels = conv.in_channels if in_channels is None else in_channels
        out_channels = conv.out_channels - 1 if out_channels is None else out_channels
        return torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
        )
        
    def _prune_conv_layer(self, conv, new_conv, filter_index, device = 'mps'):
        old_weights = conv.weight.data.cpu().numpy()
        new_weights = new_conv.weight.data.cpu().numpy()
        
        new_weights[:filter_index, :, :, :] = old_weights[:filter_index, :, :, :]
        new_weights[filter_index:, :, :, :] = old_weights[filter_index+1:, :, :, :]
        
        new_conv.weight.data = torch.from_numpy(new_weights).to(device)
        bias_numpy = conv.bias.data.cpu().numpy()
        
        bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
        bias[:filter_index] = bias_numpy[:filter_index]
        bias[filter_index:] = bias_numpy[filter_index+1:]
        
        new_conv.bias.data = torch.from_numpy(bias).to(device)
    
    def _prune_next_conv_layer(self, next_conv, new_next_conv, filter_index, device = 'mps'):
        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = new_next_conv.weight.data.cpu().numpy()
        
        new_weights[:, :filter_index, :, :] = old_weights[:, :filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index+1:, :, :]
        
        new_next_conv.weight.data = torch.from_numpy(new_weights).to(device)
        new_next_conv.bias.data = next_conv.bias.data.to(device)
    
    def _prune_last_conv_layer(self, model, conv, new_conv, layer_index, filter_index, device = 'mps'):
        model.features = torch.nn.Sequential(
            *(self.replace_layers(model.features, i, [layer_index], \
                [new_conv]) for i, _ in enumerate(model.features)))
        
        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            # Fix: Check if module is a Linear layer, not the model itself
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index += 1
        
        if old_linear_layer is None:
            raise ValueError(f"No linear layer found in classifier, Model: {model}")
        params_per_input_channel = old_linear_layer.in_features // conv.out_channels
        
        new_linear_layer = torch.nn.Linear(
            old_linear_layer.in_features - params_per_input_channel,
            old_linear_layer.out_features
        )
        
        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()
        
        new_weights[:, :filter_index * params_per_input_channel] = \
            old_weights[:, :filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel:] = \
            old_weights[:, (filter_index + 1) * params_per_input_channel:]
            
        new_linear_layer.weight.data = torch.from_numpy(new_weights).to(device)
        new_linear_layer.bias.data = old_linear_layer.bias.data.to(device)
        
        model.classifier = torch.nn.Sequential(
            *(self._replace_layers(model.classifier, i, [layer_index], \
                [new_linear_layer]) for i, _ in enumerate(model.classifier)))
        
        return model
    
    def prune_vgg_conv_layer(self, model, layer_index, filter_index, device='mps'):
        _, conv = list(model.features._modules.items())[layer_index]
        next_conv = self._get_next_conv(model, layer_index)
        new_conv = self._create_new_conv(conv)
        
        self._prune_conv_layer(conv, new_conv, filter_index)
        
        if next_conv is not None:
            # Fix: explicitly pass out_channels to keep the same number of filters
            next_new_conv = self._create_new_conv(next_conv, in_channels=next_conv.in_channels - 1, out_channels=next_conv.out_channels)
            self._prune_next_conv_layer(next_conv, next_new_conv, filter_index)
            # Replace specific layers in the Sequential rather than building tuples
            modules = list(model.features)
            modules[layer_index] = new_conv
            offset = self._get_next_conv_offset(model, layer_index)
            modules[layer_index + offset] = next_new_conv
            model.features = torch.nn.Sequential(*modules)
        else:
            # Use _prune_last_conv_layer to update classifier layers for the last conv layer
            model = self._prune_last_conv_layer(model, conv, new_conv, layer_index, filter_index, device)
        return model


class FilterPruner:
    def __init__(self, model):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.reset()
        
    def reset(self):
        self.filter_ranks = {}
        
    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}
        
        self.model.eval()
        self.model.zero_grad()
        
        activation_index = 0
        for layer_index, layer in enumerate(self.model.features):
            x = layer(x)
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer_index
                activation_index += 1
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)
        return x  # Changed: return the computed output, not self.model(x)
    
    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        
        taylor = activation * grad
        
        taylor = taylor.mean(dim=(0, 2, 3)).data
        
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to(self.device)
            
        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1
        
    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
        return nsmallest(num, data, itemgetter(2))
    
    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / torch.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()
            
    def get_pruning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
        
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)
        
        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i
        
        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))
        
        return filters_to_prune

class PruningFineTuner:
    def __init__(self, model):
        self.train_path = 'imagenet-mini/train'
        self.test_path = 'imagenet-mini/val'
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)  # move model to device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.pruner = FilterPruner(self.model)
        
    def get_images(self, folder_path, num_samples=2000):
        transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]),
    ])
        data_dataset = datasets.ImageFolder(folder_path, transform=transform)
        indices = random.sample(range(len(data_dataset)), num_samples)
        subset_dataset = Subset(data_dataset, indices)
        return DataLoader(subset_dataset, batch_size=32, shuffle=True, num_workers=1)
    
    def train_batch(self, optimizer, train_dataset, rank_filter=False):
        
        for image, label in train_dataset:
            image = image.to(self.device)
            label = label.to(self.device)
            
            self.model.zero_grad()
            input = image
            if rank_filter:
                output = self.pruner.forward(input)
                self.criterion(output, label).backward()
            else:
                # Fix: Calculate loss first and then call backward on it
                output = self.model(input)
                loss = self.criterion(output, label)
                loss.backward()
                optimizer.step()
    
    def train_epoch(self, optimizer = None, rank_filter = False):
        train_dataset = self.get_images(self.train_path)
        self.train_batch(optimizer, train_dataset, rank_filter)
            
            
    def test(self, model):
        self.model.eval()
        model = model.to(self.device)
        correct = 0
        total = 0
        compute_time = 0
        
        with torch.no_grad():
            for images, labels in self.get_images(self.test_path):
                # Fix: Properly move tensors to device and ensure return value is used
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Convert to float32 to ensure consistent tensor types
                images = images.float()
                t1 = time.time()
                outputs = model(images)
                t2 = time.time()
                compute_time += t2 - t1
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return([float(correct/total), compute_time])
    
    def get_candidates_to_prune(self, num_filter_to_prune):
        self.pruner.reset()
        self.train_epoch(rank_filter=True)
        self.pruner.normalize_ranks_per_layer()
        return self.pruner.get_pruning_plan(num_filter_to_prune)
    
    def total_num_filters(self):
        return sum(
            layer.out_channels
            for layer in self.model.features
            if isinstance(layer, torch.nn.modules.conv.Conv2d)
        )
    
    def get_model_size(self, model):
        total_size = sum(
            param.nelement() * param.element_size() for param in model.parameters()
        )
        # Convert to MB
        return total_size / (1024 ** 2)
    
    def prune(self, pruning_percentage):
        self.model.train()

        for param in self.model.features.parameters():
            param.requires_grad = True
        original_filters = self.total_num_filters()
        total_filters_to_prune = int(original_filters * (pruning_percentage / 100.0))
        print("Total Filters to prune:", total_filters_to_prune, "For Pruning Percentage:", pruning_percentage)

        # Rank and get the candidates to prune (exactly total_filters_to_prune)
        print('Ranking filters')
        prune_targets = self.get_candidates_to_prune(total_filters_to_prune)
        layers_pruned = {}
        for layer_index, filter_index in prune_targets:
            layers_pruned[layer_index] = layers_pruned.get(layer_index, 0) + 1
        print("Layers that will be pruned", layers_pruned)

        print("Pruning Filters")
        model = self.model.to(self.device)
        pruner = Pruning(model)
        for layer_index, filter_index in prune_targets:
            model = pruner.prune_vgg_conv_layer(model, layer_index, filter_index)

        # After pruning, convert model weights to float32
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                if layer.weight is not None:
                    layer.weight.data = layer.weight.data.float()
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.float()

        self.model = model.to(self.device)
        pruned_ratio = 100 * float(self.total_num_filters())/original_filters
        print(f"Filters Pruned, {pruned_ratio:.2f}% of original left")

        # Test and fine tune model once after pruning
        self.test(model)
        print("Fine Tuning")
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
        best_accuracy = 0.0
        num_finetuning_epochs = 10  # Increased number of epochs
        for epoch in range(num_finetuning_epochs):
            print(f"Epoch {epoch+1}/{num_finetuning_epochs}")
            self.train_epoch(optimizer, rank_filter=False)
            val_accuracy = self.test(self.model)[0]
            print(f"Validation Accuracy: {(val_accuracy*100):.2f}%")
            scheduler.step(val_accuracy)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
        print(f"Accuracy after fine tuning: {(self.test(self.model))[0]*100:.2f}%")
        size_mb = self.get_model_size(self.model)
        print(f"Model Size after fine tuning: {size_mb:.2f} MB")

        print("Finished")
        print(f"Final Accuracy: {(self.test(self.model)*100)[0]:.2f}%")
        print(f'Final Model Size: {self.get_model_size(self.model):.2f} MB')

def heuristic_search(original_model, min_accuracy, max_iter=6):
    """
    Heuristic search to find the best pruning percentage that gets final accuracy
    as close to (but not lower than) min_accuracy, optimized for lower memory usage.
    """
    original_state = original_model.state_dict()
    lower, upper = 0.0, 100.0
    best_percentage = 0.0
    best_final_acc = 0.0
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for _ in range(max_iter):
        mid = (lower + upper) / 2.0
        trial_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
        trial_model.load_state_dict(original_state)
        pruner = PruningFineTuner(trial_model)
        pruner.prune(pruning_percentage=mid)
        final_acc = pruner.test(pruner.model)[0]
        if final_acc >= min_accuracy:
            best_percentage = mid
            best_final_acc = final_acc
            lower = mid
        else:
            upper = mid
        del pruner
        del trial_model
        if device == 'cuda':
            torch.cuda.empty_cache()
    print(f"Best pruning percentage: {best_percentage:.2f}% yields final accuracy: {best_final_acc*100:.2f}%")
    return best_percentage

def main():
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    pruning_fine_tuner = PruningFineTuner(model)
    print(f"Original Accuracy: {(pruning_fine_tuner.test(model)[0]*100):.2f}%")
    print(f'Original Model Size: {pruning_fine_tuner.get_model_size(model):.2f} MB')
    del pruning_fine_tuner
    min_acc = float(input("Enter minimum acceptable accuracy (0-1): "))
    print(f"Minimum Accuracy: {min_acc*100}%")
    best_percentage = heuristic_search(model, min_acc)
    print(f"Recommended pruning percentage: {best_percentage:.2f}%")

if __name__ == '__main__':
    main()