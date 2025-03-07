from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
from torch import optim
import random
import time
import gc

from FilterPruner import FilterPruner
from Pruning import Pruning
from GradientFlowAnalyzer import GradientFlowAnalyzer
from GradientOptimizer import GradientOptimizer

class OptimizedPruner:
    def __init__(self, model):
        # Base paths and device selection
        self.train_path = 'imagenet-mini/train'
        self.test_path = 'imagenet-mini/val'
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.pruner = FilterPruner(self.model)
        # Initialize gradient analyzer/optimizer for unified control
        self.gradient_analyzer = GradientFlowAnalyzer(self.model)
        self.gradient_optimizer = GradientOptimizer(self.model)
    
    def get_images(self, folder_path, num_samples=5000):
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
    
    def train_batch(self, optimizer, train_dataset, rank_filter=False, epoch=None):
        # Register gradient hooks for analysis and optimization
        self.gradient_analyzer.register_hooks()
        self.gradient_optimizer.register_hooks()
        for image, label in train_dataset:
            try:
                image = image.to(self.device)
                label = label.to(self.device)
                self.model.zero_grad()
                inp = image
                if rank_filter:
                    self.pruner.reset()
                    output = self.pruner.forward(inp)
                else:
                    output = self.model(inp)
                loss = self.criterion(output, label)
                loss.backward()
                # Adjust learning rates based on gradient flow
                if optimizer is not None:
                    optimizer = self.gradient_optimizer.apply_gradient_based_lr(optimizer, epoch)
                if optimizer is not None:
                    optimizer.step()
            except Exception as e:
                print(f"Error during training: {str(e)}")
                continue
            finally:
                del image, label
                if 'inp' in locals(): 
                    del inp
                if 'output' in locals(): 
                    del output
                if 'loss' in locals():
                    del loss
                gc.collect()
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
    
    def train_epoch(self, optimizer=None, rank_filter=False, epoch=None):
        train_dataset = self.get_images(self.train_path)
        self.train_batch(optimizer, train_dataset, rank_filter, epoch)
    
    def test(self, model):
        
        self.model.eval()
        correct = total = compute_time = 0
        
        with torch.no_grad():
            for images, labels in self.get_images(self.test_path, num_samples=1000):
                images = images.to(self.device)
                labels = labels.to(self.device)
                images = images.float()
                t1 = time.time()
                outputs = model(images)
                t2 = time.time()
                compute_time += t2 - t1
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = float(correct/total)*100
        return [accuracy, compute_time]
    
    def get_candidates_to_prune(self, num_filter_to_prune):
        self.pruner.reset()
        self.train_epoch(rank_filter=True)
        self.pruner.normalize_ranks_per_layer()
        # Use gradient optimizer to prioritize filters if available
        standard_plan = self.pruner.get_pruning_plan(num_filter_to_prune)
        gradients = self.gradient_analyzer.gradients
        if not gradients:
            return standard_plan
        adjusted_plan = []
        for layer_index, filter_index in standard_plan:
            for key, grad_val in gradients.items():
                if str(layer_index) in key and grad_val < 1e-3:
                    adjusted_plan.append((layer_index, filter_index))
                    break
            else:
                adjusted_plan.append((layer_index, filter_index))
        if len(adjusted_plan) > num_filter_to_prune:
            adjusted_plan = adjusted_plan[:num_filter_to_prune]
        elif len(adjusted_plan) < num_filter_to_prune:
            remaining = [pt for pt in standard_plan if pt not in adjusted_plan]
            adjusted_plan.extend(remaining[:num_filter_to_prune - len(adjusted_plan)])
        return adjusted_plan

    def total_num_filters(self):
        return sum(layer.out_channels for layer in self.model.features if isinstance(layer, torch.nn.modules.conv.Conv2d))
    
    def get_model_size(self, model):
        total_size = sum(param.nelement() * param.element_size() for param in model.parameters())
        return total_size / (1024 ** 2)
    
    def prune(self, pruning_percentage):  # sourcery skip: extract-method
        self.model.train()
        for param in self.model.features.parameters():
            param.requires_grad = True
        original_filters = self.total_num_filters()
        total_filters_to_prune = int(original_filters * (pruning_percentage / 100.0))
        print("Total Filters to prune:", total_filters_to_prune, "For Pruning Percentage:", pruning_percentage)
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
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                if layer.weight is not None:
                    layer.weight.data = layer.weight.data.float()
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.float()
        self.model = model.to(self.device)
        acc_pre_ft = self.test(model)
        if pruning_percentage != 0.0:
            print(f"Accuracy before fine tuning: {acc_pre_ft[0]:.2f}%")
            optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
            best_accuracy = 0.0
            num_ft_epochs = 5
            for epoch in range(num_ft_epochs):
                print(f"Fine-tuning epoch {epoch+1}/{num_ft_epochs}")
                self.train_epoch(optimizer, rank_filter=False, epoch=epoch)
                val_res = self.test(self.model)
                print(f"Validation Accuracy: {val_res[0]:.2f}%")
                scheduler.step(val_res[0])
                if val_res[0] > best_accuracy:
                    best_accuracy = val_res[0]
        acc_time = self.test(self.model)
        print("Finished Pruning for", pruning_percentage)
        print(f"Accuracy after fine tuning: {acc_time[0]:.2f}%")
        size_mb = self.get_model_size(self.model)
        print(f"Model Size after fine tuning: {size_mb:.2f} MB")
        return [acc_pre_ft, acc_time[0], acc_time[3], size_mb]
    
    def reset(self):
        # ...existing code...
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'pruner'):
            del self.pruner
        if hasattr(self, 'gradient_analyzer'):
            self.gradient_analyzer.remove_hooks()
            del self.gradient_analyzer
        if hasattr(self, 'gradient_optimizer'):
            self.gradient_optimizer.remove_hooks()
            del self.gradient_optimizer
        if hasattr(self, 'dataloader'):
            del self.dataloader
        for attr in list(self.__dict__.keys()):
            if attr not in ['device'] and hasattr(self, attr):
                delattr(self, attr)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
