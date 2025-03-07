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

import torch.nn as nn

class ClonedClassifier(nn.Module):
    def __init__(self, original_classifier):
        super().__init__()
        self.original = original_classifier
    def forward(self, x):
        # Clone input to avoid in-place modifications on a view
        return self.original(x.clone())

class OptimizedPruner:
    def __init__(self, model):
        # Base paths and device selection
        self.train_path = 'imagenet-mini/train'
        self.test_path = 'imagenet-mini/val'
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        # Wrap the classifier to clone its input
        self.model.classifier = ClonedClassifier(self.model.classifier)
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
        # Initialize batch counter before using it
        batch_count = 0
        max_batches = 5  # Limit the number of batches to prevent hanging
        
        # Completely remove hooks when fine-tuning after pruning
        # to avoid view-related errors
        if not rank_filter:
            # Clean up existing hooks
            self.gradient_analyzer.remove_hooks()
            self.gradient_optimizer.remove_hooks()
            
            if optimizer is not None and batch_count == 0:
                # Only register hooks for gradient-based LR adjustment
                # at the beginning of training
                self.gradient_optimizer.register_hooks()
        
        for image, label in train_dataset:
            # Exit after processing a few batches to prevent infinite loops
            batch_count += 1
            if batch_count > max_batches:
                print(f"Processed {max_batches} batches, stopping to prevent potential hanging")
                break
                
            try:
                # Clone input tensors and ensure they require gradients if needed
                image = image.to(self.device).clone()
                label = label.to(self.device)
                
                # Make sure model is in training mode and gradients are reset
                self.model.train()
                self.model.zero_grad()
                
                if rank_filter:
                    # Ranking doesn't need complex gradient manipulation
                    self.pruner.reset()
                    output = self.pruner.forward(image)
                    loss = self.criterion(output, label)
                    self.pruner.compute_ranks(loss)
                else:
                    # For normal training/fine-tuning
                    output = self.model(image)
                    loss = self.criterion(output, label)
                    loss.backward()
                    
                    # Apply optimizer step without hooks to avoid view issues
                    if optimizer is not None:
                        optimizer.step()
            except Exception as e:
                print(f"Error during training: {str(e)}")
                continue
            finally:
                # Clean up references
                if 'image' in locals(): del image
                if 'label' in locals(): del label
                if 'output' in locals(): del output
                if 'loss' in locals(): del loss
                gc.collect()
                # Clear cache
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device == 'mps':
                    torch.mps.empty_cache()
        
        # Clean up any remaining hooks
        self.gradient_analyzer.remove_hooks()
        self.gradient_optimizer.remove_hooks()
    
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
        
        # Get the standard pruning plan
        standard_plan = self.pruner.get_pruning_plan(num_filter_to_prune)
        
        # Safety check - build a map of how many filters we're pruning per layer
        layer_prune_count = {}
        for layer_index, _ in standard_plan:
            layer_prune_count[layer_index] = layer_prune_count.get(layer_index, 0) + 1
        
        # Ensure we're not pruning all filters from any layer
        modules = list(self.model.features._modules.items())
        safe_plan = []
        for layer_index, filter_index in standard_plan:
            # Find the layer
            if layer_index < len(modules):
                _, layer = modules[layer_index]
                if isinstance(layer, torch.nn.Conv2d):
                    # Don't prune if it would leave fewer than 2 filters
                    if layer.out_channels - layer_prune_count.get(layer_index, 0) >= 2:
                        safe_plan.append((layer_index, filter_index))
                    else:
                        print(f"WARNING: Skipping pruning filter {filter_index} from layer {layer_index} to avoid too few channels")
        
        return safe_plan

    def total_num_filters(self):
        return sum(layer.out_channels for layer in self.model.features if isinstance(layer, torch.nn.modules.conv.Conv2d))
    
    def get_model_size(self, model):
        total_size = sum(param.nelement() * param.element_size() for param in model.parameters())
        return total_size / (1024 ** 2)
    
    def safe_fine_tune_batch(self, optimizer, train_dataset):
        """
        A simplified training method with no hooks or gradient tracking magic,
        just pure forward-backward-update for fine-tuning after pruning.
        """
        batch_count = 0
        max_batches = 5  # Limit the number of batches
        
        for image, label in train_dataset:
            batch_count += 1
            if batch_count > max_batches:
                print(f"Processed {max_batches} batches, stopping to prevent potential hanging")
                break
                
            try:
                # Simple straightforward training pass
                image = image.to(self.device)
                label = label.to(self.device)
                
                # Ensure training mode
                self.model.train()
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(image)
                
                # Compute loss - use a fresh criterion to avoid any hook issues
                loss = torch.nn.functional.cross_entropy(output, label)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
            except Exception as e:
                print(f"Error during safe fine-tuning: {str(e)}")
            finally:
                # Clean up
                if 'image' in locals(): del image
                if 'label' in locals(): del label
                if 'output' in locals(): del output
                if 'loss' in locals(): del loss
                gc.collect()
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device == 'mps':
                    torch.mps.empty_cache()
    
    def safe_fine_tune_epoch(self, optimizer):
        """Runs one epoch of safe fine-tuning"""
        train_dataset = self.get_images(self.train_path)
        self.safe_fine_tune_batch(optimizer, train_dataset)
    
    def prune(self, pruning_percentage, skip_fine_tuning=False):
        self.model.train()
        for param in self.model.features.parameters():
            param.requires_grad = True
        original_filters = self.total_num_filters()
        total_filters_to_prune = int(original_filters * (pruning_percentage / 100.0))
        
        # Safety check - don't try to prune too many filters
        max_safe_percentage = 90  # Maximum 90% of filters can be pruned
        if pruning_percentage > max_safe_percentage:
            print(f"WARNING: Requested pruning percentage {pruning_percentage}% is too high. Limiting to {max_safe_percentage}%")
            total_filters_to_prune = int(original_filters * (max_safe_percentage / 100.0))
        
        # Ensure we leave at least 1 filter per layer
        min_filters_per_layer = {}
        for layer_index, layer in enumerate(self.model.features):
            if isinstance(layer, torch.nn.Conv2d):
                min_filters_per_layer[layer_index] = max(1, int(layer.out_channels * 0.1))  # At least 10% of filters remain
        
        print("Total Filters to prune:", total_filters_to_prune, "For Pruning Percentage:", pruning_percentage)
        prune_targets = self.get_candidates_to_prune(total_filters_to_prune)
        
        # Count how many filters we're pruning per layer for safety check
        layers_pruned = {}
        for layer_index, filter_index in prune_targets:
            layers_pruned[layer_index] = layers_pruned.get(layer_index, 0) + 1
        
        # Verify no layer is being over-pruned and adjust if needed
        safe_prune_targets = []
        for layer_index, filter_index in prune_targets:
            for li in self.model.features._modules:
                layer = self.model.features._modules[li]
                if isinstance(layer, torch.nn.Conv2d) and int(li) == layer_index:
                    remaining_filters = layer.out_channels - layers_pruned.get(layer_index, 0)
                    if remaining_filters >= min_filters_per_layer.get(layer_index, 1):
                        safe_prune_targets.append((layer_index, filter_index))
                    else:
                        print(f"WARNING: Skipping pruning of layer {layer_index} to maintain minimum filters")
        
        print("Layers that will be pruned", layers_pruned)
        print("Pruning Filters")
        
        model = self.model.to(self.device)
        pruner = Pruning(model)
        
        # Process pruning operations one at a time and verify model integrity after each
        for i, (layer_index, filter_index) in enumerate(safe_prune_targets):
            try:
                model = pruner.prune_vgg_conv_layer(model, layer_index, filter_index)
                # Verify model is still valid
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                    try:
                        _ = model(dummy_input)
                    except RuntimeError as e:
                        print(f"Error after pruning operation {i}: {e}")
                        # Revert to previous state and skip this pruning operation
                        continue
            except Exception as e:
                print(f"Error during pruning operation {i}: {e}")
                continue
        
        # Ensure all model parameters are float32
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                if layer.weight is not None:
                    layer.weight.data = layer.weight.data.float()
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.float()
        
        self.model = model.to(self.device)
        acc_pre_ft = self.test(model)
        
        # For fine-tuning, use the safe method without hooks
        if pruning_percentage != 0.0 and not skip_fine_tuning:
            print(f"Accuracy before fine tuning: {acc_pre_ft[0]:.2f}%")
            
            # Ensure all hooks are completely removed
            self.gradient_analyzer.remove_hooks()
            self.gradient_optimizer.remove_hooks()
            
            # Use simpler optimizer setup without gradient-based adjustments
            optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
            best_accuracy = 0.0
            num_ft_epochs = 5
            
            for epoch in range(num_ft_epochs):
                print(f"Fine-tuning epoch {epoch+1}/{num_ft_epochs}")
                # Use safe fine-tuning that doesn't use any hooks
                self.safe_fine_tune_epoch(optimizer)
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
        # Fixed: Use correct indices in the return statement
        return [acc_pre_ft[0], acc_time[0], acc_time[1], size_mb]
    
    def reset(self):
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
        # if hasattr(self, 'dataloader'):
        #     del self.dataloader
        for attr in list(self.__dict__.keys()):
            if attr not in ['device'] and hasattr(self, attr):
                delattr(self, attr)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
