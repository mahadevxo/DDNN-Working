from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
from torch import optim
import random
import time
import gc
import sys
sys.path.append('./taylor_series/')
from FilterPruner import FilterPruner
from Pruning import Pruning
import numpy as np

class PruningFineTuner:
    def __init__(self, model):
        self.train_path = 'imagenet-mini/train'
        self.test_path = 'imagenet-mini/val'
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.pruner = FilterPruner(self.model)
        self._clear_memory()
        
    def _clear_memory(self):
        """Helper method to clear memory efficiently"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
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
        indices = random.sample(range(len(data_dataset)), min(num_samples, len(data_dataset)))
        data_dataset = Subset(data_dataset, indices)
        
        return DataLoader(data_dataset, batch_size=32, shuffle=True, num_workers=1, pin_memory=True)
    
    def train_batch(self, optimizer, train_loader, rank_filter=False):
        self.model.train()
        self.model.to(self.device)
        
        for batch_idx, (image, label) in enumerate(train_loader):
            try:
                with torch.autograd.set_grad_enabled(True):
                    image = image.to(self.device, non_blocking=True)
                    label = label.to(self.device, non_blocking=True)
                    
                    # Zero gradients before forward pass
                    if optimizer is not None:
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        self.model.zero_grad(set_to_none=True)
                    
                    # Forward pass
                    if rank_filter:
                        self.pruner.reset()
                        output = self.pruner.forward(image)
                    else:
                        output = self.model(image)
                    
                    # Compute loss and backprop
                    loss = self.criterion(output, label)
                    loss.backward()
                    
                    # Update weights if optimizer provided
                    if optimizer is not None:
                        optimizer.step()
                        
            except Exception as e:
                print(f"Error during training batch {batch_idx}: {str(e)}")
                continue
            finally:
                # Explicitly delete tensors to free memory
                del image, label
                if 'output' in locals(): 
                    del output
                if 'loss' in locals(): 
                    del loss
                self._clear_memory()
            
            # Periodically clear cache during training
            if batch_idx % 10 == 0:
                self._clear_memory()
    
    def train_epoch(self, optimizer=None, rank_filter=False):
        train_loader = self.get_images(self.train_path, num_samples=2000)
        self.train_batch(optimizer, train_loader, rank_filter)
        del train_loader
        self._clear_memory()
        
    def _get_mean(self, list1: list) -> float:
        return sum(list1) / len(list1) if list1 else 0.0
            
    def test(self, model, final_test=False):
        model.eval()
        model.to(self.device)
        correct_top1 = 0
        total = 0
        compute_time = 0
        accuracies = []
        computation_times = []
        
        tries = 10 if final_test else 1
        for _ in range(tries):
            test_loader = self.get_images(self.test_path, num_samples=500)
            
            with torch.inference_mode():
                for images, labels in test_loader:
                    try:
                        images = images.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                        
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
                        self._clear_memory()
            
            accuracies.append(100.0 * correct_top1 / total if total > 0 else 0)
            computation_times.append(compute_time)
            self._clear_memory()
        
        # Clean up test loader
        del test_loader
        self._clear_memory()
        
        return [self._get_mean(accuracies), self._get_mean(computation_times)]
    
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
    
    def prune(self, pruning_percentage):  # sourcery skip: extract-method, low-code-quality
        self.model.train()
        
        # Enable gradients for pruning
        for param in self.model.features.parameters():
            param.requires_grad = True
            
        original_filters = self.total_num_filters()
        total_filters_to_prune = int(original_filters * (pruning_percentage / 100.0))
        print(f"Total Filters to prune: {total_filters_to_prune} For Pruning Percentage: {pruning_percentage}")

        # Rank and get the candidates to prune
        prune_targets = self.get_candidates_to_prune(total_filters_to_prune)
        layers_pruned = {}
        for layer_index, filter_index in prune_targets:
            layers_pruned[layer_index] = layers_pruned.get(layer_index, 0) + 1
        print("Layers that will be pruned", layers_pruned)

        print("Pruning Filters")
        model = self.model
        pruner = Pruning(model)
        
        # Prune one filter at a time with memory cleanup after each
        for idx, (layer_index, filter_index) in enumerate(prune_targets):
            model = pruner.prune_vgg_conv_layer(model, layer_index, filter_index)
            if idx % 5 == 0:  # Clean up every few iterations
                self._clear_memory()

        # Convert model weights to float32 for training stability
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                if layer.weight is not None:
                    layer.weight.data = layer.weight.data.float()
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.float()

        self.model = model.to(self.device)
        self._clear_memory()

        # Test and fine tune model
        acc_pre_fine_tuning = self.test(model)
        if pruning_percentage != 0.0:
            print(f"Accuracy before fine tuning: {acc_pre_fine_tuning[0]:.2f}%")
            
            optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
            best_accuracy = 0.0
            
            # Fine-tuning phase
            epoch = 0
            prev_accs = []
            while True:
                print(f"Fine-tuning epoch {epoch+1}")
                self.train_epoch(optimizer, rank_filter=False)
                val_results = self.test(self.model)
                print(f"Validation Accuracy: {val_results[0]:.2f}%")
                prev_accs.append(val_results[0])
                if len(prev_accs) > 5:
                    prev_accs.pop(0)
                scheduler.step(val_results[0])
                if val_results[0] > best_accuracy:
                    best_accuracy = val_results[0]
                self._clear_memory()
                epoch += 1
                print(f"Mean: {self._get_mean(prev_accs)}, Best: {best_accuracy}")
                if epoch > 3 and best_accuracy - 2 <= self._get_mean(prev_accs) <= best_accuracy + 2:
                    print("No improvement in accuracy for 5 epochs, stopping fine-tuning")
                    print(f"Best accuracy: {best_accuracy:.2f}%")
                    print(f"Mean accuracy over last 5 epochs: {self._get_mean(prev_accs):.2f}%")

                
        # Final evaluation
        acc_time = self.test(self.model)
        print("Finished Pruning for", pruning_percentage)
        print(f"Accuracy after fine tuning: {acc_time[0]:.2f}%")
        print(f"Time taken for inference: {acc_time[1]:.2f} seconds")
        size_mb = self.get_model_size(self.model)
        print(f"Model Size after fine tuning: {size_mb:.2f} MB")
        
        return [acc_pre_fine_tuning, acc_time[0], acc_time[1], size_mb, acc_pre_fine_tuning[0]]
    
    def reset(self):
        """Clear memory resources completely"""
        if hasattr(self, 'pruner'):
            self.pruner.reset()
            del self.pruner

        # Clear model explicitly
        if hasattr(self, 'model'):
            del self.model

        # Clear any other stored objects
        for attr in list(self.__dict__.keys()):
            if attr not in ['device', 'train_path', 'test_path'] and hasattr(self, attr):
                delattr(self, attr)

        # Force garbage collection and clear GPU cache
        self._clear_memory()
        
    def save_model(self, path):
        torch.save(self.model, path)
        print(f"Model saved as {path}")
        
    def __del__(self):
        """Destructor to ensure memory is cleared when the object is deleted"""
        self.reset()
        print("PruningFineTuner object deleted and memory cleared.")