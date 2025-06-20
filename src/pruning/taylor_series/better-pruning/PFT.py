from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import torch
import random
import gc
import time
from FilterPruner import FilterPruner
from Pruning import Pruning
# from tools.ImgDataset import SingleImgDataset
from tqdm import tqdm

class PruningFineTuner:
    def __init__(self, model, quiet=False):
        # Dataset paths
        self.train_path = 'imagenet-mini/train'
        self.test_path = 'imagenet-mini/val'
        
        # Device selection
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.quiet = quiet
        
        # Model setup
        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.pruner = FilterPruner(self.model)
        
        # Clean initial state
        self._clear_memory()
        
    def _clear_memory(self):
        """Release unused memory from GPU/MPS"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
    def _log(self, message):
        """Conditionally print messages based on quiet mode"""
        if not self.quiet:
            print(message)
        
    # New method for Imagenet-mini with standard preprocessing
    def get_imagenet_mini_images(self, test_or_train, num_samples=2000):
        """Create data loader for Imagenet-mini dataset"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip() if test_or_train == 'train' else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        dataset = datasets.ImageFolder(
            root=self.train_path if test_or_train == 'train' else self.test_path, 
            transform=transform
        )
        
        # Create subset for efficiency
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        dataset = Subset(dataset, indices)
        
        if not self.quiet:
            self._log(f"Imagenet-mini {test_or_train}: {len(dataset)} samples")
            
        return DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
        )
    
    def train_batch(self, optimizer, train_loader, rank_filter=False):
        """Train for a single batch"""
        self.model.train()
        self.model.to(self.device)
        
        # Wrap with tqdm for progress visualization
        pbar = tqdm(
            train_loader, 
            desc="Train", 
            leave=False,
            disable=self.quiet,
            ncols=80
        )
        
        for batch_idx, (image, label) in enumerate(pbar):
            try:
                with torch.autograd.set_grad_enabled(True):
                    # Move data to device
                    image = image.to(self.device, non_blocking=False)
                    label = label.to(self.device, non_blocking=False)
                    
                    # Zero gradients
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
                    
                    # Loss and backprop
                    loss = self.criterion(output, label)
                    loss.backward()
                    
                    # Update weights if optimizer provided
                    if optimizer is not None:
                        optimizer.step()
                        
                    # Update progress bar
                    if not self.quiet:
                        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                        
            except Exception as e:
                self._log(f"Error in batch {batch_idx}: {str(e)}")
                continue
            finally:
                # Free memory
                del image, label
                if batch_idx % 10 == 0:
                    self._clear_memory()
    
    def train_epoch(self, optimizer=None, rank_filter=False):
        """Train model for one epoch"""
        train_loader = self.get_imagenet_mini_images('train', num_samples=2000)
        self.train_batch(optimizer, train_loader, rank_filter)
        del train_loader
        self._clear_memory()
        return self.model
    
    def get_val_accuracy(self, model):
        """Calculate validation accuracy"""
        test_loader = self.get_imagenet_mini_images('val', num_samples=1000)
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(
                test_loader, 
                desc="Validating", 
                leave=False,
                disable=self.quiet,
                ncols=80
            )
            for image, label in pbar:
                # Inference
                image = image.to(self.device, non_blocking=False)
                label = label.to(self.device, non_blocking=False)
                output = model(image)

                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                # Update progress
                accuracy = 100 * correct / total
                if not self.quiet:
                    pbar.set_postfix({"acc": f"{accuracy:.2f}%"})

        return self._clean_up(test_loader, accuracy) # type: ignore
    
    def get_comp_time(self, model):
        """Measure computation time"""
        start_time = time.time()
        model.eval()
        model.to('cpu')

        # Run inference on CPU for fair comparison
        test_loader = self.get_imagenet_mini_images('val', num_samples=100)
        with torch.no_grad():
            for image, label in test_loader:
                image = image.to('cpu', non_blocking=False)
                _ = model(image)
                del image, label

        elapsed = time.time() - start_time
        model = model.to(self.device)  # Move back to original device

        return self._clean_up(test_loader, elapsed)

    def _clean_up(self, test_loader, arg1):
        del test_loader
        self._clear_memory()
        return arg1
    
    def get_candidates_to_prune(self, num_filter_to_prune):
        """Identify filters to prune based on importance ranking"""
        self._log("Ranking filters...")
        self.pruner.reset()
        self.train_epoch(rank_filter=True)
        self.pruner.normalize_ranks_per_layer()
        return self.pruner.get_pruning_plan(num_filter_to_prune)
    
    def total_num_filters(self):
        """Count total filters in model"""
        return sum(
            layer.out_channels
            for layer in self.model.net_1
            if isinstance(layer, torch.nn.modules.conv.Conv2d)
        )
    
    def get_model_size(self, model):
        """Calculate model size in MB"""
        total_size = sum(
            param.nelement() * param.element_size() 
            for param in model.parameters()
        )
        return total_size / (1024 ** 2)  # Convert to MB
    
    def prune(self, pruning_amount, only_model=True, prune_targets=None):
        """Prune filters based on Taylor criterion"""
        self.model.train()
        
        # Enable gradients for pruning
        for param in self.model.net_1.parameters():
            param.requires_grad = True
            
        # Calculate pruning targets
        original_filters = self.total_num_filters()
        total_filters_to_prune = int(original_filters * pruning_amount)
        self._log(f"Pruning {total_filters_to_prune} filters ({pruning_amount:.2f})")

        # Identify filters to remove
        if prune_targets is None:
            prune_targets = self.get_candidates_to_prune(total_filters_to_prune)
        
        # Perform pruning
        model = self.model
        pruner = Pruning(model)
        
        # Use tqdm for progress tracking
        pbar = tqdm(
            enumerate(prune_targets), 
            total=len(prune_targets), 
            desc="Pruning", 
            leave=False,
            disable=self.quiet,
            ncols=80
        )
        
        for idx, (layer_index, filter_index) in pbar:
            model = pruner.prune_vgg_conv_layer(model, layer_index, filter_index)
            if idx % 100 == 0:
                self._clear_memory()
                
        # Ensure weights are float32 for training stability
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                if layer.weight is not None:
                    layer.weight.data = layer.weight.data.float()
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.float()

        # Update model and clean memory
        self.model = model.to(self.device)
        self._clear_memory()
        
        return self.model
    
    def reset(self):
        """Clean up resources"""
        if hasattr(self, 'pruner'):
            self.pruner.reset()
            del self.pruner

        # Clear model explicitly
        if hasattr(self, 'model'):
            del self.model

        # Force garbage collection and clear GPU cache
        self._clear_memory()
        
    def save_model(self, path):
        """Save model to disk"""
        torch.save(self.model.state_dict(), path)
        self._log(f"Model saved as {path}")
        
    def __del__(self):
        """Cleanup on deletion"""
        self.reset()