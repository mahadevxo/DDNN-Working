from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch
import random
import gc
from FilterPruner import FilterPruner
from Pruning import Pruning
from MVCNN.tools.ImgDataset import SingleImgDataset

class PruningFineTuner:
    def __init__(self, model):
        self.train_path = 'ModelNet40-12View/*/train'
        self.test_path = 'ModelNet40-12View/*/test'
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.pruner = FilterPruner(self.model)
        self._clear_memory()
        
    def _clear_memory(self):
        """
        Clears GPU/MPS memory by forcing garbage collection and emptying caches.
        
        Helps prevent memory leaks during pruning and fine-tuning by explicitly
        freeing unused memory after operations that might create large temporary tensors.
        
        Args:
            None
            
        Returns:
            None
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
    def get_images(self, folder_path, num_samples=5000):
        """
        Creates a DataLoader for image data with specified transformations.
        
        Loads images from the provided folder path, applies transformations,
        and creates a DataLoader with a random subset of the data.
        
        Args:
            folder_path: Path to the image dataset
            num_samples: Maximum number of samples to include
            
        Returns:
            DataLoader object for the image dataset
        """
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])
        data_dataset = SingleImgDataset(
            folder_path,)
        indices = random.sample(range(len(data_dataset)), min(num_samples, len(data_dataset)))
        data_dataset = Subset(data_dataset, indices)
        # print(f"Number of samples in dataset: {len(data_dataset)}")
        
        data_dataset.transform = transform        
        return DataLoader(data_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    
    def train_batch(self, optimizer, train_loader, rank_filter=False):
        """
        Trains the model on a single batch of data.
        
        Processes a batch of images through the model, computes loss,
        and performs backpropagation. If rank_filter is True, uses the
        pruner's forward method to compute filter importance scores.
        
        Args:
            optimizer: Optimizer for weight updates (None if just computing rankings)
            train_loader: DataLoader providing training data
            rank_filter: If True, computes filter rankings instead of regular training
            
        Returns:
            None, but updates model weights and/or filter rankings
        """
        self.model.train()
        self.model.to(self.device)
        for batch_idx, (label, image, _) in enumerate(train_loader):
            try:
                with torch.autograd.set_grad_enabled(True):
                    image = image.to(self.device, non_blocking=False)
                    label = label.to(self.device, non_blocking=False)
                    
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
        """
        Trains the model for one epoch on a subset of training data.
        
        Creates a DataLoader for a small subset of training data and
        runs one training pass through it, either for standard training
        or for computing filter rankings.
        
        Args:
            optimizer: Optimizer for weight updates (None if just computing rankings)
            rank_filter: If True, computes filter rankings instead of regular training
            
        Returns:
            None, but updates model weights and/or filter rankings
        """
        train_loader = self.get_images(self.train_path, num_samples=200)
        self.train_batch(optimizer, train_loader, rank_filter)
        del train_loader
        self._clear_memory()
    
    def get_candidates_to_prune(self, num_filter_to_prune):
        """
        Identifies the least important filters in the model for pruning.
        
        Computes filter importance scores using the pruner's ranking method
        and returns a list of filters to prune, sorted by importance.
        
        Args:
            num_filter_to_prune: Number of filters to select for pruning
            
        Returns:
            List of (layer_index, filter_index) tuples identifying filters to prune
        """
        self.pruner.reset()
        self.train_epoch(rank_filter=True)
        self.pruner.normalize_ranks_per_layer()
        return self.pruner.get_pruning_plan(num_filter_to_prune)
    
    def total_num_filters(self):
        """
        Counts the total number of filters (output channels) in all convolutional layers.
        
        Sums the out_channels attribute of all Conv2d layers in the model
        to determine the total number of filters that could potentially be pruned.
        
        Args:
            None
            
        Returns:
            Integer representing total number of filters
        """
        return sum(
            layer.out_channels
            for layer in self.model.net_1
            if isinstance(layer, torch.nn.modules.conv.Conv2d)
        )
    
    def get_model_size(self, model):
        """
        Calculates the size of the model in megabytes.
        
        Sums the memory usage of all parameters in the model to determine
        the total model size in MB.
        
        Args:
            model: The model whose size to calculate
            
        Returns:
            Float representing the model size in MB
        """
        total_size = sum(
            param.nelement() * param.element_size() for param in model.parameters()
        )
        # Convert to MB
        return total_size / (1024 ** 2)
    
    def get_ranks(self):
        """
        Computes importance ranks for all filters in the model.
        
        Wrapper function that gets pruning candidates for all filters,
        effectively computing ranks for every filter in the model.
        
        Args:
            None
            
        Returns:
            List of (layer_index, filter_index) tuples with rank information
        """
        original_filters = self.total_num_filters()
        return self.get_candidates_to_prune(original_filters)
    
    def prune(self, pruning_percentage, only_model=True, prune_targets=None):
        """
        Prunes the model by removing a specified percentage of filters.

        This method ranks filters by importance and removes the least important ones, updating the model in place. 
        Optionally, a custom list of filters to prune can be provided.

        Args:
            pruning_percentage: Percentage of filters to prune from the model.
            only_model: If True, returns only the pruned model.
            prune_targets: Optional list of (layer_index, filter_index) tuples to specify which filters to prune.

        Returns:
            The pruned model if only_model is True; otherwise, None.
        """

        self.model.train()
        
        # Enable gradients for pruning
        for param in self.model.net_1.parameters():
            param.requires_grad = True
            
        original_filters = self.total_num_filters()
        total_filters_to_prune = int(original_filters * (pruning_percentage / 100.0))
        print(f"Total Filters to prune: {total_filters_to_prune} For Pruning Percentage: {pruning_percentage}")

        # Rank and get the candidates to prune
        
        if prune_targets is None:
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
            if idx % 100 == 0:  # Clean up every few iterations
                print(f"Pruned {idx} filters")
                self._clear_memory()
        # print("Finished Pruning Filters")
        print(f"Total Filters Pruned: {len(prune_targets)}")

        # Convert model weights to float32 for training stability
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                if layer.weight is not None:
                    layer.weight.data = layer.weight.data.float()
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.float()

        self.model = model.to(self.device)
        self._clear_memory()
        
        return self.model
    
    def reset(self):
        """
        Releases memory resources used by the pruner.
        
        Explicitly deletes objects and clears caches to free memory,
        particularly important when processing multiple pruning iterations.
        
        Args:
            None
            
        Returns:
            None
        """
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
        """
        Saves the pruned model's state dictionary to a file.
        
        Stores the model weights at the specified path for later loading.
        
        Args:
            path: File path to save the model
            
        Returns:
            None
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved as {path}")
        
    def __del__(self):
        """
        Destructor that ensures memory is cleared when the object is deleted.
        
        Calls the reset method to free resources when the PruningFineTuner
        object is garbage collected.
        
        Args:
            None
            
        Returns:
            None
        """
        self.reset()
        print("PruningFineTuner object deleted and memory cleared.")