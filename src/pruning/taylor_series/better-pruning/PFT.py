from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import torch
import random
import gc
from FilterPruner import FilterPruner
from Pruning import Pruning
from tools.ImgDataset import SingleImgDataset

class PruningFineTuner:
    def __init__(self, model):
        # self.train_path = 'ModelNet40-12View/*/train'
        # self.test_path = 'ModelNet40-12View/*/test'
        # self.train_path = 'places365/train'
        # self.test_path = 'places365/val'
        # New paths for Imagenet-mini
        self.train_path = 'imagenet-mini/train'
        self.test_path = 'imagenet-mini/val'
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.pruner = FilterPruner(self.model)
        self._clear_memory()
        
    def _clear_memory(self):
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
        data_dataset = SingleImgDataset(
            folder_path,)
        indices = random.sample(range(len(data_dataset)), min(num_samples, len(data_dataset)))
        data_dataset = Subset(data_dataset, indices)
        print(f"Number of samples in dataset: {len(data_dataset)}")
        
        data_dataset.transform = transform  # type: ignore
        return DataLoader(data_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    
    def get_places365_images(self, test_or_train, num_samples=5000):
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])
        
        dataset = datasets.ImageFolder(root=self.train_path if test_or_train == 'train' else self.test_path, transform=transform)
        
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        dataset = Subset(dataset, indices)
        print(f"Number of samples in {test_or_train} dataset: {len(dataset)}")
        
        return DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    
    # New method for Imagenet-mini
    def get_imagenet_mini_images(self, test_or_train, num_samples=5000):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip() if test_or_train == 'train' else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])
        
        dataset = datasets.ImageFolder(root=self.train_path if test_or_train == 'train' else self.test_path, transform=transform)
        
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        dataset = Subset(dataset, indices)
        print(f"Number of samples in {test_or_train} Imagenet-mini dataset: {len(dataset)}")
        
        return DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    
    def train_batch(self, optimizer, train_loader, rank_filter=False):
        self.model.train()
        self.model.to(self.device)
        for batch_idx, (image, label) in enumerate(train_loader):  #make it label, image, _ if using SingleImgDataset
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
                self._clear_memory()
            
            # Periodically clear cache during training
            if batch_idx % 10 == 0:
                self._clear_memory()
    
    def train_epoch(self, optimizer=None, rank_filter=False):
        # train_loader = self.get_images(self.train_path, num_samples=2000)
        # train_loader = self.get_places365_images('train', num_samples=2000)
        train_loader = self.get_imagenet_mini_images('train', num_samples=2000)
        self.train_batch(optimizer, train_loader, rank_filter)
        del train_loader
        self._clear_memory()
        return self.model
    
    def get_val_accuracy(self, model):
        # test_loader = self.get_images(self.test_path, num_samples=1000)
        # test_loader = self.get_places365_images('val', num_samples=1000)
        test_loader = self.get_imagenet_mini_images('val', num_samples=1000)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for image, label in test_loader: #make it label, image, _ if using SingleImgDataset
                image = image.to(self.device, non_blocking=False)
                label = label.to(self.device, non_blocking=False)
                output = model(image)
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test set: {accuracy:.2f}%')
        return accuracy
    
    def get_comp_time(self, model):
        import time
        start_time = time.time()
        model.eval()
        model.to('cpu')
        # test_loader = self.get_images(self.test_path, num_samples=100)
        # test_loader = self.get_places365_images('val', num_samples=100)
        test_loader = self.get_imagenet_mini_images('val', num_samples=100)
        with torch.no_grad():
            for image, label in test_loader: #make it label, image, _ if using SingleImgDataset
                image = image.to('cpu', non_blocking=False)
                _ = model(image)
                del image, label
                self._clear_memory()
        end_time = time.time()
        model = model.to(self.device)  # Move back to original device
        return end_time - start_time
    
    def get_candidates_to_prune(self, num_filter_to_prune):
        self.pruner.reset()
        self.train_epoch(rank_filter=True)
        self.pruner.normalize_ranks_per_layer()
        return self.pruner.get_pruning_plan(num_filter_to_prune)
    
    def total_num_filters(self):
        return sum(
            layer.out_channels
            for layer in self.model.net_1
            if isinstance(layer, torch.nn.modules.conv.Conv2d)
        )
    
    def get_model_size(self, model):
        total_size = sum(
            param.nelement() * param.element_size() for param in model.parameters()
        )
        # Convert to MB
        return total_size / (1024 ** 2)
    
    def get_ranks(self):
        original_filters = self.total_num_filters()
        return self.get_candidates_to_prune(original_filters)
    
    def prune(self, pruning_amount, only_model=True, prune_targets=None):
        self.model.train()
        
        # Enable gradients for pruning
        for param in self.model.net_1.parameters():
            param.requires_grad = True
            
        original_filters = self.total_num_filters()
        total_filters_to_prune = int(original_filters * (pruning_amount))
        print(f"Total Filters to prune: {total_filters_to_prune} For Pruning Amount: {pruning_amount}")

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
                
        print("Finished Pruning Filters")
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
        torch.save(self.model.state_dict(), path)
        print(f"Model saved as {path}")
        
    def __del__(self):
        self.reset()
        print("PruningFineTuner object deleted and memory cleared.")