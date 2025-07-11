from math import ceil
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch
import random
import gc
import time
from FilterPruner import FilterPruner
from Pruning import Pruning
from tools.ImgDataset import MultiviewImgDataset
from tqdm import tqdm
import numpy as np

class PruningFineTuner:
    def __init__(self, model=None, quiet=False, view_importance=False, num_classes=33):
        # Dataset paths
        self.train_path = '../../../MVCNN/ModelNet40-12View/*/train'
        self.test_path = '../../../MVCNN/ModelNet40-12View/*/test'
        self.num_classes = num_classes  # Support for 33 classes
        
        print("Initializing PruningFineTuner...")

        # Device selection
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.quiet = quiet

        if view_importance:
            self.imp =  self.get_modelnet33_images('train', num_samples=10000)
        else:
            if model is None:
                raise ValueError("Model must be provided for fine-tuning.")
            
            self.model = model.to(self.device)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.pruner = FilterPruner(self.model)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0.0)

            self.val_dataset = self.get_modelnet33_images('test')
        
        self.num_views = 12
        self.model_name = 'mvcnn' if model is not None else 'unknown'

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
    
    def get_imp_set(self):
        return self.imp
    
    def get_modelnet33_images(self, test_or_train, num_samples=-1):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ]) if test_or_train == 'train' else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        if test_or_train == 'train':
            dataset = self.get_test_dataset(num_samples)
        elif test_or_train == 'time':
            num_samples=100
            full_dataset = MultiviewImgDataset(root_dir=self.train_path,
                                               scale_aug=False,
                                               rot_aug=False,
                                               num_models=0,
                                               num_views=12)
            dataset_indices = random.sample(range(len(full_dataset)), num_samples)
            dataset = Subset(full_dataset, dataset_indices)

        else:  # 'test or val or whatever else'
            dataset = MultiviewImgDataset(root_dir=self.test_path,
                                          scale_aug=False,
                                          rot_aug=False,
                                          test_mode=True,
                                          num_models=0,
                                          num_views=12)

        dataset.transform = transform  # type: ignore
    
        # Enable shuffling for Subset datasets at DataLoader level
        shuffle_data = hasattr(dataset, 'dataset')  # True if it's a Subset
    
        return DataLoader(
            dataset,
            batch_size=8,
            shuffle=shuffle_data,  # Enable shuffle for Subset datasets
            num_workers=4,
            pin_memory=True,
        )

    def get_test_dataset(self, num_samples):
        full_dataset = MultiviewImgDataset(root_dir=self.train_path,
                                           scale_aug=False,
                                           rot_aug=False,
                                           num_models=0,
                                           num_views=12)

        if num_samples < 0:
            num_samples = len(full_dataset) // 3
        self._log(f"Total samples in ModelNet33 full train dataset: {len(full_dataset)}")
        self._log(f"Sub-sampling ModelNet33 train dataset to {num_samples} samples")

        indices = torch.randperm(len(full_dataset)).tolist()
        split_point = int(len(indices) * 1)

        dataset_indices = indices[:split_point]

        # Sub-sample if needed for faster runs
        if num_samples < len(dataset_indices):
            dataset_indices = random.sample(dataset_indices, num_samples)

        return Subset(full_dataset, dataset_indices)
    
    
    def train_batch(self, train_loader, optimizer=None): # FOR RANKS
        # sourcery skip: class-extract-method
        self.model.train()

        # Skip filepath shuffling for Subset datasets to avoid index errors
        dataset = train_loader.dataset
        if not hasattr(dataset, 'dataset'):
            current_filepaths = dataset.filepaths
            rand_idx = np.random.permutation(len(current_filepaths) // 12)
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(current_filepaths[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            dataset.filepaths = filepaths_new

        # lr = self.optimizer.state_dict()['param_groups'][0]['lr'] # type: ignore
        out_data = None
        in_data = None

        if optimizer is None:
            optimizer = self.optimizer

        pbar = tqdm(train_loader, desc="Training")
        running_loss = 0.0
        running_acc = 0.0
        for data in pbar:
            if self.model_name == 'mvcnn':
                N, V, C, H, W = data[1].size()
                in_data = data[1].view(-1, C, H, W).to(self.device)
                target = data[0].to(self.device).repeat_interleave(V)
            else:
                in_data = data[1].to(self.device)
                target = data[0].to(self.device)
            if optimizer is not None:
                optimizer.zero_grad()
            else:
                self.model.zero_grad()

            self.pruner.reset()

            out_data = self.model.forward(in_data)

            loss = self.criterion(out_data, target)

            running_loss += loss.item()
            pred = torch.max(out_data, 1)[1]
            results = pred == target
            correct_points = torch.sum(results.long())
            acc = correct_points.float() / results.size()[0]
            running_acc += acc.item()
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            else:
                self.model.zero_grad()
                loss.backward()
                self.model.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc.item():.4f}"})
    
    def train_model(self, train_loader, optimizer=None):
        self.model.train()

        # Skip filepath shuffling for Subset datasets to avoid index errors
        dataset = train_loader.dataset
        if not hasattr(dataset, 'dataset'):
            current_filepaths = dataset.filepaths
            rand_idx = np.random.permutation(len(current_filepaths) // 12)
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(current_filepaths[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            dataset.filepaths = filepaths_new

        # lr = self.optimizer.state_dict()['param_groups'][0]['lr'] # type: ignore
        out_data = None
        in_data = None

        pbar = tqdm(train_loader, desc="Training")
        running_loss = 0.0
        running_acc = 0.0
        total_steps = 0
        for data in pbar:
            if self.model_name == 'mvcnn':
                N, V, C, H, W = data[1].size()
                in_data = data[1].view(-1, C, H, W).to(self.device)
                target = data[0].to(self.device).repeat_interleave(V)
            else:
                in_data = data[1].to(self.device)
                target = data[0].to(self.device)
            if optimizer is not None:
                optimizer.zero_grad()
            else:
                self.model.zero_grad()

            out_data = self.model(in_data)
            loss = self.criterion(out_data, target)

            running_loss += loss.item()
            pred = torch.max(out_data, 1)[1]
            results = pred == target
            correct_points = torch.sum(results.long())
            acc = correct_points.float() / results.size()[0]
            running_acc += acc.item()
            total_steps += 1
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            else:
                self.model.zero_grad()
                loss.backward()
                self.model.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc.item():.4f}"})

        train_loss = running_loss / total_steps
        train_acc = running_acc / total_steps
        self._log(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        self._clear_memory()
        return self.model
            
    def train_epoch(self, optimizer=None, rank_filter=False):
        """Train model for one epoch"""
        train_loader = self.get_modelnet33_images('train', num_samples=800 if rank_filter else -1)
        
        if train_loader is None:
            self._log("Error: train_loader is None")
            return self.model
        
        # Fix parameter order: train_batch expects (train_loader, optimizer)
        if rank_filter:
            self.train_batch(train_loader, optimizer)
        else:
            self.train_model(train_loader, optimizer)
        
        del train_loader
        self._clear_memory()
        return self.model
    
    def validate_model(self,
                   single_view: bool = False,
                   view_idx: int = 0):  # sourcery skip: low-code-quality
        all_correct = 0
        all_samples = 0
        all_loss = 0.0

        # for per‐class stats - using dynamic num_classes
        wrong_class   = np.zeros(self.num_classes, dtype=int)
        samples_class = np.zeros(self.num_classes, dtype=int)

        self.model.eval()
        pbar = tqdm(self.val_dataset, desc=f'Val {self.model_name}'+(' (1 view)' if single_view else ''), unit='batch')
        for batch_i, data in enumerate(pbar):
            labels, views = data[0].to(self.device), data[1].to(self.device)
            # views: (N, V, C, H, W)
            if self.model_name == 'mvcnn' and not single_view:
                # —— full MVCNN voting as before ——
                N, V, C, H, W = views.size()
                x = views.view(-1, C, H, W)                # (N*V, C, H, W)
                tgt = labels.repeat_interleave(V, dim=0)   # (N*V,)
                out = self.model(x)
                preds = out.argmax(1)

                # compute batch loss
                all_loss += torch.nn.functional.cross_entropy(out, tgt).item()
                
                # majority vote per object
                batch_correct = 0
                for i in range(N):
                    vp = preds[i*V:(i+1)*V].cpu()
                    voted = torch.mode(vp)[0]
                    if voted == labels[i].cpu():
                        batch_correct += 1
                    else:
                        wrong_class[labels[i].item()] += 1
                    samples_class[labels[i].item()] += 1

                all_correct += batch_correct
                all_samples += N
                acc = batch_correct / N

            else:
                # —— single‐view branch (or any non-mvcnn model) ——
                # pick one view:
                x     = views[:, view_idx, ...]    # (N, C, H, W)
                tgt   = labels                     # (N,)
                out   = self.model(x)
                preds = out.argmax(1)

                all_loss += torch.nn.functional.cross_entropy(out, tgt).item()
                
                # accumulate per‐class if you want
                matches = (preds == tgt)
                for i, correct in enumerate(matches):
                    samples_class[tgt[i].item()] += 1
                    if not correct:
                        wrong_class[tgt[i].item()] += 1

                batch_correct = matches.sum().item()
                all_correct += batch_correct
                all_samples += tgt.size(0)
                acc = batch_correct / tgt.size(0)

            pbar.set_postfix({
                'acc':  f'{acc:.4f}',
                'loss': f'{(all_loss / (batch_i+1)):.4f}'
            })

        overall_acc = all_correct / all_samples
        per_cls     = (samples_class - wrong_class) / np.maximum(samples_class, 1)
        mean_cls    = per_cls[samples_class>0].mean()

        print(f'\nOverall Acc: {overall_acc:.4f}   Mean Class Acc: {mean_cls:.4f}')
        print(f'Dataset contains {self.num_classes} classes')

        self._clear_memory()
        return mean_cls
    
    def get_comp_time(self, model):
        """Measure computation time"""
        print("Measuring computation time...")
        start_time = time.time()
        model.eval()
        model.to('cpu')

        # Run inference on CPU for fair comparison
        test_loader = self.get_modelnet33_images('time', num_samples=100)
        with torch.no_grad():
            for label, image, _ in test_loader:
                image = image.to('cpu', non_blocking=False)
                
                # Handle MVCNN input format correctly
                if self.model_name == 'mvcnn':
                    # image shape: (batch_size, num_views, channels, height, width)
                    N, V, C, H, W = image.size()
                    # Reshape to (batch_size * num_views, channels, height, width)
                    image = image.view(-1, C, H, W)
                
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
    
    def prune(self, pruning_amount, only_model=True, prune_targets=None, num_filters_to_prune=None):
        """Prune the model by removing filters with lowest Taylor scores"""
        # Initialize or use provided pruning targets
        if prune_targets is None:
            # Calculate number of filters to prune
            print(f"num_filters_to_prune not provided, calculating based on pruning amount {pruning_amount:.3f}") if num_filters_to_prune is None else None
            num_filters_to_prune = ceil(num_filters_to_prune*pruning_amount) if num_filters_to_prune is not None else ceil(pruning_amount * self.total_num_filters()) 
            self._log(f"Pruning {num_filters_to_prune} filters at pruning amount {pruning_amount*100:.3f}%")
            filters_to_prune = self.get_candidates_to_prune(num_filters_to_prune)
        else:
            filters_to_prune = prune_targets

        # Handle case where no filters can be pruned
        if filters_to_prune is None or len(filters_to_prune) == 0:
            self._log(f"No more filters can be pruned at pruning amount {pruning_amount:.3f}")
            
            # Get metrics for current model state
            accuracy = self.validate_model()
            model_size = self.get_model_size(self.model)
            comp_time = self.get_comp_time(self.model)
            
            self._log(f"Current metrics - Accuracy: {accuracy:.2f}%, Model size: {model_size:.2f}MB, Comp time: {comp_time:.2f}s")
            
            return False

    if not only_model:
        no_filters = self.total_num_filters()
        self._log(f"Pruning {len(filters_to_prune)} filters out of {no_filters} ({100 * len(filters_to_prune) / no_filters:.1f}%)") # type: ignore
        
    # Use batch pruning for better performance
    pruner = Pruning(self.model)
    self.model = pruner.batch_prune_filters(self.model, filters_to_prune)
    self.pruner = FilterPruner(self.model)

    self._clear_memory()
    return True
    
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