from math import ceil
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn.functional as F
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
        self.num_classes = num_classes
        
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

        if test_or_train == 'train':
            dataset = MultiviewImgDataset(root_dir=self.train_path,
                                          scale_aug=True,
                                          rot_aug=True,
                                          num_models=0,
                                          num_views=12)
            
            #use random sampling if num_samples is specified
            if num_samples > 0:
                dataset_indices = random.sample(range(len(dataset)), num_samples)
                dataset = Subset(dataset, dataset_indices)
            
            print(f"Dataset Size Training: {len(dataset)}")
        elif test_or_train == 'time':
            # num_samples=8
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
        return DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )    
    
    def rank_filter_train(self, train_loader, optimizer=None): # FOR RANKS
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
                target = data[0].to(self.device)
            else:
                in_data = data[1].to(self.device)
                target = data[0].to(self.device)
            if optimizer is not None:
                optimizer.zero_grad()
            else:
                self.model.zero_grad()

            out_data = self.pruner.forward(in_data, self.model_name)

            loss = self.criterion(out_data, target)
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/inf loss detected: {loss.item()}")
                continue  # Skip this batch

            running_loss += loss.item()
            pred = torch.max(out_data, 1)[1]
            results = pred == target
            correct_points = torch.sum(results.long())
            acc = correct_points.float() / results.size()[0]
            running_acc += acc.item()
            
            if optimizer is not None:
                loss.backward()
                # Add gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.model.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc.item():.4f}"})
    
    def train_model(self, train_loader, optimizer=None):
        self.model.train()
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
                target = data[0].to(self.device)
            else:
                in_data = data[1].to(self.device)
                target = data[0].to(self.device)
            if optimizer is not None:
                optimizer.zero_grad()
            else:
                self.model.zero_grad()

            
            Y = self.model.net_1(in_data)
            if self.model_name == 'mvcnn':
                Y = F.adaptive_avg_pool2d(Y, (7, 7))
                Y = Y.view(N, V, Y.shape[-3], Y.shape[-2], Y.shape[-1]) # type: ignore
                Y = torch.max(Y, 1)[0]
                Y = Y.view(N, -1) # type: ignore
                
            out_data = self.model.net_2(Y)
            loss = self.criterion(out_data, target)
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/inf loss detected: {loss.item()}, skipping batch")
                continue

            running_loss += loss.item()
            pred = out_data.argmax(dim=1)
            results = pred == target
            correct_points = torch.sum(results.long())
            acc = correct_points.float() / results.size()[0]
            running_acc += acc.item()
            total_steps += 1
            
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                optimizer.step()
            else:
                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.model.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc.item():.4f}"})

        train_loss = running_loss / total_steps
        train_acc = running_acc / total_steps
        self._log(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        self._clear_memory()
        return self.model
            
    def train_epoch(self, optimizer=None, rank_filter=False):
        """Train model for one epoch"""
        if rank_filter:
            train_loader = self.get_modelnet33_images('train', num_samples=300)
        else:
            train_loader = self.get_modelnet33_images('train', num_samples=1000)
        
        if train_loader is None:
            self._log("Error: train_loader is None")
            return self.model
        
        if rank_filter:
            self.rank_filter_train(train_loader, optimizer)
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
                N, V, C, H, W = views.size()
                x = views.view(-1, C, H, W)                # (N*V, C, H, W)
                tgt = labels
    
                net_1_out = self.model.net_1(x)  # (N*V, 33)
                y = F.adaptive_avg_pool2d(net_1_out, (7, 7))
                y = y.view(N, V, y.shape[-3], y.shape[-2], y.shape[-1])
                y = torch.max(y, dim=1)[0]
                y = y.view(N, -1)

                out = self.model.net_2(y)
                preds = out.argmax(dim=1)
    
                # compute batch loss
                all_loss += torch.nn.functional.cross_entropy(out, tgt).item()
                
                batch_correct = (preds == tgt).sum().item()
                all_correct += batch_correct
                all_samples += N
                acc = batch_correct / N
                
                for i in range(N):
                    if preds[i] != tgt[i]:
                        wrong_class[tgt[i].item()] += 1
                    samples_class[tgt[i].item()] += 1

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
        test_loader = self.get_modelnet33_images('time', num_samples=8)
        with torch.no_grad():
            for data in test_loader:
                label, image = data[0], data[1]
                
                # Handle multi-view data properly
                if self.model_name == 'mvcnn' and len(image.shape) == 5:
                    # image shape: [batch_size, num_views, channels, height, width]
                    N, V, C, H, W = image.size()
                    image = image.view(-1, C, H, W)  # Reshape to [batch_size * num_views, channels, height, width]
                
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
        self.pruner.reset()  # Start fresh
        self.train_epoch(rank_filter=True)  # Accumulate rankings
        
        # Check if we actually have rankings
        if not hasattr(self.pruner, 'filter_ranks') or not self.pruner.filter_ranks:
            self._log("Error: No filter rankings accumulated!")
            return []
        
        self._log(f"Accumulated rankings for {len(self.pruner.filter_ranks)} layer groups")
        for layer_idx, ranks in self.pruner.filter_ranks.items():
            self._log(f"Layer group {layer_idx}: {ranks.size(0)} filters ranked")
        
        self.pruner.normalize_ranks_per_layer()
        pruning_plan = self.pruner.get_pruning_plan(num_filter_to_prune)
        
        # Now we can safely reset since we have the pruning plan
        self.pruner.reset()
        
        return pruning_plan
    
    def total_num_filters(self):
        """Count total filters in model, excluding the last Conv2d layer"""
        # gather all conv layers' filter counts
        conv_counts = [
            layer.out_channels
            for layer in self.model.net_1
            if isinstance(layer, torch.nn.modules.conv.Conv2d)
        ]
        # drop last conv
        if conv_counts:
            conv_counts = conv_counts[:-1]
        return sum(conv_counts)
    
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
            print(f"num_filters_to_prune not provided, calculating based on pruning amount {pruning_amount:.3f}") if num_filters_to_prune is None else None
            num_filters_to_prune = ceil(num_filters_to_prune*pruning_amount) if num_filters_to_prune is not None else ceil(pruning_amount * self.total_num_filters()) 
            self._log(f"Pruning {num_filters_to_prune} filters at pruning amount {pruning_amount*100:.3f}%")
            filters_to_prune = self.get_candidates_to_prune(num_filters_to_prune)
        else:
            filters_to_prune = prune_targets

        # Handle case where no filters can be pruned
        if filters_to_prune is None or len(filters_to_prune) == 0:
            model_size = self.get_model_size(self.model)
            comp_time = self.get_comp_time(self.model)
            
            self._log(f"No more filters can be pruned at pruning amount {pruning_amount:.3f}")
            self._log(f"Current metrics - Accuracy: 0.0%, Model size: {model_size:.2f}M, Comp time: {comp_time:.2f}ms")
            
            return False

        if not only_model:
            no_filters = self.total_num_filters()
            self._log(f"Pruning {len(filters_to_prune)} filters out of {no_filters} ({100 * len(filters_to_prune) / no_filters:.1f}%)")
            
        # Use batch pruning for better performance
        pruner = Pruning(self.model)
        self.model = pruner.batch_prune_filters(self.model, filters_to_prune)
        
        # Validate model architecture after pruning
        try:
            # Test with a small batch to ensure model still works
            test_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                test_output = self.model(test_input)
                if test_output.size(1) != 33:
                    print(f"Error: Model output size is {test_output.size(1)}, expected 33")
                    return False
                print(f"Model validation passed: output shape {test_output.shape}")
        except Exception as e:
            print(f"Model validation failed after pruning: {e}")
            return False
        
        # Create new pruner for the updated model
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