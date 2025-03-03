import torch
import numpy as np
from EnhancedFilterPruner import EnhancedFilterPruner
from GradientFlowAnalyzer import GradientFlowAnalyzer
from Pruning import Pruning
import time
from torch import optim
import gc

class EnhancedPruning(Pruning):
    """
    Enhanced pruning implementation that builds upon the base Pruning class
    with higher-order Taylor approximation and gradient flow analysis.
    """
    
    def __init__(self, model):
        super().__init__(model)
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gradient_analyzer = GradientFlowAnalyzer(model)
        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        print(f"Using device: {self.device}")
    
    def prune_vgg_conv_layer(self, model, layer_index, filter_index, preserve_output_size=False):
        """
        Enhanced pruning with additional options to preserve output dimensions.
        
        Args:
            model: The model to prune
            layer_index: Index of the layer to prune
            filter_index: Index of the filter to prune
            preserve_output_size: Whether to preserve the output dimensions by adjusting subsequent layers
            
        Returns:
            The pruned model
        """
        # Call the original pruning method
        model = super().prune_vgg_conv_layer(model, layer_index, filter_index)
        
        # If preserving output size is enabled, make additional adjustments
        if preserve_output_size:
            model = self._adjust_for_preserved_output(model, layer_index)
            
        return model
    
    def _adjust_for_preserved_output(self, model, layer_index):
        """
        Make additional adjustments to preserve output dimensions after pruning.
        This is an experimental feature for more stable pruning.
        """
        try:
            # This is just a placeholder for potential adjustments
            # In a real implementation, this could involve various strategies
            # such as adding lightweight skip connections or adjustment layers
            return model
        except Exception as e:
            print(f"Error in output preservation adjustment: {e}")
            return model
    
    def prune_with_taylor_expansion(self, model, pruning_ratio, taylor_order=2, use_gradient_flow=True):
        """
        Prune the model using higher-order Taylor expansion and gradient flow analysis.
        
        Args:
            model: The model to prune
            pruning_ratio: Percentage of filters to prune (0-100)
            taylor_order: Order of Taylor expansion (1, 2, or 3)
            use_gradient_flow: Whether to use gradient flow analysis
            
        Returns:
            The pruned model and a dictionary of pruning statistics
        """
        # Calculate number of filters to prune
        total_filters = sum(
            layer.out_channels for layer in model.features 
            if isinstance(layer, torch.nn.modules.conv.Conv2d)
        )
        num_filters_to_prune = int(total_filters * (pruning_ratio / 100.0))
        
        # Initialize the enhanced filter pruner
        pruner = EnhancedFilterPruner(model, taylor_order=taylor_order, use_gradient_flow=use_gradient_flow)
        
        # Reset model and compute filter importances
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        
        # Build a dataset loader (assuming this is provided elsewhere)
        # This is a placeholder - in actual usage you would pass your dataloader here
        data_loader = self._get_data_loader()
        
        # Propagate a few batches to compute filter importances
        batch_count = 0
        try:
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Reset pruner for each batch to avoid gradient accumulation
                pruner.reset()
                
                # Forward pass with importance calculation
                outputs = pruner.forward(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                batch_count += 1
                if batch_count >= 10:  # Limit to 10 batches for efficiency
                    break
                
                # Clear memory
                del inputs, targets, outputs, loss
                gc.collect()
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device == 'mps':
                    torch.mps.empty_cache()
        except Exception as e:
            print(f"Error during importance computation: {e}")
        
        # Normalize the computed ranks
        pruner.normalize_ranks_per_layer()
        
        # Get the pruning plan
        filters_to_prune = pruner.get_pruning_plan(num_filters_to_prune)
        
        # Execute pruning
        pruning_stats = {
            'total_filters': total_filters,
            'pruned_filters': len(filters_to_prune),
            'pruned_percentage': (len(filters_to_prune) / total_filters) * 100,
            'filters_per_layer': {},
            'taylor_order': taylor_order,
            'used_gradient_flow': use_gradient_flow
        }
        
        # Group filters by layer for statistics
        for layer_index, filter_index in filters_to_prune:
            if layer_index not in pruning_stats['filters_per_layer']:
                pruning_stats['filters_per_layer'][layer_index] = 0
            pruning_stats['filters_per_layer'][layer_index] += 1
        
        print(f"Pruning {len(filters_to_prune)} filters ({pruning_stats['pruned_percentage']:.2f}%)")
        print(f"Filters per layer: {pruning_stats['filters_per_layer']}")
        
        # Actually prune the model
        for layer_index, filter_index in filters_to_prune:
            model = self.prune_vgg_conv_layer(model, layer_index, filter_index)
        
        # Ensure model weights are float32
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                if layer.weight is not None:
                    layer.weight.data = layer.weight.data.float()
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.float()
        
        return model, pruning_stats
    
    def _get_data_loader(self):
        """
        Get data loader for importance computation. 
        This is a placeholder that should be replaced with actual data loading.
        """
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader, Subset
        import random
        
        # Similar to PruningFineTuner.get_images
        train_path = 'imagenet-mini/train'  # Should be configurable
        num_samples = 1000  # Smaller sample for filter importance calculation
        
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        try:
            data_dataset = datasets.ImageFolder(train_path, transform=transform)
            indices = random.sample(range(len(data_dataset)), min(num_samples, len(data_dataset)))
            subset_dataset = Subset(data_dataset, indices)
            return DataLoader(subset_dataset, batch_size=32, shuffle=True, num_workers=1)
        except Exception as e:
            print(f"Error creating data loader: {e}")
            # Return an empty data loader as fallback
            return []
    
    def fine_tune(self, model, epochs=5, learning_rate=0.001, momentum=0.9):
        """
        Fine-tune the pruned model to recover accuracy.
        
        Args:
            model: The pruned model to fine-tune
            epochs: Number of fine-tuning epochs
            learning_rate: Learning rate for optimizer
            momentum: Momentum for optimizer
            
        Returns:
            The fine-tuned model and a dictionary of fine-tuning statistics
        """
        model = model.to(self.device)
        model.train()
        
        # Set up optimizer and scheduler
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
        
        # Get validation accuracy before fine-tuning
        pre_tuning_accuracy = self.test_model(model)
        print(f"Accuracy before fine-tuning - Top-1: {pre_tuning_accuracy[1]:.2f}%, Top-5: {pre_tuning_accuracy[2]:.2f}%")
        
        # Statistics to track
        stats = {
            'pre_tuning_accuracy': pre_tuning_accuracy,
            'epoch_accuracies': [],
            'final_accuracy': None,
            'compute_time': None,
        }
        
        # Fine-tuning loop
        best_accuracy = 0.0
        data_loader = self._get_data_loader()
        
        for epoch in range(epochs):
            # Train for one epoch
            self._train_epoch(model, optimizer, data_loader)
            
            # Test the model
            epoch_accuracy = self.test_model(model)
            stats['epoch_accuracies'].append(epoch_accuracy)
            
            print(f"Epoch {epoch+1}/{epochs} - Top-1: {epoch_accuracy[1]:.2f}%, Top-5: {epoch_accuracy[2]:.2f}%")
            
            # Update scheduler
            scheduler.step(epoch_accuracy[0])  # Use raw accuracy for scheduler
            
            # Save best model
            if epoch_accuracy[0] > best_accuracy:
                best_accuracy = epoch_accuracy[0]
        
        # Final evaluation
        stats['final_accuracy'] = self.test_model(model)
        stats['compute_time'] = stats['final_accuracy'][3]  # Extract compute time
        
        print(f"Fine-tuning complete. Final accuracy - Top-1: {stats['final_accuracy'][1]:.2f}%, Top-5: {stats['final_accuracy'][2]:.2f}%")
        
        return model, stats
    
    def _train_epoch(self, model, optimizer, data_loader):
        """Train the model for one epoch."""
        model.train()
        
        for images, labels in data_loader:
            try:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                model.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
            except Exception as e:
                print(f"Error during training: {e}")
                continue
                
            finally:
                # Clean up
                del images, labels
                if 'outputs' in locals(): del outputs
                if 'loss' in locals(): del loss
                gc.collect()
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device == 'mps':
                    torch.mps.empty_cache()
    
    def test_model(self, model):
        """
        Test the model and return accuracy metrics.
        Similar to PruningFineTuner.test but integrated here.
        
        Returns:
            [accuracy, top1_percent, top5_percent, compute_time]
        """
        model.eval()
        model = model.to(self.device)
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        compute_time = 0
        
        # Get test data loader
        test_loader = self._get_test_loader()
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Convert to float32
                images = images.float()
                
                # Measure inference time
                t1 = time.time()
                outputs = model(images)
                t2 = time.time()
                compute_time += t2 - t1
                
                # Top-1 accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct_top1 += (predicted == labels).sum().item()
                
                # Top-5 accuracy
                _, top5_indices = torch.topk(outputs, 5, dim=1)
                for i in range(labels.size(0)):
                    if labels[i] in top5_indices[i]:
                        correct_top5 += 1
        
        accuracy = float(correct_top1/total)  # Raw accuracy
        top1_percent = accuracy * 100  # Top-1 accuracy percentage
        top5_percent = (float(correct_top5/total)) * 100  # Top-5 accuracy percentage
        
        return [accuracy, top1_percent, top5_percent, compute_time]
    
    def _get_test_loader(self):
        """Get data loader for testing."""
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader, Subset
        import random
        
        # Similar to PruningFineTuner.get_images for test data
        test_path = 'imagenet-mini/val'  # Should be configurable
        num_samples = 1000
        
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        try:
            data_dataset = datasets.ImageFolder(test_path, transform=transform)
            indices = random.sample(range(len(data_dataset)), min(num_samples, len(data_dataset)))
            subset_dataset = Subset(data_dataset, indices)
            return DataLoader(subset_dataset, batch_size=32, shuffle=False, num_workers=1)
        except Exception as e:
            print(f"Error creating test data loader: {e}")
            return []
    
    def get_model_size(self, model):
        """Calculate the model size in MB."""
        total_size = sum(
            param.nelement() * param.element_size() for param in model.parameters()
        )
        # Convert to MB
        return total_size / (1024 ** 2)
    
    def run_full_pruning(self, model, pruning_ratio, taylor_order=2, use_gradient_flow=True, fine_tune_epochs=5):
        """
        Run the complete pruning and fine-tuning pipeline.
        
        Args:
            model: The model to prune
            pruning_ratio: Percentage of filters to prune (0-100)
            taylor_order: Order of Taylor expansion (1, 2, or 3)
            use_gradient_flow: Whether to use gradient flow analysis
            fine_tune_epochs: Number of fine-tuning epochs
            
        Returns:
            The pruned and fine-tuned model, and comprehensive statistics
        """
        print(f"Starting pruning with {pruning_ratio}% target, Taylor order {taylor_order}")
        
        # Initial model size and accuracy
        initial_size = self.get_model_size(model)
        initial_accuracy = self.test_model(model)
        
        print(f"Initial model size: {initial_size:.2f} MB")
        print(f"Initial accuracy - Top-1: {initial_accuracy[1]:.2f}%, Top-5: {initial_accuracy[2]:.2f}%")
        
        # Step 1: Prune the model
        pruned_model, pruning_stats = self.prune_with_taylor_expansion(
            model, 
            pruning_ratio, 
            taylor_order=taylor_order, 
            use_gradient_flow=use_gradient_flow
        )
        
        # Size after pruning
        post_pruning_size = self.get_model_size(pruned_model)
        
        # Step 2: Test accuracy after pruning
        post_pruning_accuracy = self.test_model(pruned_model)
        print(f"Accuracy after pruning - Top-1: {post_pruning_accuracy[1]:.2f}%, Top-5: {post_pruning_accuracy[2]:.2f}%")
        
        # Step 3: Fine-tune the pruned model
        fine_tuned_model, fine_tuning_stats = self.fine_tune(pruned_model, epochs=fine_tune_epochs)
        
        # Final model size
        final_size = self.get_model_size(fine_tuned_model)
        
        # Compile comprehensive statistics
        stats = {
            'initial': {
                'size_mb': initial_size,
                'accuracy': initial_accuracy,
            },
            'post_pruning': {
                'size_mb': post_pruning_size,
                'accuracy': post_pruning_accuracy,
                'size_reduction_percent': (1 - post_pruning_size / initial_size) * 100,
                'accuracy_change_percent': post_pruning_accuracy[1] - initial_accuracy[1],
            },
            'post_fine_tuning': {
                'size_mb': final_size,
                'accuracy': fine_tuning_stats['final_accuracy'],
                'size_reduction_percent': (1 - final_size / initial_size) * 100,
                'accuracy_change_percent': fine_tuning_stats['final_accuracy'][1] - initial_accuracy[1],
            },
            'pruning_details': pruning_stats,
            'fine_tuning_details': fine_tuning_stats,
        }
        
        # Print summary
        print("="*50)
        print("PRUNING SUMMARY")
        print("="*50)
        print(f"Model size reduction: {stats['post_fine_tuning']['size_reduction_percent']:.2f}%")
        print(f"From {initial_size:.2f} MB to {final_size:.2f} MB")
        print(f"Final Top-1 accuracy: {stats['post_fine_tuning']['accuracy'][1]:.2f}%")
        print(f"Final Top-5 accuracy: {stats['post_fine_tuning']['accuracy'][2]:.2f}%")
        print(f"Accuracy change: {stats['post_fine_tuning']['accuracy_change_percent']:.2f}%")
        print("="*50)
        
        return fine_tuned_model, stats
