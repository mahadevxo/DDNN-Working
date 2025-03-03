import torch
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import gc

class EnhancedFilterPruner:
    """
    Enhanced filter pruner with higher-order Taylor expansion and safeguards
    to prevent invalid pruning configurations.
    """
    
    def __init__(self, model, taylor_order=2, use_gradient_flow=False):
        """
        Initialize the filter pruner.
        
        Args:
            model: The neural network model to prune
            taylor_order: Order of Taylor expansion (1, 2, or 3)
            use_gradient_flow: Whether to use gradient flow analysis
        """
        self.model = model
        self.taylor_order = taylor_order
        self.use_gradient_flow = use_gradient_flow
        self.device = next(model.parameters()).device
        
        # Register for storing activation and gradient info
        self.activation_stats = {}
        self.gradient_stats = {}
        self.filter_ranks = {}
        self.hooks = []
        
        print(f"Initialized filter pruner with Taylor order {taylor_order} on device {self.device}")
        
        # Register forward and backward hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to collect activations and gradients."""
        self.hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Register forward hook for activations
                self.hooks.append(
                    module.register_forward_hook(
                        lambda m, inp, out, name=name: self._record_activation(m, inp, out, name)
                    )
                )
                
                # Register backward hook for gradients
                if module.weight.requires_grad:
                    self.hooks.append(
                        module.register_full_backward_hook(
                            lambda m, grad_in, grad_out, name=name: self._record_gradient(m, grad_in, grad_out, name)
                        )
                    )
    
    def reset(self):
        """Reset all statistics and remove hooks."""
        self.activation_stats = {}
        self.gradient_stats = {}
        self.filter_ranks = {}
        
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _record_activation(self, module, inp, out, name):
        """Record activations during forward pass."""
        try:
            # Store activations for each filter
            if name not in self.activation_stats:
                self.activation_stats[name] = []
            
            # Record memory usage
            if torch.cuda.is_available():
                memory = torch.cuda.memory_allocated() / (1024 * 1024)
                print(f"Peak memory usage during forward pass: {memory:.2f} MB")
            
            # Record only the necessary statistics to save memory
            activations = out.detach()
            
            # For higher-order Taylor expansions, we need to keep activations
            if self.taylor_order >= 2:
                self.activation_stats[name].append(activations)
            else:
                # For first-order, just store norms to save memory
                norms = torch.norm(activations, dim=(0, 2, 3))
                self.activation_stats[name].append(norms)
                
        except Exception as e:
            print(f"Error recording activation for {name}: {e}")
    
    def _record_gradient(self, module, grad_input, grad_output, name):
        """Record gradients during backward pass."""
        try:
            if name not in self.gradient_stats:
                self.gradient_stats[name] = []
            
            # Store gradients w.r.t output
            grads = grad_output[0].detach()
            
            # For higher-order analysis
            if self.taylor_order >= 2:
                self.gradient_stats[name].append(grads)
            else:
                # First-order analysis needs just the norm
                norms = torch.norm(grads, dim=(0, 2, 3))
                self.gradient_stats[name].append(norms)
                
        except Exception as e:
            print(f"Error recording gradient for {name}: {e}")
    
    def forward(self, x):
        """
        Forward pass through the model to collect statistics.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        self.model.zero_grad()
        return self.model(x)
    
    def _compute_taylor_importance(self, layer_name):
        """
        Compute filter importance using Taylor expansion.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Tensor of importance scores for each filter in the layer
        """
        if layer_name not in self.activation_stats or layer_name not in self.gradient_stats:
            return None
            
        activations_list = self.activation_stats[layer_name]
        gradients_list = self.gradient_stats[layer_name]
        
        # Ensure we have matching statistics
        min_samples = min(len(activations_list), len(gradients_list))
        if min_samples == 0:
            return None
            
        # Compute Taylor importance based on order
        importance = None
        
        try:
            if self.taylor_order == 1:
                # First-order Taylor: |g|
                importance = torch.zeros(activations_list[0].size(0), device=self.device)
                for i in range(min_samples):
                    importance += torch.abs(gradients_list[i])
                    
            elif self.taylor_order == 2:
                # Second-order Taylor: |g * h|
                importance = torch.zeros(activations_list[0].size(1), device=self.device)
                for i in range(min_samples):
                    # Element-wise product of gradient and activation
                    act = activations_list[i]
                    grad = gradients_list[i]
                    
                    # Compute per-filter importance
                    for f_idx in range(act.size(1)):  # For each filter
                        filter_act = act[:, f_idx, :, :]
                        filter_grad = grad[:, f_idx, :, :]
                        importance[f_idx] += torch.sum(torch.abs(filter_act * filter_grad)).item()
                        
            elif self.taylor_order == 3:
                # Third-order Taylor (approximate): |g * h * h|
                importance = torch.zeros(activations_list[0].size(1), device=self.device)
                for i in range(min_samples):
                    act = activations_list[i]
                    grad = gradients_list[i]
                    
                    # Compute per-filter importance with higher-order term
                    for f_idx in range(act.size(1)):  # For each filter
                        filter_act = act[:, f_idx, :, :]
                        filter_grad = grad[:, f_idx, :, :]
                        # Approximate third-order term
                        importance[f_idx] += torch.sum(torch.abs(filter_act * filter_act * filter_grad)).item()
            
            # Normalize and return
            if importance is not None:
                # Prevent division by zero
                if torch.sum(importance) > 0:
                    importance = importance / torch.sum(importance)
                    
            return importance
            
        except Exception as e:
            print(f"Error computing Taylor importance for {layer_name}: {e}")
            return None

    def normalize_ranks_per_layer(self):
        """Normalize the filter ranks within each layer."""
        self.filter_ranks = {}
        
        # Compute Taylor importance for each layer
        for layer_name in self.activation_stats.keys():
            if layer_name not in self.gradient_stats:
                continue
                
            layer_importance = self._compute_taylor_importance(layer_name)
            if layer_importance is not None:
                self.filter_ranks[layer_name] = layer_importance
        
        # Get index mapping for each layer
        self.layer_indices = {}
        idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.layer_indices[name] = idx
                idx += 1
    
    def get_pruning_plan(self, num_filters_to_prune):
        """
        Get the pruning plan ensuring at least one filter remains per layer.
        
        Args:
            num_filters_to_prune: Total number of filters to prune
            
        Returns:
            List of (layer_index, filter_index) tuples to prune
        """
        if not self.filter_ranks:
            print("No filter ranks computed. Call normalize_ranks_per_layer first.")
            return []
            
        # Calculate minimum filters to keep per layer (at least 1)
        min_filters_per_layer = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Ensure at least 2 filters remain (more conservative)
                min_filters_per_layer[name] = min(2, module.out_channels)
        
        # Create a flat list of all filters with their importance scores
        all_filters = []
        for layer_name, ranks in self.filter_ranks.items():
            if layer_name not in self.layer_indices:
                continue
                
            layer_idx = self.layer_indices[layer_name]
            
            # Get the corresponding module
            module = None
            for name, mod in self.model.named_modules():
                if name == layer_name:
                    module = mod
                    break
            
            if not module or not isinstance(module, nn.Conv2d):
                continue
                
            for filter_idx, importance in enumerate(ranks):
                all_filters.append((layer_idx, filter_idx, importance.item()))
        
        # Sort filters by importance (ascending, less important first)
        all_filters.sort(key=lambda x: x[2])
        
        # Count filters per layer
        filters_per_layer = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                filters_per_layer[name] = module.out_channels
        
        # Select filters to prune while respecting minimum filters per layer
        filters_to_prune = []
        for layer_idx, filter_idx, _ in all_filters:
            # Get layer name from index
            layer_name = None
            for name, idx in self.layer_indices.items():
                if idx == layer_idx:
                    layer_name = name
                    break
            
            if not layer_name or layer_name not in filters_per_layer:
                continue
                
            # Check if we can prune more filters from this layer
            if filters_per_layer[layer_name] - 1 >= min_filters_per_layer.get(layer_name, 1):
                filters_to_prune.append((layer_idx, filter_idx))
                filters_per_layer[layer_name] -= 1
                
                if len(filters_to_prune) >= num_filters_to_prune:
                    break
        
        # Ensure we don't prune more than requested
        return filters_to_prune[:num_filters_to_prune]