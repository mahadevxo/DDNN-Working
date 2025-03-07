import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from GradientFlowAnalyzer import GradientFlowAnalyzer

class GradientOptimizer:
    """Uses gradient flow analysis to optimize neural network training and pruning."""
    
    def __init__(self, model, output_dir='gradient_optimization_results'):
        """
        Initialize the gradient optimizer.
        
        Args:
            model: PyTorch model to optimize
            output_dir: Directory to save optimization results
        """
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create gradient analyzer
        self.gradient_analyzer = GradientFlowAnalyzer(model, output_dir)
        
        # History tracking
        self.lr_history = defaultdict(list)
        self.gradient_history = defaultdict(list)
        self.optimization_history = []
        
    def register_hooks(self):
        """Register hooks for gradient analysis"""
        self.gradient_analyzer.register_hooks()
        return self
        
    def remove_hooks(self):
        """Remove gradient analysis hooks"""
        self.gradient_analyzer.remove_hooks()
        return self
    
    def compute_layer_lr_multipliers(self, base_lr=0.001, min_factor=0.1, max_factor=3.0):
        """
        Compute per-layer learning rate multipliers based on gradient flow.
        
        Args:
            base_lr: Base learning rate
            min_factor: Minimum multiplier factor
            max_factor: Maximum multiplier factor
        
        Returns:
            Dictionary mapping layer names to learning rate multipliers
        """
        gradients = self.gradient_analyzer.gradients
        if not gradients:
            print("No gradients recorded. Run backward pass with hooks registered first.")
            return {}
            
        # Get median gradient as reference
        median_grad = np.median(list(gradients.values()))
        if median_grad == 0:  # Avoid division by zero
            median_grad = 1e-8
            
        lr_multipliers = {}
        for layer_name, grad_value in gradients.items():
            # Inverse relationship - smaller gradients get larger learning rates
            # but we constrain to a reasonable range
            if grad_value == 0:
                ratio = max_factor  # Maximum boost for zero gradients
            else:
                ratio = median_grad / grad_value
                ratio = np.clip(ratio, min_factor, max_factor)
            
            lr_multipliers[layer_name] = ratio
            
            # Store history for analysis
            self.lr_history[layer_name].append(ratio)
            self.gradient_history[layer_name].append(grad_value)
            
        return lr_multipliers
    
    def apply_gradient_based_lr(self, optimizer, epoch):
        """
        Apply gradient-based learning rates to optimizer.
        
        Args:
            optimizer: PyTorch optimizer
            epoch: Current epoch number
        
        Returns:
            Modified optimizer
        """
        lr_multipliers = self.compute_layer_lr_multipliers()
        if not lr_multipliers:
            return optimizer
            
        # Map parameter names to their position in the optimizer
        param_to_idx = {}
        for i, param_group in enumerate(optimizer.param_groups):
            for param in param_group['params']:
                for name, module_param in self.model.named_parameters():
                    if param is module_param:
                        param_to_idx[name] = i
        
        # Create new param groups or adjust existing ones
        default_lr = optimizer.param_groups[0]['lr']
        
        # First, collect parameters for each multiplier
        param_groups = defaultdict(list)
        named_parameters = list(self.model.named_parameters())
        
        for name, param in named_parameters:
            # Find the best matching layer name (could be a prefix)
            matches = [layer_name for layer_name in lr_multipliers.keys() if layer_name in name]
            if matches:
                # Use the longest match (most specific)
                best_match = max(matches, key=len)
                multiplier = lr_multipliers[best_match]
                
                # Group parameters by their multiplier (rounded to 2 decimals for fewer groups)
                mult_key = round(multiplier, 2)
                param_groups[mult_key].append((name, param))
            
        # Create new optimizer with parameter groups
        param_group_list = []
        for mult, params in param_groups.items():
            param_group_list.append({
                'params': [p for _, p in params],
                'lr': default_lr * mult,
                'param_names': [n for n, _ in params]
            })
            
        # Create a new optimizer with these param groups
        if isinstance(optimizer, torch.optim.SGD):
            new_optimizer = torch.optim.SGD(
                param_group_list,
                lr=default_lr,
                momentum=optimizer.param_groups[0].get('momentum', 0),
                weight_decay=optimizer.param_groups[0].get('weight_decay', 0)
            )
        elif isinstance(optimizer, torch.optim.Adam):
            new_optimizer = torch.optim.Adam(
                param_group_list,
                lr=default_lr,
                betas=optimizer.param_groups[0].get('betas', (0.9, 0.999)),
                weight_decay=optimizer.param_groups[0].get('weight_decay', 0)
            )
        else:
            # Fall back to original optimizer if type not supported
            print(f"Optimizer type {type(optimizer)} not supported for gradient-based LR")
            return optimizer
        
        # Log optimization actions
        action = {
            'epoch': epoch,
            'type': 'learning_rate_adjustment',
            'groups': len(param_group_list),
            'multipliers': {
                f"group_{i}": {
                    "multiplier": round(pg['lr'] / default_lr, 2),
                    "num_params": len(pg['params']),
                    "param_names": pg.get('param_names', [])[:3]  # Just log first 3 for brevity
                }
                for i, pg in enumerate(new_optimizer.param_groups)
            }
        }
        self.optimization_history.append(action)
        
        # Plot updated learning rates
        self._plot_learning_rates(epoch)
        
        return new_optimizer
    
    def prioritize_filters_for_pruning(self, pruner, num_filters_to_prune):
        """
        Prioritize filters for pruning based on gradient flow.
        
        Args:
            pruner: FilterPruner instance with filter rankings
            num_filters_to_prune: Number of filters to select for pruning
            
        Returns:
            List of (layer_index, filter_index) pairs to prune
        """
        if not self.gradient_analyzer.gradients:
            print("No gradients recorded. Using standard pruning criteria.")
            return pruner.get_pruning_plan(num_filters_to_prune)
            
        # Get the standard pruning plan
        standard_plan = pruner.get_pruning_plan(num_filters_to_prune)
        
        # Get per-layer gradient information
        layer_gradients = {}
        for name, grad_value in self.gradient_analyzer.gradients.items():
            # Extract layer index if name format matches 'features.{index}'
            if 'features.' in name:
                try:
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if part == 'features' and i+1 < len(parts) and parts[i+1].isdigit():
                            layer_idx = int(parts[i+1])
                            if layer_idx not in layer_gradients:
                                layer_gradients[layer_idx] = []
                            layer_gradients[layer_idx].append(grad_value)
                except (ValueError, IndexError):
                    continue
        
        # Compute average gradient per layer
        avg_layer_gradients = {
            layer_idx: np.mean(grads) if grads else 0
            for layer_idx, grads in layer_gradients.items()
        }
        
        # Adjust pruning plan based on gradients (prioritize layers with smaller gradients)
        if avg_layer_gradients:
            # Count filters per layer in the original plan
            layer_counts = {}
            for layer_idx, _ in standard_plan:
                layer_counts[layer_idx] = layer_counts.get(layer_idx, 0) + 1
            
            # Adjust filter counts based on gradient flow
            total_grad = sum(avg_layer_gradients.values()) if avg_layer_gradients else 1
            if total_grad == 0:
                total_grad = 1  # Avoid division by zero
            
            adjusted_counts = {}
            for layer_idx, count in layer_counts.items():
                # Layers with smaller gradients get more filters pruned
                if layer_idx in avg_layer_gradients and avg_layer_gradients[layer_idx] > 0:
                    gradient_factor = 1.0 - (avg_layer_gradients[layer_idx] / total_grad)
                    adjusted_count = int(round(count * (1.0 + gradient_factor * 0.5)))
                    adjusted_counts[layer_idx] = min(adjusted_count, 
                                                    pruner.filter_ranks[layer_idx].size(0) - 1)  # Leave at least one filter
                else:
                    adjusted_counts[layer_idx] = count
            
            # Create a new pruning plan based on adjusted counts
            adjusted_plan = []
            for layer_idx, count in adjusted_counts.items():
                # Get the lowest ranking filters for this layer
                filter_ranks = [(i, pruner.filter_ranks[layer_idx][i].item()) 
                                for i in range(pruner.filter_ranks[layer_idx].size(0))]
                filter_ranks.sort(key=lambda x: x[1])
                
                # Take the specified count
                for i in range(min(count, len(filter_ranks))):
                    adjusted_plan.append((layer_idx, filter_ranks[i][0]))
            
            # Ensure we're pruning exactly num_filters_to_prune
            if len(adjusted_plan) > num_filters_to_prune:
                # Remove some filters from the plan
                adjusted_plan = adjusted_plan[:num_filters_to_prune]
            elif len(adjusted_plan) < num_filters_to_prune:
                # Add more filters from the standard plan
                remaining = [x for x in standard_plan if x not in adjusted_plan]
                adjusted_plan.extend(remaining[:num_filters_to_prune - len(adjusted_plan)])
            
            # Log optimization action
            action = {
                'type': 'pruning_adjustment',
                'standard_filters': len(standard_plan),
                'adjusted_filters': len(adjusted_plan),
                'layer_adjustments': {
                    layer_idx: {
                        'original_count': layer_counts.get(layer_idx, 0),
                        'adjusted_count': adjusted_counts.get(layer_idx, 0),
                        'avg_gradient': avg_layer_gradients.get(layer_idx, 0)
                    }
                    for layer_idx in set(layer_counts.keys()) | set(adjusted_counts.keys())
                }
            }
            self.optimization_history.append(action)
            
            return adjusted_plan
        
        return standard_plan
    
    def suggest_architecture_improvements(self):
        """
        Analyze gradient flow and suggest architectural improvements.
        
        Returns:
            Dictionary of suggestions
        """
        if not self.gradient_analyzer.gradients:
            return {"error": "No gradients recorded. Run backward pass with hooks registered first."}
            
        gradients = self.gradient_analyzer.gradients
        layer_types = self.gradient_analyzer.layer_types
        suggestions = []
        
        # Find layers with extremely low gradients (potentially useless)
        very_low_threshold = 1e-6
        very_low_layers = [name for name, grad in gradients.items() 
                          if grad < very_low_threshold]
        
        if very_low_layers:
            for layer in very_low_layers:
                layer_type = layer_types.get(layer, "unknown")
                if 'conv' in layer_type.lower():
                    suggestions.append(f"Consider removing or reducing filters in {layer} ({layer_type}) - gradient is very low ({gradients[layer]:.8f})")
                elif 'linear' in layer_type.lower():
                    suggestions.append(f"Consider reducing neurons in {layer} ({layer_type}) - gradient is very low ({gradients[layer]:.8f})")
        
        # Check for gradient explosion
        high_threshold = 1.0  # Relatively high gradient value
        high_grad_layers = [name for name, grad in gradients.items() 
                           if grad > high_threshold]
        
        if high_grad_layers:
            for layer in high_grad_layers:
                layer_type = layer_types.get(layer, "unknown")
                suggestions.append(f"Consider adding batch normalization after {layer} ({layer_type}) - gradient is high ({gradients[layer]:.4f})")
        
        # Check for gradient imbalance between adjacent layers
        sorted_layers = []
        for name in gradients.keys():
            # Try to extract numerical position in layers
            if 'features.' in name:
                try:
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if part == 'features' and i+1 < len(parts) and parts[i+1].isdigit():
                            pos = int(parts[i+1])
                            sorted_layers.append((pos, name))
                            break
                except:
                    continue

        if sorted_layers:
            sorted_layers.sort()  # Sort by position
            prev_grad = None
            prev_name = None
            
            for _, name in sorted_layers:
                if prev_grad is not None:
                    ratio = gradients[name] / prev_grad if prev_grad > 0 else float('inf')
                    if ratio > 10 or ratio < 0.1:
                        suggestions.append(f"Large gradient imbalance between {prev_name} ({prev_grad:.6f}) and {name} ({gradients[name]:.6f}) - consider adding skip connections")
                
                prev_grad = gradients[name]
                prev_name = name
        
        # Give general recommendations based on overall gradient health
        grad_stats = self.gradient_analyzer.analyze_vanishing_gradients()
        
        if grad_stats["vanishing_gradient_risk"] == "High":
            suggestions.append("Model shows signs of vanishing gradients - consider using residual connections or different activation functions (e.g., LeakyReLU)")
        
        if len(suggestions) == 0:
            suggestions.append("No specific architectural issues detected in gradient flow")
            
        return {
            "suggestions": suggestions,
            "gradient_statistics": grad_stats
        }
    
    def _plot_learning_rates(self, epoch):
        """Plot the learning rate adjustments"""
        if not self.lr_history:
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot average LR multiplier per layer type
        layer_types = self.gradient_analyzer.layer_types
        lr_by_type = defaultdict(list)
        
        for layer_name, lr_history in self.lr_history.items():
            if lr_history:  # Make sure we have data
                layer_type = layer_types.get(layer_name, "unknown")
                lr_by_type[layer_type].append(lr_history[-1])  # Get the latest LR
        
        # Prepare data for plotting
        types = []
        lr_means = []
        lr_stds = []
        
        for layer_type, multipliers in lr_by_type.items():
            if multipliers:
                types.append(layer_type)
                lr_means.append(np.mean(multipliers))
                lr_stds.append(np.std(multipliers))
        
        # Sort by type name
        sorted_indices = np.argsort(types)
        types = [types[i] for i in sorted_indices]
        lr_means = [lr_means[i] for i in sorted_indices]
        lr_stds = [lr_stds[i] for i in sorted_indices]
        
        # Plot
        plt.bar(range(len(types)), lr_means, yerr=lr_stds, alpha=0.7)
        plt.xticks(range(len(types)), types, rotation=45)
        plt.ylabel("Learning Rate Multiplier")
        plt.title(f"Learning Rate Adjustments by Layer Type (Epoch {epoch})")
        plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.5, label="Base LR")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, f'lr_adjustments_epoch_{epoch}.png'))
        plt.close()
        
    def save_optimization_history(self):
        """Save optimization history to a file"""
        import json
        
        # Convert any non-serializable values to strings
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        serializable_history = convert_to_serializable(self.optimization_history)
        
        # Save to file
        with open(os.path.join(self.output_dir, 'optimization_history.json'), 'w') as f:
            json.dump(serializable_history, f, indent=2)
