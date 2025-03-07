import torch
import numpy as np
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from GradientFlowAnalyzer import GradientFlowAnalyzer

class GradientOptimizer:
    """
    Uses gradient flow information to adjust learning rates per layer
    and to help prioritize filters for pruning.
    """
    def __init__(self, model, output_dir='gradient_optimization_results'):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.gradient_analyzer = GradientFlowAnalyzer(model, output_dir)
        self.lr_history = defaultdict(list)
        self.optimization_history = []

    def register_hooks(self):
        # First remove any existing hooks to avoid duplicates
        self.remove_hooks()
        
        # Then register new hooks with safety wrapper
        self.gradient_analyzer = GradientFlowAnalyzer(self.model, self.output_dir)
        
        # Register with clone-safety when creating hooks
        self.gradient_analyzer.register_hooks()
        return self

    def remove_hooks(self):
        if hasattr(self, 'gradient_analyzer'):
            self.gradient_analyzer.remove_hooks()
        return self
    
    def safe_clone(self, tensor):
        """Safely clone a tensor if it requires gradients to avoid in-place modification issues"""
        if tensor is not None and tensor.requires_grad:
            return tensor.clone()
        return tensor

    def compute_lr_multipliers(self, base_lr=0.001, min_factor=0.1, max_factor=3.0):
        """
        Compute per-layer learning rate multipliers inversely proportional to gradient norms.
        Layers with lower gradients get a higher multiplier.
        """
        gradients = self.gradient_analyzer.gradients
        if not gradients:
            print("No gradients recorded. Run a backward pass with hooks registered first.")
            return {}
        median_grad = np.median(list(gradients.values()))
        if median_grad == 0:
            median_grad = 1e-8
        lr_multipliers = {}
        for layer, grad_val in gradients.items():
            # Use safe value comparison to avoid potential view issues
            ratio = median_grad/grad_val if grad_val > 0 else max_factor
            ratio = np.clip(ratio, min_factor, max_factor)
            lr_multipliers[layer] = ratio
            self.lr_history[layer].append(ratio)
        return lr_multipliers

    def apply_gradient_based_lr(self, optimizer, epoch):
        """
        Adjust optimizer parameter groups based on computed learning rate multipliers.
        """
        multipliers = self.compute_lr_multipliers()
        if not multipliers:
            return optimizer

        default_lr = optimizer.param_groups[0]['lr']
        # Group parameters according to their layer name match 
        new_groups = defaultdict(list)
        for name, param in self.model.named_parameters():
            # Find a matching layer key (if any)
            for layer, mult in multipliers.items():
                if layer in name:
                    new_groups[round(mult, 2)].append(param)
                    break
            else:
                new_groups[1.0].append(param)
        # Build new param groups with adjusted learning rates
        new_param_groups = []
        for mult, params in new_groups.items():
            new_param_groups.append({'params': params, 'lr': default_lr * mult})
        # Recreate optimizer with new param groups based on type
        if isinstance(optimizer, torch.optim.SGD):
            new_optimizer = torch.optim.SGD(
                new_param_groups,
                lr=default_lr,
                momentum=optimizer.param_groups[0].get('momentum', 0),
                weight_decay=optimizer.param_groups[0].get('weight_decay', 0)
            )
        elif isinstance(optimizer, torch.optim.Adam):
            new_optimizer = torch.optim.Adam(
                new_param_groups,
                lr=default_lr,
                betas=optimizer.param_groups[0].get('betas', (0.9, 0.999)),
                weight_decay=optimizer.param_groups[0].get('weight_decay', 0)
            )
        else:
            print(f"Optimizer type {type(optimizer)} not supported for gradient-based adjustment.")
            return optimizer
        
        self.optimization_history.append({
            'epoch': epoch,
            'num_groups': len(new_param_groups),
            'multipliers': {mult: len(params) for mult, params in new_groups.items()}
        })
        self._plot_lr_adjustments(epoch, new_param_groups, default_lr)
        return new_optimizer

    def _plot_lr_adjustments(self, epoch, param_groups, base_lr):
        """Plot a bar chart of the learning rate multipliers across parameter groups."""
        multipliers = [pg['lr']/base_lr for pg in param_groups]
        group_ids = list(range(len(multipliers)))
        plt.figure(figsize=(10,6))
        plt.bar(group_ids, multipliers, color='blue', alpha=0.7)
        plt.xlabel("Parameter Group")
        plt.ylabel("LR Multiplier")
        plt.title(f"Learning Rate Multipliers (Epoch {epoch})")
        plt.xticks(group_ids)
        plt.axhline(y=1.0, color='red', linestyle='--', label="Base LR")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"lr_multipliers_epoch_{epoch}.png"))
        plt.close()

    def prioritize_filters_for_pruning(self, pruner, num_filters_to_prune):
        """
        Adjust the standard pruning plan using the stored gradient norms.
        Filters in layers with lower average gradients are prioritized.
        """
        if not self.gradient_analyzer.gradients:
            print("No gradient info available; falling back to standard pruning criteria.")
            return pruner.get_pruning_plan(num_filters_to_prune)

        # Compute average gradients per layer based on keys from GradientFlowAnalyzer
        layer_gradients = {}
        for layer, grad_val in self.gradient_analyzer.gradients.items():
            layer_gradients[layer] = grad_val

        standard_plan = pruner.get_pruning_plan(num_filters_to_prune)
        adjusted_plan = []
        for layer_index, filter_index in standard_plan:
            # If a matching gradient exists and is low, boost its pruning priority
            for key, avg_grad in layer_gradients.items():
                if str(layer_index) in key and avg_grad < 1e-3:
                    adjusted_plan.append((layer_index, filter_index))
                    break
            else:
                adjusted_plan.append((layer_index, filter_index))
        # Ensure the plan contains exactly num_filters_to_prune entries
        if len(adjusted_plan) > num_filters_to_prune:
            adjusted_plan = adjusted_plan[:num_filters_to_prune]
        elif len(adjusted_plan) < num_filters_to_prune:
            remaining = [pt for pt in standard_plan if pt not in adjusted_plan]
            adjusted_plan.extend(remaining[:num_filters_to_prune - len(adjusted_plan)])
        return adjusted_plan

    def save_optimization_history(self):
        """Save recorded optimization actions for later review."""
        import json
        path = os.path.join(self.output_dir, "optimization_history.json")
        with open(path, "w") as f:
            json.dump(self.optimization_history, f, indent=2)
