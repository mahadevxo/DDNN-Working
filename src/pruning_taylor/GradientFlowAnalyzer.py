import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class SafeBackwardHook:
    """Custom hook class to safely handle gradient flow analysis"""
    def __init__(self, name, storage_dict):
        self.name = name
        self.storage_dict = storage_dict
        
    def __call__(self, module, grad_input, grad_output):
        if grad_output[0] is not None:
            # Completely detach and clone to avoid any view issues
            safe_grad = grad_output[0].detach().clone()
            self.storage_dict[self.name] = safe_grad.norm(2).cpu().item()
        # Don't return anything to avoid modifying the gradient flow

class GradientFlowAnalyzer:
    def __init__(self, model, output_dir='gradient_flow_results'):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.gradients = {}
        self.hooks = []
        self.layer_types = {}
    
    def _make_hook(self, name):
        # Use the safer hook implementation
        return SafeBackwardHook(name, self.gradients)
    
    def register_hooks(self):
        # Remove any existing hooks first
        self.remove_hooks()
        self.gradients.clear()
        
        for name, module in self.model.named_modules():
            # Register hooks only on leaf modules (no children)
            if not list(module.children()) and not isinstance(
                module, (torch.nn.Dropout, torch.nn.BatchNorm2d)
            ):
                self.layer_types[name] = module.__class__.__name__
                # Use safer hook registration
                hook = module.register_full_backward_hook(self._make_hook(name))
                self.hooks.append(hook)
        return self
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        return self
    
    def plot_grad_flow(self, epoch=None):
        if not self.gradients:
            print("No gradients recorded. Run a backward pass with hooks registered first.")
            return
        plt.figure(figsize=(12, 8))
        sorted_layers = sorted(self.gradients.items(), key=lambda x: (self.layer_types.get(x[0], ''), x[0]))
        layer_names = [name for name, _ in sorted_layers]
        grad_values = [val for _, val in sorted_layers]
        colors = []
        for name, _ in sorted_layers:
            ltype = self.layer_types.get(name, '')
            if 'Conv' in ltype:
                colors.append('blue')
            elif 'Linear' in ltype:
                colors.append('red')
            else:
                colors.append('gray')
        plt.bar(range(len(grad_values)), grad_values, color=colors, alpha=0.5)
        plt.xticks(range(len(grad_values)), layer_names, rotation=90)
        plt.xlabel("Layers")
        plt.ylabel("Gradient Norm")
        plt.title(f"Gradient Flow (Epoch {epoch})" if epoch is not None else "Gradient Flow")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"grad_flow_epoch_{epoch}.png") if epoch is not None else os.path.join(self.output_dir, "grad_flow.png")
        plt.savefig(save_path)
        plt.close()
    
    def plot_gradient_distribution(self, epoch=None):
        if not self.gradients:
            print("No gradients recorded. Run a backward pass with hooks registered first.")
            return
        plt.figure(figsize=(10, 6))
        grad_values = list(self.gradients.values())
        plt.hist(grad_values, bins=50, color='blue', alpha=0.7)
        plt.xlabel("Gradient Norm")
        plt.ylabel("Count")
        plt.title(f"Gradient Distribution (Epoch {epoch})" if epoch is not None else "Gradient Distribution")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"grad_distribution_epoch_{epoch}.png") if epoch is not None else os.path.join(self.output_dir, "grad_distribution.png")
        plt.savefig(save_path)
        plt.close()
    
    def analyze_vanishing_gradients(self):
        if not self.gradients:
            print("No gradients recorded. Run a backward pass with hooks registered first.")
            return {}
        grads = list(self.gradients.values())
        avg = np.mean(grads)
        median = np.median(grads)
        min_grad = np.min(grads)
        max_grad = np.max(grads)
        small_threshold = 1e-4
        small_layers = [name for name, val in self.gradients.items() if val < small_threshold]
        risk = "High" if avg < small_threshold else "Low"
        return {
            "average": avg,
            "median": median,
            "min": min_grad,
            "max": max_grad,
            "small_layers": small_layers,
            "risk": risk
        }
