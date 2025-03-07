import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

class GradientFlowAnalyzer:
    def __init__(self, model, output_dir='gradient_flow_results'):
        """
        Initializes the gradient flow analyzer.
        
        Args:
            model: PyTorch model to analyze
            output_dir: Directory to save gradient flow visualizations
        """
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # For storing gradients during training
        self.gradients = {}
        self.hooks = []
        self.layer_types = {}
        
    def _make_hook(self, name):
        """Create a hook function for a specific layer."""
        def hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple) and len(grad_output) > 0:
                # Store the gradient norm for this layer
                if grad_output[0] is not None:
                    self.gradients[name] = grad_output[0].detach().cpu().norm(2).item()
        return hook
    
    def register_hooks(self):
        """Register hooks on all layers to capture gradients."""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Only register hooks on leaf modules
                # Skip certain layers like dropout that don't have useful gradients
                if not isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm2d)):
                    self.layer_types[name] = module.__class__.__name__
                    hook = module.register_full_backward_hook(self._make_hook(name))
                    self.hooks.append(hook)
        return self
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        return self
    
    def clear_gradients(self):
        """Clear stored gradients."""
        self.gradients = {}
        return self
    
    def plot_grad_flow(self, epoch=None):
        """
        Plot gradients flowing through the network during training.
        
        Args:
            epoch: Current training epoch (used for plot title)
        """
        if not self.gradients:
            print("No gradients recorded. Run backward pass with hooks registered first.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Sort layers by type then name for better visualization
        sorted_layers = sorted(self.gradients.items(), 
                              key=lambda x: (self.layer_types.get(x[0], ''), x[0]))
        
        layer_names = [name for name, _ in sorted_layers]
        grad_values = [value for _, value in sorted_layers]
        
        # Create bar colors based on layer type
        colors = []
        for name, _ in sorted_layers:
            layer_type = self.layer_types.get(name, '')
            if 'Conv' in layer_type:
                colors.append('blue')
            elif 'Linear' in layer_type:
                colors.append('red')
            elif 'BatchNorm' in layer_type:
                colors.append('green')
            else:
                colors.append('gray')
        
        plt.bar(range(len(grad_values)), grad_values, color=colors, alpha=0.5)
        plt.hlines(y=0, xmin=0, xmax=len(grad_values)+1, linewidth=1, color="k")
        plt.xticks(range(len(grad_values)), layer_names, rotation='vertical')
        plt.xlim(xmin=0, xmax=len(grad_values))
        plt.ylabel("Gradient Norm")
        plt.title(f"Gradient Flow (Epoch: {epoch})" if epoch is not None else "Gradient Flow")
        
        # Add legend
        legend_elements = [
            Line2D([0], [0], color='blue', lw=4, label='Conv'),
            Line2D([0], [0], color='red', lw=4, label='Linear'),
            Line2D([0], [0], color='green', lw=4, label='BatchNorm'),
            Line2D([0], [0], color='gray', lw=4, label='Other')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save figure if epoch is specified
        if epoch is not None:
            plt.savefig(os.path.join(self.output_dir, f'grad_flow_epoch_{epoch}.png'))
        
        plt.close()
        return self
    
    def analyze_vanishing_gradients(self):
        """
        Analyze if the model suffers from vanishing gradients.
        Returns a dictionary with vanishing gradient statistics.
        """
        if not self.gradients:
            print("No gradients recorded. Run backward pass with hooks registered first.")
            return {}
        
        grad_values = list(self.gradients.values())
        avg_grad = np.mean(grad_values)
        median_grad = np.median(grad_values)
        min_grad = min(grad_values)
        max_grad = max(grad_values)
        
        # Identify potentially problematic layers
        small_grad_threshold = 1e-4
        small_grad_layers = [name for name, grad in self.gradients.items() 
                             if grad < small_grad_threshold]
        
        return {
            "average_gradient": avg_grad,
            "median_gradient": median_grad,
            "min_gradient": min_grad,
            "max_gradient": max_grad,
            "gradient_range": max_grad - min_grad,
            "small_gradient_layers": small_grad_layers,
            "vanishing_gradient_risk": "High" if avg_grad < small_grad_threshold else 
                                      "Medium" if avg_grad < small_grad_threshold * 10 else 
                                      "Low"
        }
    
    def plot_gradient_distribution(self, epoch=None):
        """
        Plot histogram of gradient distributions across layers.
        
        Args:
            epoch: Current training epoch (used for plot title and saving)
        """
        if not self.gradients:
            print("No gradients recorded. Run backward pass with hooks registered first.")
            return
        
        plt.figure(figsize=(10, 6))
        
        grad_values = list(self.gradients.values())
        plt.hist(grad_values, bins=50, alpha=0.7, color='blue')
        plt.xlabel("Gradient Norm")
        plt.ylabel("Count")
        plt.title(f"Gradient Distribution (Epoch: {epoch})" if epoch is not None else "Gradient Distribution")
        plt.grid(True, alpha=0.3)
        
        # Add statistics to the plot
        stats = self.analyze_vanishing_gradients()
        stat_text = f"Mean: {stats['average_gradient']:.6f}\n" \
                   f"Median: {stats['median_gradient']:.6f}\n" \
                   f"Min: {stats['min_gradient']:.6f}\n" \
                   f"Max: {stats['max_gradient']:.6f}\n" \
                   f"Risk: {stats['vanishing_gradient_risk']}"
        
        plt.annotate(stat_text, xy=(0.7, 0.7), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))
        
        # Save figure if epoch is specified
        if epoch is not None:
            plt.savefig(os.path.join(self.output_dir, f'grad_dist_epoch_{epoch}.png'))
        
        plt.close()
        return self
