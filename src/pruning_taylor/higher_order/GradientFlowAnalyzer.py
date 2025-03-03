import torch
import numpy as np
from collections import defaultdict

class GradientFlowAnalyzer:
    """
    Analyzes gradient flow in neural networks to identify bottlenecks
    and guide pruning decisions.
    """
    
    def __init__(self, model):
        """
        Initialize the gradient flow analyzer.
        
        Args:
            model: The neural network model to analyze
        """
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.gradient_stats = {}
        self.activation_stats = {}
        self.flow_metrics = {}
        self.hooks = []
        
    def reset(self):
        """Reset all statistics."""
        self.gradient_stats = {}
        self.activation_stats = {}
        self.flow_metrics = {}
        self.remove_hooks()
        
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def register_hooks(self):
        """Register hooks for gradient and activation collection."""
        self.remove_hooks()
        
        # Register hooks for all layers we're interested in
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                # Forward hook to capture activations
                self.hooks.append(module.register_forward_hook(
                    lambda m, inp, out, name=name: self._activation_hook(m, inp, out, name)
                ))
                
                # Backward hook to capture gradients
                if hasattr(module, 'weight') and module.weight is not None:
                    self.hooks.append(module.register_full_backward_hook(
                        lambda m, grad_in, grad_out, name=name: self._gradient_hook(m, grad_in, grad_out, name)
                    ))
    
    def _activation_hook(self, module, inp, out, name):
        """Hook to capture activations."""
        if name not in self.activation_stats:
            self.activation_stats[name] = []
            
        # Store activation statistics
        with torch.no_grad():
            if isinstance(out, tuple):
                out = out[0]
            
            # Compute statistics - mean, norm, etc.
            stats = {
                'mean': out.abs().mean().item(),
                'std': out.std().item(),
                'norm': out.norm().item(),
                'shape': list(out.shape)
            }
            
            self.activation_stats[name].append(stats)
    
    def _gradient_hook(self, module, grad_in, grad_out, name):
        """Hook to capture gradients."""
        if name not in self.gradient_stats:
            self.gradient_stats[name] = []
        
        # Store gradient statistics
        with torch.no_grad():
            if isinstance(grad_in, tuple) and len(grad_in) > 0 and grad_in[0] is not None:
                grad = grad_in[0]
                
                # Compute statistics
                stats = {
                    'mean': grad.abs().mean().item(),
                    'std': grad.std().item(),
                    'norm': grad.norm().item(),
                    'shape': list(grad.shape)
                }
                
                self.gradient_stats[name].append(stats)
    
    def analyze_one_batch(self, inputs, targets, criterion):
        """
        Analyze gradient flow for a single batch.
        
        Args:
            inputs: Input batch
            targets: Target batch
            criterion: Loss function
        """
        self.model.train()
        self.register_hooks()
        
        # Forward pass
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass to compute gradients
        self.model.zero_grad()
        loss.backward()
        
        # Compute flow metrics after backward pass
        self._compute_flow_metrics()
        
        # Clean up
        self.remove_hooks()
        
        return self.flow_metrics
    
    def _compute_flow_metrics(self):
        """Compute gradient flow metrics after collecting statistics."""
        flow_metrics = defaultdict(dict)
        
        for name in self.gradient_stats:
            if name in self.activation_stats:
                # Average across all recorded statistics
                grad_norms = [stat['norm'] for stat in self.gradient_stats[name]]
                act_norms = [stat['norm'] for stat in self.activation_stats[name]]
                
                if grad_norms and act_norms:
                    avg_grad_norm = sum(grad_norms) / len(grad_norms)
                    avg_act_norm = sum(act_norms) / len(act_norms)
                    
                    # Compute various flow metrics
                    flow_metrics[name]['grad_norm'] = avg_grad_norm
                    flow_metrics[name]['act_norm'] = avg_act_norm
                    flow_metrics[name]['product'] = avg_grad_norm * avg_act_norm
                    
                    # Taylor-based importance
                    flow_metrics[name]['taylor_importance'] = flow_metrics[name]['product']
                    
                    # Normalized metrics
                    if avg_act_norm > 0:
                        flow_metrics[name]['grad_to_act_ratio'] = avg_grad_norm / avg_act_norm
                    else:
                        flow_metrics[name]['grad_to_act_ratio'] = 0
        
        self.flow_metrics = dict(flow_metrics)
        return self.flow_metrics
    
    def get_bottleneck_layers(self, top_n=3):
        """
        Identify potential bottleneck layers based on gradient flow.
        
        Args:
            top_n: Number of top bottleneck candidates to return
            
        Returns:
            List of (layer_name, metric) tuples for potential bottlenecks
        """
        if not self.flow_metrics:
            print("No flow metrics available. Run analyze_one_batch first.")
            return []
        
        # Potential indicators of bottlenecks
        bottleneck_candidates = []
        
        for name, metrics in self.flow_metrics.items():
            # High gradient-to-activation ratio might indicate a bottleneck
            if 'grad_to_act_ratio' in metrics:
                bottleneck_candidates.append((name, metrics['grad_to_act_ratio']))
        
        # Sort by metric value (highest first)
        bottleneck_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return bottleneck_candidates[:top_n]
    
    def get_layer_importance(self):
        """
        Get layer importance based on Taylor criteria.
        
        Returns:
            Dictionary mapping layer names to importance values
        """
        importance_dict = {}
        
        for name, metrics in self.flow_metrics.items():
            if 'taylor_importance' in metrics:
                importance_dict[name] = metrics['taylor_importance']
        
        return importance_dict
    
    def visualize_gradient_flow(self):
        """
        Generate data for gradient flow visualization.
        In a real implementation, this would create a useful visualization.
        
        Returns:
            Dictionary with visualization data
        """
        if not self.flow_metrics:
            print("No flow metrics available. Run analyze_one_batch first.")
            return {}
        
        # Prepare data for visualization
        layer_names = list(self.flow_metrics.keys())
        grad_norms = [self.flow_metrics[name].get('grad_norm', 0) for name in layer_names]
        act_norms = [self.flow_metrics[name].get('act_norm', 0) for name in layer_names]
        importance = [self.flow_metrics[name].get('taylor_importance', 0) for name in layer_names]
        
        # Return the data that could be used for visualization
        return {
            'layer_names': layer_names,
            'gradient_norms': grad_norms,
            'activation_norms': act_norms,
            'taylor_importance': importance
        }