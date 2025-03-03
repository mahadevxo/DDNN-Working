import torch
from heapq import nsmallest
from operator import itemgetter
import numpy as np

class EnhancedFilterPruner:
    def __init__(self, model, taylor_order=2, use_gradient_flow=True):
        """
        Enhanced Filter Pruner using higher-order Taylor approximation and gradient flow analysis.
        
        Args:
            model: The neural network model to prune
            taylor_order: Order of Taylor expansion (1, 2, or 3)
            use_gradient_flow: Whether to incorporate gradient flow analysis
        """
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.taylor_order = taylor_order
        self.use_gradient_flow = use_gradient_flow
        self.reset()
        
    def reset(self):
        self.filter_ranks = {}
        self.activations = []
        self.gradients = []
        self.second_gradients = []  # For second-order derivatives
        self.third_gradients = []   # For third-order derivatives
        self.gradient_flow = {}     # For gradient flow analysis
        self.activation_to_layer = {}
        self.grad_index = 0
        
    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.second_gradients = []
        self.third_gradients = []
        self.grad_index = 0
        self.model.eval()
        self.model.zero_grad()
        
        activation_index = 0
        for layer_index, layer in enumerate(self.model.features):   
            x = layer(x)
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                x.retain_grad()  # Ensure we retain gradients for higher-order computation
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer_index
                
                # Register hooks for computing Taylor expansion and gradient flow
                if self.taylor_order >= 1:
                    x.register_hook(lambda grad, idx=activation_index: self._compute_first_order(grad, idx))
                    
                if self.taylor_order >= 2:
                    # Additional hook for second-order
                    x.register_hook(lambda grad, idx=activation_index: self._register_second_order_hook(grad, idx))
                    
                if self.taylor_order >= 3:
                    # Additional hook for third-order
                    x.register_hook(lambda grad, idx=activation_index: self._register_third_order_hook(grad, idx))
                    
                if self.use_gradient_flow:
                    x.register_hook(lambda grad, idx=activation_index: self._compute_gradient_flow(grad, idx))
                    
                activation_index += 1
                
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)
        return x
    
    def _compute_first_order(self, grad, activation_index):
        """Compute first-order Taylor expansion term."""
        activation = self.activations[activation_index]
        taylor = activation * grad  # First-order term: a * dL/da
        
        # Average across batch and spatial dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).abs().data
        
        # Initialize filter_ranks for this activation if not already done
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to(self.device)
            
        # Update the ranks with first-order term
        self.filter_ranks[activation_index] += taylor
        
        # Store gradients for higher-order computations
        self.gradients.append(grad.detach().clone())
    
    def _register_second_order_hook(self, grad, activation_index):
        """Register hook for computing second-order derivatives."""
        if self.taylor_order >= 2:
            activation = self.activations[activation_index]
            
            # Create a hook to compute and accumulate second-order derivatives
            def second_order_hook():
                with torch.enable_grad():
                    # Compute d²L/da² using autograd
                    second_grad = torch.autograd.grad(
                        outputs=grad.sum(),
                        inputs=activation,
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    
                    # Second-order term: 0.5 * a² * d²L/da²
                    second_term = 0.5 * (activation ** 2) * second_grad
                    second_term = second_term.mean(dim=(0, 2, 3)).abs().data
                    
                    # Add to filter ranks
                    self.filter_ranks[activation_index] += second_term
                    
                    # Store for third-order computation
                    self.second_gradients.append(second_grad.detach().clone())
            
            # Execute second-order computation if tensors require gradients
            if activation.requires_grad:
                try:
                    second_order_hook()
                except Exception as e:
                    print(f"Error in second-order hook: {e}")
    
    def _register_third_order_hook(self, grad, activation_index):
        """Register hook for computing third-order derivatives."""
        if self.taylor_order < 3:
            return
        activation = self.activations[activation_index]

        # Create a hook to compute and accumulate third-order derivatives
        def third_order_hook():
            with torch.enable_grad():
                try:
                    # Ensure we have second-order gradients
                    if len(self.second_gradients) <= activation_index:
                        return

                    second_grad = self.second_gradients[activation_index]

                    # Compute d³L/da³ using autograd
                    third_grad = torch.autograd.grad(
                        outputs=second_grad.sum(),
                        inputs=activation,
                        create_graph=True,
                        retain_graph=True
                    )[0]

                    # Third-order term: (1/6) * a³ * d³L/da³
                    third_term = (1/6) * (activation ** 3) * third_grad
                    third_term = third_term.mean(dim=(0, 2, 3)).abs().data

                    # Add to filter ranks
                    self.filter_ranks[activation_index] += third_term

                    # Store for potential further computations
                    self.third_gradients.append(third_grad.detach().clone())
                except Exception as e:
                    print(f"Error in third-order hook: {e}")

        # Execute third-order computation if tensors require gradients
        if activation.requires_grad:
            try:
                third_order_hook()
            except Exception as e:
                print(f"Error in third-order hook registration: {e}")
    
    def _compute_gradient_flow(self, grad, activation_index):
        """Analyze gradient flow through the network."""
        if self.use_gradient_flow:
            activation = self.activations[activation_index]
            layer_index = self.activation_to_layer[activation_index]
            
            # Initialize gradient flow tracking for this layer
            if layer_index not in self.gradient_flow:
                self.gradient_flow[layer_index] = torch.FloatTensor(activation.size(1)).zero_()
                self.gradient_flow[layer_index] = self.gradient_flow[layer_index].to(self.device)
            
            # Compute gradient norm per filter
            grad_norm = grad.norm(dim=(0, 2, 3)).data
            
            # Update gradient flow metrics
            self.gradient_flow[layer_index] += grad_norm
            
            # Adjust filter ranks based on gradient flow analysis
            if activation_index in self.filter_ranks:
                # Weight the importance by gradient flow (smoother flow = more important)
                grad_flow_weight = torch.sigmoid(grad_norm)
                self.filter_ranks[activation_index] *= grad_flow_weight
    
    def lowest_ranking_filters(self, num):
        """Get the lowest ranking filters based on computed importance."""
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
        return nsmallest(num, data, itemgetter(2))
    
    def normalize_ranks_per_layer(self):
        """Normalize the ranks within each layer for fair comparison."""
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / torch.sqrt(torch.sum(v * v) + 1e-8)  # Added epsilon for numerical stability
            self.filter_ranks[i] = v.cpu()
            
    def get_pruning_plan(self, num_filters_to_prune):
        """Create a plan for which filters to prune."""
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
        
        # Group filters by layer
        filters_to_prune_per_layer = {}
        for (layer_n, f, _) in filters_to_prune:
            if layer_n not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[layer_n] = []
            filters_to_prune_per_layer[layer_n].append(f)
        
        # Adjust indices to account for previously pruned filters
        for layer_n in filters_to_prune_per_layer:
            filters_to_prune_per_layer[layer_n] = sorted(filters_to_prune_per_layer[layer_n])
            for i in range(len(filters_to_prune_per_layer[layer_n])):
                filters_to_prune_per_layer[layer_n][i] = filters_to_prune_per_layer[layer_n][i] - i
        
        # Flatten the pruning plan
        filters_to_prune = []
        for layer_n in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[layer_n]:
                filters_to_prune.append((layer_n, i))
        
        return filters_to_prune
    
    def get_layer_importance_distribution(self):
        """Return importance distribution across layers for analysis."""
        layer_importance = {}
        for activation_idx, ranks in self.filter_ranks.items():
            layer_idx = self.activation_to_layer[activation_idx]
            importance = ranks.mean().item()
            layer_importance[layer_idx] = importance
        
        return layer_importance