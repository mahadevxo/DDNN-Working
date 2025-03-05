from heapq import nsmallest
from operator import itemgetter
import torch
import copy

class FilterPruner:
    def __init__(self, model):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.reset()
        
    def reset(self):
        self.filter_ranks = {}
        self.activations = []
        self.activation_to_layer = {}
        
    # New approach: Split the forward pass into two steps to avoid hook conflicts
    def forward(self, x):
        """
        Forward pass that captures activations without using hooks.
        """
        self.activations = []
        self.model.eval()
        
        # Store all activations during forward pass
        activation_index = 0
        layer_activations = {}
        
        # Get a handle to the original input for gradient computation
        x_clone = x.clone().detach().requires_grad_(True)
        current_input = x_clone
        
        # First, run through model and store activations
        for layer_index, layer in enumerate(self.model.features):
            current_input = layer(current_input)
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                # Store activation separately - ensure it's a leaf tensor that requires grad
                activation = current_input.clone().detach().requires_grad_(True)
                # Explicitly tell PyTorch to retain gradients for this tensor
                activation.retain_grad()
                layer_activations[activation_index] = (layer_index, activation)
                self.activation_to_layer[activation_index] = layer_index
                activation_index += 1
        
        # Complete forward pass - need to rebuild from the last activation
        if layer_activations:
            # Get the last activation
            last_act_idx = max(layer_activations.keys())
            _, last_activation = layer_activations[last_act_idx]
            # Complete the forward pass from this activation
            current_input = last_activation
        
        output = current_input.view(current_input.size(0), -1)
        output = self.model.classifier(output)
        
        # Store for later use in compute_ranks
        self.stored_x = x_clone
        self.stored_output = output
        self.layer_activations = layer_activations
        
        return output
        
    def compute_ranks(self, y):
        """
        Compute Taylor ranks based on output y (usually the loss).
        Must be called after forward().
        """
        # Get gradient of output with respect to input
        self.model.zero_grad()
        y.backward(retain_graph=True)
        
        # Now calculate Taylor criterion for each activation
        for act_idx, (layer_idx, activation) in self.layer_activations.items():
            # Check if the gradient was computed for this activation
            if activation.grad is not None:
                # Compute Taylor criterion
                taylor = activation.grad.data * activation.data
                taylor = taylor.mean(dim=(0, 2, 3)).abs()
                
                # Initialize filter_ranks for this activation if not already done
                if act_idx not in self.filter_ranks:
                    self.filter_ranks[act_idx] = torch.zeros(activation.size(1), device=self.device)
                
                # Update ranks
                self.filter_ranks[act_idx] += taylor
            else:
                print(f"Warning: No gradient for activation {act_idx}")
        
        # Add activations for access in other methods
        self.activations = [act for _, act in self.layer_activations.values()]
    
    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
        return nsmallest(num, data, itemgetter(2))
    
    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / torch.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()
            
    def get_pruning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
        
        filters_to_prune_per_layer = {}
        for (layer_n, f, _) in filters_to_prune:
            if layer_n not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[layer_n] = []
            filters_to_prune_per_layer[layer_n].append(f)
        
        for layer_n in filters_to_prune_per_layer:
            filters_to_prune_per_layer[layer_n] = sorted(filters_to_prune_per_layer[layer_n])
            for i in range(len(filters_to_prune_per_layer[layer_n])):
                filters_to_prune_per_layer[layer_n][i] = filters_to_prune_per_layer[layer_n][i] - i
        
        filters_to_prune = []
        for layer_n in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[layer_n]:
                filters_to_prune.append((layer_n, i))
        
        return filters_to_prune