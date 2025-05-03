from heapq import nsmallest
from operator import itemgetter
import torch
import gc

class FilterPruner:
    def __init__(self, model):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.reset()
        
    def reset(self):
        """
        Resets the pruner's state and clears memory.
        
        Clears data structures that store filter rankings, activations, and gradients
        to prepare for a new pruning iteration and prevent memory leaks.
        
        Args:
            None
            
        Returns:
            None
        """
        # Clear previous data structures to prevent memory leaks
        if hasattr(self, 'filter_ranks'):
            for key in list(self.filter_ranks.keys()):
                del self.filter_ranks[key]
        if hasattr(self, 'activations'):
            for act in self.activations:
                del act
            
        self.filter_ranks = {}
        self.activations = []
        self.gradients = []
        self.activation_to_layer = {}
        self.grad_index = 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
    def forward(self, x):
        """
        Forward pass through the model with hooks to capture activations and gradients.
        
        Processes the input through each layer of the model, registering hooks on
        convolutional layer outputs to capture information needed for Taylor ranking.
        
        Args:
            x: Input tensor (batch of images)
            
        Returns:
            Model output tensor
        """
        self.activations = []
        self.grad_index = 0
        self.model.eval()
        self.model.zero_grad()
        
        activation_index = 0
        for layer_index, layer in enumerate(self.model.net_1):   
            x = layer(x)
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer_index
                x.register_hook(lambda grad, idx=activation_index: self.compute_rank(grad, idx))
                activation_index += 1
        x = x.view(x.size(0), -1)
        x = self.model.net_2(x)
        return x
    
    def compute_rank(self, grad, activation_index):
        """
        Computes filter importance rankings using Taylor expansion.
        
        Calculates the importance of each filter in a layer using the product of
        activations and gradients (first-order Taylor expansion), which approximates
        the effect of removing that filter on the loss function.
        
        Args:
            grad: Gradient tensor flowing back through the layer
            activation_index: Index of the activation in the activations list
            
        Returns:
            None, but updates the filter_ranks dictionary
        """
        # Safety check to ensure activation_index is valid
        if activation_index >= len(self.activations) or activation_index < 0:
            print(f"Warning: Invalid activation_index {activation_index}, max is {len(self.activations)-1}")
            return
            
        activation = self.activations[activation_index]
        
        # Compute Taylor criterion with L1 norm
        taylor = torch.abs(activation * grad)
        
        # Average across batch and spatial dimensions with better numerical stability
        taylor = taylor.mean(dim=(0, 2, 3)).data
        
        # Initialize filter_ranks for this activation if not already done
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to(self.device)
            
        # Update the ranks
        self.filter_ranks[activation_index] += taylor
        del taylor, activation, grad
        
        
    def lowest_ranking_filters(self, num):
        """
        Identifies the least important filters based on computed rankings.
        
        Sorts all filters across all layers by their importance scores and
        returns the specified number of least important filters.
        
        Args:
            num: Number of filters to return
            
        Returns:
            List of (layer_index, filter_index, rank) tuples for the least important filters
        """
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
        return nsmallest(num, data, itemgetter(2))
    
    def normalize_ranks_per_layer(self):
        """
        Normalizes filter ranks using percentile ranking.
        
        Converts raw importance scores to percentile ranks (0-1) to make
        rankings comparable across different layers and to reduce the impact
        of outliers.
        
        Args:
            None
            
        Returns:
            None, but updates the filter_ranks dictionary
        """
        if not self.filter_ranks:
            return

        # Collect all ranks into a flat array
        all_ranks = []
        rank_indices = {}

        # Create a flat list with identifiers
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                all_ranks.append(self.filter_ranks[i][j].item())
                rank_indices[(i, j)] = len(all_ranks) - 1

        # Sort for percentile calculation
        all_ranks_sorted = sorted(all_ranks)
        n = len(all_ranks_sorted)

        rank_to_percentile = {
            val: idx / (n - 1) if n > 1 else 0.5
            for idx, val in enumerate(all_ranks_sorted)
        }
        # Convert each filter's rank to its percentile
        for i in self.filter_ranks:
            for j in range(self.filter_ranks[i].size(0)):
                val = self.filter_ranks[i][j].item()
                # Find closest value (handle floating point precision issues)
                closest_val = min(all_ranks_sorted, key=lambda x: abs(x - val))
                percentile = rank_to_percentile[closest_val]
                self.filter_ranks[i][j] = torch.tensor(percentile)

            # Move to CPU to save memory
            self.filter_ranks[i] = self.filter_ranks[i].cpu()
    
    def smooth_distributions(self):
        """
        Applies small random noise to filter ranks to prevent ties.
        
        Adds small random perturbations to importance scores to break ties
        and prevent clustered rankings, which helps make pruning decisions
        more deterministic.
        
        Args:
            None
            
        Returns:
            None, but updates the filter_ranks dictionary
        """
        for i in self.filter_ranks:
            # Add small random noise to break ties and clusters
            noise = torch.randn_like(self.filter_ranks[i]) * 1e-5
            self.filter_ranks[i] += noise
            
    def get_pruning_plan(self, num_filters_to_prune):
        """
        Generates a plan for which filters to prune across the model.
        
        Identifies the least important filters and adjusts filter indices to
        account for previous pruning within the same layer.
        
        Args:
            num_filters_to_prune: Total number of filters to select for pruning
            
        Returns:
            List of (layer_index, filter_index) tuples identifying filters to prune
        """
    # Apply smoothing before ranking
        self.smooth_distributions()
        
        # Apply normalization
        self.normalize_ranks_per_layer()
        
        # Get the lowest ranking filters
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
        
        # Clean up after pruning plan is created
        self.reset()
        
        return filters_to_prune