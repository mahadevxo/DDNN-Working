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
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
        return nsmallest(num, data, itemgetter(2))
    
    def normalize_ranks_per_layer(self):
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
        for i in self.filter_ranks:
            # Add MASSIVE noise to create huge variation in rankings
            noise = torch.randn_like(self.filter_ranks[i]) * 0.1  # HUGE noise
            self.filter_ranks[i] += noise
            
    def get_pruning_plan(self, num_filters_to_prune):
        # Apply smoothing before ranking
        self.smooth_distributions()
        
        # Apply normalization
        self.normalize_ranks_per_layer()
        
        # Get the lowest ranking filters
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
        
        # If no filters can be pruned, return empty list
        if not filters_to_prune:
            print("Warning: No filters available for pruning.")
            return []
        
        # Group by layer for safety checking
        filters_to_prune_per_layer = {}
        for (layer_n, f, _) in filters_to_prune:
            if layer_n not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[layer_n] = []
            filters_to_prune_per_layer[layer_n].append(f)
        
        # BRUTALLY AGGRESSIVE PRUNING - remove almost ALL safety checks
        valid_filters_to_prune = []
        
        for layer_n in filters_to_prune_per_layer:
            # Find the total number of filters in this layer
            layer_found = False
            total_filters = 0
            for layer_idx, layer in enumerate(self.model.net_1):
                if layer_idx == layer_n and isinstance(layer, torch.nn.Conv2d):
                    total_filters = layer.out_channels
                    layer_found = True
                    break
            
            if not layer_found:
                print(f"Warning: Layer {layer_n} not found, skipping")
                continue
                
            # BRUTALLY AGGRESSIVE: Allow pruning up to 99% of filters per layer
            filters_for_this_layer = filters_to_prune_per_layer[layer_n]
            max_prunable = max(1, int(total_filters * 0.99))  # Keep only 1% minimum
            
            # DON'T limit the number - take whatever was requested
            # if len(filters_for_this_layer) > max_prunable:
            #     filters_for_this_layer = filters_for_this_layer[:max_prunable]
            
            print(f"Layer {layer_n}: BRUTALLY pruning {len(filters_for_this_layer)}/{total_filters} filters")
            
            # Add ALL requested filters to the pruning plan (remove safety)
            for f in sorted(filters_for_this_layer):
                valid_filters_to_prune.append((layer_n, f))
        
        print(f"BRUTAL Pruning plan: {len(valid_filters_to_prune)} filters requested: {num_filters_to_prune}")
        
        return valid_filters_to_prune