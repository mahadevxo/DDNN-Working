from heapq import nsmallest
from operator import itemgetter
import torch
import gc
import numpy as np

class FilterPruner:
    def __init__(self, model):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.reset()
        
    def reset(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        self.filter_ranks = {}          # initialize ranking storage
        
    def forward(self, x):
        self.activations = []
        self.activation_to_layer = {}
        self.grad_index = 0
        
        self.model.eval()
        self.model.zero_grad()
        
        conv_count = 0
        for module in self.model.net_1:
            x = module(x)
            if isinstance(module, torch.nn.Conv2d):
                idx = len(self.activations)
                x.register_hook(lambda grad, idx=idx: self.compute_rank(grad, idx))
                self.activations.append(x)
                self.activation_to_layer[idx] = conv_count
                conv_count += 1
        
        return self.model.net_2(x.view(x.size(0), -1))
    
    def compute_rank(self, grad, activation_index):
        # Safety check to ensure activation_index is valid
        if activation_index >= len(self.activations) or activation_index < 0:
            print(f"Warning: Invalid activation_index {activation_index}, max is {len(self.activations)-1}")
            return
            
        activation = self.activations[activation_index]
        
        # Compute Taylor criterion
        taylor = activation * grad
        
        # Average across batch and spatial dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data
        
        # Initialize filter_ranks for this activation if not already done
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to(self.device)
            
        # Update the ranks
        self.filter_ranks[activation_index] += taylor.to(self.device)  # Ensure device consistency
        
    def lowest_ranking_filters(self, num, filter_ranks):
        data = []
        for i in sorted(filter_ranks.keys()):
            for j in range(filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, filter_ranks[i][j]))
        return nsmallest(num, data, itemgetter(2))
    
    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v/(np.sqrt(torch.sum(v**2).cpu().numpy()) + 1e-10)
            self.filter_ranks[i] = v
        return self.filter_ranks      # return normalized scores
            
    def get_pruning_plan(self, num_filters_to_prune: int, filter_ranks = None):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune, filter_ranks=filter_ranks)
        x = [(layer_n, f, float(amt)) for layer_n, f, amt in filters_to_prune]

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
        
        return x, filters_to_prune
    
    def get_sorted_filters(self, filter_ranks):
        data = []
        for i in sorted(filter_ranks.keys()):
            for j in range(filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, filter_ranks[i][j]))
        return sorted(data, key=lambda x: x[2])