from heapq import nsmallest
from operator import itemgetter
import torch
import gc
import numpy as np

class FilterPruner:
    def __init__(self, model, rank_type='taylor'):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.rank_type = rank_type  # 'taylor', 'l1_norm', 'apoz', 'fisher', 'combined'
        self.reset()
        
    def reset(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        self.filter_ranks = {}          # initialize ranking storage
        
        # For activation-based metrics
        self.activation_count = {}
        self.activation_stats = {}
        
        # For Fisher information
        self.fisher_info = {}
        
    def forward(self, x):
        # Release any previous activations to free memory
        if hasattr(self, 'activations'):
            del self.activations
            
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
                
                # Store activation statistics for other metrics
                if self.rank_type in ['apoz', 'combined']:
                    self._update_activation_statistics(x, idx)
                    
                self.activations.append(x)
                self.activation_to_layer[idx] = conv_count
                conv_count += 1
        
        # For interface-aware pruning - store the final output features
        self.net_1_output = x
        
        return self.model.net_2(x.view(x.size(0), -1))
    
    def _update_activation_statistics(self, activation, idx):
        """Update running statistics for activation-based metrics"""
        # Initialize if needed
        if idx not in self.activation_stats:
            self.activation_stats[idx] = {
                'mean': torch.zeros(activation.size(1)).to(self.device),
                'squared_mean': torch.zeros(activation.size(1)).to(self.device),
                'apoz': torch.zeros(activation.size(1)).to(self.device)
            }
            self.activation_count[idx] = 0
            
        # Update counts
        batch_size = activation.size(0)
        self.activation_count[idx] += batch_size
        
        # Compute batch statistics
        batch_mean = activation.mean(dim=(0, 2, 3)).data
        batch_squared_mean = (activation ** 2).mean(dim=(0, 2, 3)).data
        
        # APoZ (Average Percentage of Zeros)
        # Higher APoZ means the filter activates less often (more zeros)
        batch_apoz = (activation <= 0).float().mean(dim=(0, 2, 3)).data
        
        # Update running statistics with exponential moving average
        momentum = 0.1
        self.activation_stats[idx]['mean'] = (1 - momentum) * self.activation_stats[idx]['mean'] + momentum * batch_mean
        self.activation_stats[idx]['squared_mean'] = (1 - momentum) * self.activation_stats[idx]['squared_mean'] + momentum * batch_squared_mean
        self.activation_stats[idx]['apoz'] = (1 - momentum) * self.activation_stats[idx]['apoz'] + momentum * batch_apoz
    
    def compute_rank(self, grad, activation_index):
        # Safety check to ensure activation_index is valid
        if activation_index >= len(self.activations) or activation_index < 0:
            print(f"Warning: Invalid activation_index {activation_index}, max is {len(self.activations)-1}")
            return
            
        activation = self.activations[activation_index]
        
        # Additional safety check for gradient shape matching activation
        if grad.shape != activation.shape:
            print(f"Warning: Gradient shape {grad.shape} does not match activation shape {activation.shape}")
            return
        
        # Initialize filter_ranks for this activation if not already done
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to(self.device)
        
        # Compute different ranking criteria based on rank_type
        if self.rank_type == 'taylor' or self.rank_type == 'combined':
            # Taylor criterion (1st order approximation of output change)
            taylor = activation * grad
            taylor = torch.abs(taylor.mean(dim=(0, 2, 3)).data)
            self.filter_ranks[activation_index] += taylor
            
        if self.rank_type == 'fisher' or self.rank_type == 'combined':
            # Fisher information (approximated by squared gradient)
            fisher = grad ** 2
            fisher = fisher.mean(dim=(0, 2, 3)).data
            
            # Initialize or update Fisher information
            if activation_index not in self.fisher_info:
                self.fisher_info[activation_index] = torch.zeros_like(fisher)
            self.fisher_info[activation_index] += fisher
        
        if self.rank_type == 'l1_norm' or self.rank_type == 'combined':
            # L1-norm of the filter weights
            module_name = f"net_1.{activation_index*2}"  # Approximation - adjust based on your model structure
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d) and name.startswith(module_name):
                    l1_norm = torch.norm(module.weight.data, p=1, dim=(1, 2, 3))
                    
                    # Normalize by filter size
                    filter_size = module.weight.size(1) * module.weight.size(2) * module.weight.size(3)
                    l1_norm = l1_norm / filter_size
                    
                    # Update the ranks with L1 norm (lower is better for pruning)
                    self.filter_ranks[activation_index] += l1_norm
                    break
    
    def lowest_ranking_filters(self, num, filter_ranks):
        data = []
        for i in sorted(filter_ranks.keys()):
            for j in range(filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, filter_ranks[i][j]))
        return nsmallest(num, data, itemgetter(2))
    
    def normalize_ranks_per_layer(self):
        """Normalize ranks within each layer and combine different metrics if using 'combined'"""
        for i in self.filter_ranks:
            # Start with Taylor criterion or whatever is in filter_ranks
            ranks = torch.abs(self.filter_ranks[i])
            
            # Add other metrics if using combined approach
            if self.rank_type == 'combined':
                # Add APoZ (more zeros = higher value = more likely to prune)
                if i in self.activation_stats:
                    apoz = self.activation_stats[i]['apoz']
                    # Normalize APoZ
                    apoz = apoz / (torch.max(apoz) + 1e-10)
                    ranks = ranks + apoz
                
                # Add Fisher Information (lower Fisher = higher value for pruning)
                if i in self.fisher_info:
                    fisher = self.fisher_info[i]
                    # Invert and normalize Fisher (we want to prune filters with LESS information)
                    max_fisher = torch.max(fisher) + 1e-10
                    inv_fisher = (max_fisher - fisher) / max_fisher
                    ranks = ranks + inv_fisher
            
            # For L1-norm criterion, we directly use the L1-norm without additional computation
            # For APoZ only criterion
            elif self.rank_type == 'apoz' and i in self.activation_stats:
                ranks = self.activation_stats[i]['apoz']
            
            # Normalize the final ranks
            if torch.sum(ranks) > 0:
                v = ranks / (torch.sqrt(torch.sum(ranks**2)) + 1e-10)
                self.filter_ranks[i] = v
            else:
                self.filter_ranks[i] = ranks
        
        # If we're pruning net_1 for a split model, we should favor keeping filters
        # that contribute most to the output that net_2 needs
        if hasattr(self, 'net_1_output'):
            # Not implemented yet - would require analyzing net_2's first layer gradients
            pass
        
        return self.filter_ranks
            
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