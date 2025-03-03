from heapq import nsmallest
from operator import itemgetter
import torch
class FilterPruner:
    def __init__(self, model, taylor=1):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.taylor = taylor
        self.reset()
        
    def reset(self):
        self.filter_ranks = {}
        self.activations = []
        self.gradients = []
        self.activation_to_layer = {}
        self.grad_index = 0
        # Initialize dictionaries for 2nd and 3rd order Taylor scores
        self.filter_ranks_2nd = {}
        self.filter_ranks_3rd = {}
        
    def forward(self, x):
        self.activations = []
        self.grad_index = 0
        self.model.eval()
        self.model.zero_grad()
        
        activation_index = 0
        for layer_index, layer in enumerate(self.model.features):   
            x = layer(x)
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer_index
                if self.taylor == 1:
                    x.register_hook(lambda grad, idx=activation_index: self.compute_rank(grad, idx))
                elif self.taylor == 2:
                    x.register_hook(lambda grad, idx=activation_index: self.compute_rank_2nd_order(grad, idx))
                else:
                    x.register_hook(lambda grad, idx=activation_index: self.compute_rank_3rd_order(grad, idx))
                activation_index += 1
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)
        return x
    
    def compute_rank(self, grad, activation_index):
        """Compute rank of the filters using Taylor expansion.
        
        Args:
            grad: The gradient of the criterion with respect to the output of the layer
            activation_index: The index of the activation in self.activations
        """
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
        self.filter_ranks[activation_index] += taylor

    def compute_rank_2nd_order(self, grad, activation_index):
        # Compute 2nd order Taylor score: 0.5 * (activation^2 * grad^2)
        activation = self.activations[activation_index]
        second_order = 0.5 * (activation ** 2 * grad ** 2).mean(dim=(0, 2, 3)).data
        if activation_index not in self.filter_ranks_2nd:
            self.filter_ranks_2nd[activation_index] = torch.zeros(activation.size(1), device=self.device)
        self.filter_ranks_2nd[activation_index] += second_order

    def compute_rank_3rd_order(self, grad, activation_index):
        # Compute 3rd order Taylor score: (1/6) * (activation^3 * grad^3)
        activation = self.activations[activation_index]
        third_order = (1.0/6.0) * (activation ** 3 * grad ** 3).mean(dim=(0, 2, 3)).data
        if activation_index not in self.filter_ranks_3rd:
            self.filter_ranks_3rd[activation_index] = torch.zeros(activation.size(1), device=self.device)
        self.filter_ranks_3rd[activation_index] += third_order
        
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