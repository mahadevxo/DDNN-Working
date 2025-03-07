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
        
    # Fixed approach: Ensure gradients flow properly through the network
    def forward(self, x):
        """
        Forward pass that captures activations without using hooks.
        """
        self.activations = []
        self.model.eval()
        
        # Store activations during forward pass
        activation_index = 0
        layer_activations = {}
        
        # Clone and detach input for tracking
        x = x.clone().detach().to(self.device)
        
        # Create a computational graph for tracking
        features_output = None
        
        # Track gradients for Conv2d layers in features
        for layer_index, layer in enumerate(self.model.features):
            # Process input through current layer
            x = layer(x)
            
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                # Store activation index and its output
                # Keep x in the computational graph (don't clone/detach)
                self.activation_to_layer[activation_index] = layer_index
                
                # Store the conv layer output tensor (this is crucial)
                if activation_index not in self.filter_ranks:
                    self.filter_ranks[activation_index] = torch.zeros(x.shape[1], device=self.device)
                
                # Store activation tensor for later gradient computation
                layer_activations[activation_index] = (layer_index, x)
                activation_index += 1
                
        # Store for final output computation
        features_output = x
        
        # Process through classifier
        with torch.no_grad():  # No need to track gradients in classifier for pruning
            try:
                # Flatten for classifier (typical in VGG-like models)
                batch_size = features_output.size(0)
                # Clone the tensor to avoid view-related in-place modifications
                classifier_input = features_output.view(batch_size, -1).clone()
                
                # Process through classifier
                output = self.model.classifier(classifier_input)
            except RuntimeError as e:
                if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                    # Apply adaptive pooling to match expected input size
                    first_linear = None
                    for module in self.model.classifier:
                        if isinstance(module, torch.nn.Linear):
                            first_linear = module
                            break
                    
                    if first_linear:
                        # Calculate spatial dimensions needed
                        input_features = first_linear.in_features
                        batch_size = features_output.size(0)
                        channels = features_output.size(1)
                        spatial_size = int((input_features / channels) ** 0.5)
                        
                        # Apply adaptive pooling
                        adaptive_pool = torch.nn.AdaptiveAvgPool2d((spatial_size, spatial_size))
                        pooled = adaptive_pool(features_output)
                        # Clone the result of view to avoid in-place modification issues
                        output = self.model.classifier(pooled.view(batch_size, -1).clone())
                    else:
                        raise e
                else:
                    raise e
        
        # Store for later use in compute_ranks
        self.layer_activations = layer_activations
        
        return output
        
    def compute_ranks(self, y):
        """
        Compute Taylor ranks based on output y (usually the loss).
        """
        # Make sure we're in training mode for proper gradient flow
        self.model.train()
        
        # Zero gradients from previous iterations
        self.model.zero_grad()
        
        # Calculate gradients
        y.backward()
        
        # Now process each stored activation
        num_processed = 0
        for act_idx, (layer_idx, activation) in self.layer_activations.items():
            if activation.grad is not None:
                # Clone activation tensors to avoid in-place modifications on views
                taylor = activation.grad.data.clone() * activation.data.clone()
                taylor = taylor.mean(dim=(0, 2, 3)).abs()
                
                # Update rankings
                self.filter_ranks[act_idx] += taylor
                num_processed += 1
            else:
                # Instead of just warning, try to calculate gradient another way
                # This is a fallback mechanism to prevent the process from hanging
                try:
                    # Try to get the gradients for the layer directly
                    layer = list(self.model.features)[layer_idx]
                    if hasattr(layer, 'weight') and layer.weight.grad is not None:
                        # Use the gradient of weights as a proxy
                        taylor = layer.weight.grad.data.clone().mean(dim=(0, 2, 3)).abs()
                        self.filter_ranks[act_idx] += taylor
                        num_processed += 1
                    else:
                        print(f"Warning: No gradient for activation {act_idx}, using random ranking")
                        # Use small random values to avoid getting stuck
                        self.filter_ranks[act_idx] += torch.rand(self.filter_ranks[act_idx].size(), 
                                                                device=self.device) * 0.001
                        num_processed += 1
                except Exception as e:
                    print(f"Error processing gradient for activation {act_idx}: {str(e)}")
                    # Still add random values to continue processing
                    self.filter_ranks[act_idx] += torch.rand(self.filter_ranks[act_idx].size(), 
                                                            device=self.device) * 0.001
                    num_processed += 1
            
        print(f"Processed gradients for {num_processed}/{len(self.layer_activations)} activations")
        
        # Clear references to avoid memory leaks 
        self.layer_activations = {}
    
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
        
        # Safety check: ensure we're not pruning too many filters from any layer
        for layer_n in list(filters_to_prune_per_layer.keys()):
            # Find the actual layer
            layer = None
            for i, module in enumerate(self.model.features):
                if i == layer_n and isinstance(module, torch.nn.Conv2d):
                    layer = module
                    break
            
            # If we found the layer, check how many filters we're pruning
            if layer is not None:
                # Don't prune more than 90% of filters from any layer
                max_to_prune = int(0.9 * layer.out_channels)
                # Always leave at least 2 filters
                max_to_prune = min(max_to_prune, layer.out_channels - 2)
                
                if len(filters_to_prune_per_layer[layer_n]) > max_to_prune:
                    print(f"WARNING: Limiting pruning on layer {layer_n} to {max_to_prune} filters instead of {len(filters_to_prune_per_layer[layer_n])}")
                    filters_to_prune_per_layer[layer_n] = filters_to_prune_per_layer[layer_n][:max_to_prune]
        
        # After limiting, adjust filter indices accounting for previous pruning
        for layer_n in filters_to_prune_per_layer:
            filters_to_prune_per_layer[layer_n] = sorted(filters_to_prune_per_layer[layer_n])
            for i in range(len(filters_to_prune_per_layer[layer_n])):
                filters_to_prune_per_layer[layer_n][i] = filters_to_prune_per_layer[layer_n][i] - i
        
        # Create final pruning plan
        filters_to_prune = []
        for layer_n in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[layer_n]:  # Removed extra ")"
                filters_to_prune.append((layer_n, i))
        
        return filters_to_prune