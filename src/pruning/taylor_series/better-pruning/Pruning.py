import torch
import gc

class Pruning:
    def __init__(self, model):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.layer_info = {}
        self._precompute_layer_info(model)
    
    def _clear_memory(self):
        """
        Clears GPU/MPS memory by forcing garbage collection and emptying caches.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def _precompute_layer_info(self, model):
        """
        Cache layer relationships to avoid repeated model traversal.
        
        Args:
            model: The model to analyze
            
        Returns:
            None, but populates self.layer_info with cached relationship data
        """
        self.layer_info = {}
        modules = list(model.net_1)

        for i, module in enumerate(modules):
            if isinstance(module, torch.nn.Conv2d):
                next_conv_index = next(
                    (
                        j
                        for j in range(i + 1, len(modules))
                        if isinstance(modules[j], torch.nn.Conv2d)
                    ),
                    None,
                )
                is_last_conv = not any(
                    isinstance(modules[j], torch.nn.Conv2d)
                    for j in range(i + 1, len(modules))
                )
                self.layer_info[i] = {
                    "next_conv_index": next_conv_index,
                    "is_last_conv": is_last_conv
                }
    
    def _replace_layers(self, model, i, indices, layers):
        """
        Replaces specific layers in a sequential model.
        """
        return layers[indices.index(i)] if i in indices else model[i]
    
    def _get_next_conv(self, model, layer_index):
        """
        Get next conv layer using cached information instead of traversing.
        """
        if layer_index in self.layer_info and self.layer_info[layer_index]["next_conv_index"] is not None:
            modules = list(model.net_1)
            return modules[self.layer_info[layer_index]["next_conv_index"]]
        return None
    
    def _get_next_conv_offset(self, model, layer_index):
        """
        Get next conv layer offset using cached information.
        """
        if layer_index in self.layer_info and self.layer_info[layer_index]["next_conv_index"] is not None:
            return self.layer_info[layer_index]["next_conv_index"] - layer_index
        return 0
    
    def _create_new_conv(self, conv, in_channels=None, out_channels=None):
        """
        Creates a new convolutional layer with specified dimensions.
        """
        if in_channels is None:
            in_channels = conv.in_channels

        if out_channels is None:
            out_channels = conv.out_channels

        return torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
        ).to(self.device)

    def _prune_conv_layer(self, conv, new_conv, filter_indices):
        """
        Transfers weights from original conv layer to new layer, removing specified filters.
        
        Args:
            conv: Original convolutional layer
            new_conv: New convolutional layer with fewer filters
            filter_indices: List of indices of filters to remove
            
        Returns:
            None, but modifies new_conv in-place
        """
        # Bounds checking
        max_idx = conv.weight.data.size(0) - 1
        filter_indices = [idx for idx in filter_indices if idx <= max_idx]

        # Sort filter indices in descending order to avoid shifting problems
        filter_indices = sorted(filter_indices, reverse=True)

        # Ensure we don't prune too many filters
        if len(filter_indices) >= conv.out_channels:
            filter_indices = filter_indices[:conv.out_channels-1]

        source_to_target = self._map_indices(
            conv, 0, filter_indices
        )
        with torch.no_grad():
            # Copy weights for filters not being pruned
            for source_idx, target_idx in source_to_target.items():
                new_conv.weight.data[target_idx] = conv.weight.data[source_idx]

                # Copy biases if present
                if conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.data[target_idx] = conv.bias.data[source_idx]

    def _prune_next_conv_layer(self, next_conv, new_next_conv, filter_indices):
        """
        Adjusts the next convolutional layer to account for removed input channels.
        
        Args:
            next_conv: Original next convolutional layer
            new_next_conv: New next convolutional layer with fewer input channels
            filter_indices: List of indices of filters to remove
            
        Returns:
            None, but modifies new_next_conv in-place
        """
        # Sort filter indices in descending order to avoid shifting problems
        filter_indices = sorted(filter_indices, reverse=True)

        source_to_target = self._map_indices(
            next_conv, 1, filter_indices
        )
        with torch.no_grad():
            # Copy weights for all output filters, skipping pruned input channels
            for out_idx in range(next_conv.weight.data.size(0)):
                for source_in_idx, target_in_idx in source_to_target.items():
                    new_next_conv.weight.data[out_idx, target_in_idx] = next_conv.weight.data[out_idx, source_in_idx]

            # Copy biases directly (not affected by input channel pruning)
            if next_conv.bias is not None and new_next_conv.bias is not None:
                new_next_conv.bias.data = next_conv.bias.data

    def _map_indices(self, arg0, arg1, filter_indices):
        result = {}
        target_idx = 0
        for source_idx in range(arg0.weight.data.size(arg1)):
            if source_idx not in filter_indices:
                result[source_idx] = target_idx
                target_idx += 1
        return result

    def _prune_last_conv_layer(self, model, conv, new_conv, layer_index, filter_indices):
        """
        Prunes the last convolutional layer and adjusts the first fully connected layer.
        
        Args:
            model: The model being pruned
            conv: Original last convolutional layer
            new_conv: New convolutional layer with fewer filters
            layer_index: Index of the layer in the model
            filter_indices: List of indices of filters to remove
            
        Returns:
            Modified model with updated layers
        """
        # Sort filter indices in descending order
        filter_indices = sorted(filter_indices, reverse=True)
        
        # First, prune the conv layer
        self._prune_conv_layer(conv, new_conv, filter_indices)
        
        # Replace conv layer in model
        modules = list(model.net_1)
        modules[layer_index] = new_conv
        model.net_1 = torch.nn.Sequential(*modules)
        
        # Find the first fully connected layer
        layer_index = 0
        old_linear_layer = None
        for _, module in model.net_2._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index += 1
        
        if old_linear_layer is None:
            raise ValueError(f"No linear layer found in classifier, Model: {model}")
        
        # Calculate parameters per input channel
        params_per_input_channel = old_linear_layer.in_features // conv.out_channels
        
        # Create new linear layer with reduced input size
        new_linear_layer = torch.nn.Linear(
            old_linear_layer.in_features - (len(filter_indices) * params_per_input_channel),
            old_linear_layer.out_features
        ).to(self.device)
        
        # Compute mapping from old to new indices
        with torch.no_grad():
            target_idx = 0
            for source_idx in range(conv.out_channels):
                if source_idx not in filter_indices:
                    # Copy weights for this feature map
                    start_old = source_idx * params_per_input_channel
                    end_old = (source_idx + 1) * params_per_input_channel
                    
                    start_new = target_idx * params_per_input_channel
                    end_new = (target_idx + 1) * params_per_input_channel
                    
                    new_linear_layer.weight.data[:, start_new:end_new] = old_linear_layer.weight.data[:, start_old:end_old]
                    target_idx += 1
            
            # Copy biases directly
            new_linear_layer.bias.data = old_linear_layer.bias.data
        
        # Update the classifier
        modules = list(model.net_2)
        modules[layer_index] = new_linear_layer
        model.net_2 = torch.nn.Sequential(*modules)
        
        return model

    def batch_prune_filters(self, model, prune_targets):
        """
        Prune multiple filters at once per layer to reduce reconstruction.
        """
        # Group filters by layer and remove duplicates
        filters_by_layer = {}
        for layer_idx, filter_idx in prune_targets:
            if layer_idx not in filters_by_layer:
                filters_by_layer[layer_idx] = set()  # Use a set to avoid duplicates
            filters_by_layer[layer_idx].add(filter_idx)
        
        # Process layers in reverse order to avoid index shifting issues
        for layer_idx in sorted(filters_by_layer.keys(), reverse=True):
            filter_indices = sorted(list(filters_by_layer[layer_idx]), reverse=True)  # Convert to sorted list
            modules = list(model.net_1)
            
            # Safety check: ensure we don't prune too many filters
            if layer_idx >= len(modules):
                print(f"Warning: Layer index {layer_idx} out of bounds, skipping")
                continue
                
            conv = modules[layer_idx]
            if not isinstance(conv, torch.nn.Conv2d):
                print(f"Warning: Layer {layer_idx} is not Conv2d, skipping")
                continue
                
            # Safety check: ensure we don't prune all filters
            if len(filter_indices) >= conv.out_channels:
                print(f"Warning: Trying to prune {len(filter_indices)} filters from layer {layer_idx} with only {conv.out_channels} filters")
                # Limit pruning to leave at least one filter
                filter_indices = filter_indices[:conv.out_channels-1]
                
            # Verify indices are in bounds
            valid_indices = [idx for idx in filter_indices if idx < conv.out_channels]
            if len(valid_indices) < len(filter_indices):
                print(f"Warning: Some filter indices for layer {layer_idx} are out of bounds. Max index should be < {conv.out_channels}")
                filter_indices = valid_indices
                
            # Create new conv with reduced output channels
            new_conv = self._create_new_conv(
                conv=conv, 
                out_channels=conv.out_channels - len(filter_indices)
            )
            
            # Check if this is the last conv layer
            is_last_conv = self.layer_info[layer_idx]["is_last_conv"] if layer_idx in self.layer_info else False
            
            if is_last_conv:
                # Handle last conv layer differently due to FC layer connection
                model = self._prune_last_conv_layer(
                    model=model,
                    conv=conv,
                    new_conv=new_conv,
                    layer_index=layer_idx,
                    filter_indices=filter_indices
                )
            else:
                # Regular conv layer pruning
                self._prune_conv_layer(conv, new_conv, filter_indices)
                
                # Update model with new conv layer
                modules[layer_idx] = new_conv
                model.net_1 = torch.nn.Sequential(*modules)
                
                # Get next conv layer that needs input channel adjustment
                next_conv_idx = self.layer_info[layer_idx]["next_conv_index"] if layer_idx in self.layer_info else None
                if next_conv_idx is not None:
                    next_conv = modules[next_conv_idx]
                    new_next_conv = self._create_new_conv(
                        conv=next_conv,
                        in_channels=next_conv.in_channels - len(filter_indices)
                    )
                    self._prune_next_conv_layer(next_conv, new_next_conv, filter_indices)
                    
                    # Update model with new next_conv layer
                    modules[next_conv_idx] = new_next_conv
                    model.net_1 = torch.nn.Sequential(*modules)
            
            # Selective memory clearing (not every iteration)
            if len(filter_indices) > 10:
                self._clear_memory()
                
        # Final memory cleanup
        self._clear_memory()
        return model

    def prune_vgg_conv_layer(self, model, layer_index, filter_index):
        """
        Legacy method for single filter pruning - redirects to batch pruning for efficiency
        """
        return self.batch_prune_filters(model, [(layer_index, filter_index)])