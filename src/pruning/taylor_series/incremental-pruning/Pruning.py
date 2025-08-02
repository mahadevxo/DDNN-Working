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
            
            # Ensure no NaN/inf values in the new layer
            if torch.isnan(new_conv.weight.data).any() or torch.isinf(new_conv.weight.data).any():
                print("Warning: NaN/inf detected in conv weights")
                # Reinitialize problematic weights
                torch.nn.init.kaiming_normal_(new_conv.weight.data, mode='fan_out', nonlinearity='relu')
                if new_conv.bias is not None:
                    torch.nn.init.zeros_(new_conv.bias.data)

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
                
            # Check for NaN/inf values
            if torch.isnan(new_next_conv.weight.data).any() or torch.isinf(new_next_conv.weight.data).any():
                print("Warning: NaN/inf detected in next conv weights")
                torch.nn.init.kaiming_normal_(new_next_conv.weight.data, mode='fan_out', nonlinearity='relu')
                if new_next_conv.bias is not None:
                    torch.nn.init.zeros_(new_next_conv.bias.data)

    def _map_indices(self, conv_layer, dimension, filter_indices):
        """
        Create mapping from old indices to new indices, skipping pruned filters.
        
        Args:
            conv_layer: The convolutional layer
            dimension: 0 for output channels, 1 for input channels  
            filter_indices: List of indices to be pruned
            
        Returns:
            Dictionary mapping old_index -> new_index
        """
        result = {}
        target_idx = 0
        total_size = conv_layer.weight.data.size(dimension)
        
        for source_idx in range(total_size):
            if source_idx not in filter_indices:
                result[source_idx] = target_idx
                target_idx += 1
        return result

    def _prune_last_conv_layer(self, model, conv, new_conv, layer_index, filter_indices):
        """
        Prunes the last convolutional layer and adjusts the first fully connected layer.
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
        fc_layer_index = 0
        old_linear_layer = None
        for _, module in model.net_2._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            fc_layer_index += 1
        
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
            
            # Initialize weights properly to prevent NaN
            # Clamp weights to reasonable range
            new_linear_layer.weight.data = torch.clamp(new_linear_layer.weight.data, -1.0, 1.0)
            new_linear_layer.bias.data = torch.clamp(new_linear_layer.bias.data, -1.0, 1.0)
            
            # Check for NaN/inf values
            if torch.isnan(new_linear_layer.weight.data).any() or torch.isinf(new_linear_layer.weight.data).any():
                print("Warning: NaN/inf detected in FC weights, reinitializing")
                torch.nn.init.xavier_uniform_(new_linear_layer.weight.data)
                torch.nn.init.zeros_(new_linear_layer.bias.data)
        
        # Update the classifier
        fc_modules = list(model.net_2)
        fc_modules[fc_layer_index] = new_linear_layer
        model.net_2 = torch.nn.Sequential(*fc_modules)
        
        return model

    def batch_prune_filters(self, model, prune_targets):
        """
        Prune multiple filters at once per layer to reduce reconstruction.
        """
        if not prune_targets:
            print("Warning: No targets to prune")
            return model

        # Find the last Conv2d layer index
        last_conv_layer_idx = None
        for layer_idx, layer in enumerate(model.net_1):
            if isinstance(layer, torch.nn.Conv2d):
                last_conv_layer_idx = layer_idx

        # Group filters by layer and remove duplicates
        filters_by_layer = {}
        for layer_idx, filter_idx in prune_targets:
            # Skip last Conv2d layer to preserve FC layer compatibility
            if layer_idx == last_conv_layer_idx:
                print(f"Skipping filter pruning in last Conv2d layer {layer_idx} to preserve FC layer")
                continue
                
            if layer_idx not in filters_by_layer:
                filters_by_layer[layer_idx] = set()
            filters_by_layer[layer_idx].add(filter_idx)

        if not filters_by_layer:
            print("Warning: No valid layers to prune (last Conv2d layer excluded)")
            return model

        print(f": Pruning filters from {len(filters_by_layer)} layers (excluding last Conv2d)")

        # Process layers in order (not reverse) to avoid dependency issues
        for layer_idx in sorted(filters_by_layer.keys()):
            filter_indices = sorted(list(filters_by_layer[layer_idx]))
            modules = list(model.net_1)

            # Safety checks
            if layer_idx >= len(modules):
                print(f"Warning: Layer index {layer_idx} out of bounds, skipping")
                continue

            conv = modules[layer_idx]
            if not isinstance(conv, torch.nn.Conv2d):
                print(f"Warning: Layer {layer_idx} is not Conv2d, skipping")
                continue

            print(f" Layer {layer_idx}: Pruning {len(filter_indices)} filters from {conv.out_channels} total")

            # REMOVE MOST SAFETY CHECKS - only keep the absolute minimum
            if len(filter_indices) >= conv.out_channels:
                # Only leave 1 filter minimum instead of refusing
                filter_indices = filter_indices[:conv.out_channels-1]
                print(f": Limiting to {len(filter_indices)} filters to leave 1 remaining")

            # Verify indices are in bounds
            valid_indices = [idx for idx in filter_indices if 0 <= idx < conv.out_channels]
            if len(valid_indices) != len(filter_indices):
                print(f"Warning: Some filter indices for layer {layer_idx} are out of bounds")
                filter_indices = valid_indices

            if not filter_indices:  # No valid indices to prune
                continue

            # Create new conv with reduced output channels
            new_out_channels = conv.out_channels - len(filter_indices)
            if new_out_channels <= 0:
                print(f": Would result in {new_out_channels} filters for layer {layer_idx}, setting to 1")
                new_out_channels = 1
                filter_indices = filter_indices[:conv.out_channels-1]

            new_conv = self._create_new_conv(
                conv=conv, 
                out_channels=new_out_channels
            )

            # Prune the current layer
            self._prune_conv_layer(conv, new_conv, filter_indices)

            # Update model with new conv layer
            modules[layer_idx] = new_conv
            model.net_1 = torch.nn.Sequential(*modules)

            # Only update next conv layer if it's not the last conv layer
            next_conv_idx = self.layer_info.get(layer_idx, {}).get("next_conv_index")
            if next_conv_idx is not None and next_conv_idx != last_conv_layer_idx:
                model = self._update_next_conv_layer(model, next_conv_idx, filter_indices)

            print(f" Layer {layer_idx}: Successfully pruned to {new_conv.out_channels} filters")

        # Final memory cleanup
        self._clear_memory()
        return model

    def prune_vgg_conv_layer(self, model, layer_index, filter_index):
        """
        Legacy method for single filter pruning - redirects to batch pruning for efficiency
        """
        return self.batch_prune_filters(model, [(layer_index, filter_index)])

    def _update_next_conv_layer(self, model, next_conv_idx, filter_indices):
        """Update the next convolutional layer to handle reduced input channels."""
        modules = list(model.net_1)
        if next_conv_idx >= len(modules):
            return model
            
        next_conv = modules[next_conv_idx]
        if not isinstance(next_conv, torch.nn.Conv2d):
            return model
            
        new_in_channels = next_conv.in_channels - len(filter_indices)
        if new_in_channels <= 0:
            print(f"Error: Next conv layer {next_conv_idx} would have {new_in_channels} input channels")
            return model
            
        new_next_conv = self._create_new_conv(
            conv=next_conv,
            in_channels=new_in_channels
        )
        
        self._prune_next_conv_layer(next_conv, new_next_conv, filter_indices)
        
        # Update model
        modules[next_conv_idx] = new_next_conv
        model.net_1 = torch.nn.Sequential(*modules)
        
        return model

    def _update_fc_layer_for_pruning(self, model, layer_idx, filter_indices):
        """Update the fully connected layer when the last conv layer is pruned."""
        # Find the first FC layer
        fc_modules = list(model.net_2)
        fc_layer = None
        fc_idx = None
        
        for i, module in enumerate(fc_modules):
            if isinstance(module, torch.nn.Linear):
                fc_layer = module
                fc_idx = i
                break
                
        if fc_layer is None:
            print("Warning: No FC layer found to update")
            return model
            
        # Get the conv layer BEFORE pruning to calculate the original relationship
        conv_layer = list(model.net_1)[layer_idx]
        if not isinstance(conv_layer, torch.nn.Conv2d):
            return model
    
        # Calculate the feature map size after the conv layer
        # For VGG11: final conv output is typically 7x7 per filter
        # This needs to match the actual architecture
        feature_map_size = 7 * 7  # Standard for VGG11 with 224x224 input
    
        # Calculate original filters before this pruning operation
        original_filters = conv_layer.out_channels + len(filter_indices)
        expected_fc_input = original_filters * feature_map_size
        
        # Verify our calculation matches the actual FC layer
        if fc_layer.in_features != expected_fc_input:
            # Try to auto-detect the feature map size
            feature_map_size = fc_layer.in_features // original_filters
            print(f"Auto-detected feature map size: {feature_map_size}")
    
        params_per_filter = feature_map_size
        new_in_features = fc_layer.in_features - (len(filter_indices) * params_per_filter)
        
        if new_in_features <= 0:
            print(f"Error: FC layer would have {new_in_features} input features")
            return model
            
        print(f"Updating FC layer: {fc_layer.in_features} -> {new_in_features} features")
        new_fc = torch.nn.Linear(new_in_features, fc_layer.out_features).to(self.device)
        
        # Copy weights, skipping pruned filter contributions
        with torch.no_grad():
            target_idx = 0
            
            for source_filter in range(original_filters):
                if source_filter not in filter_indices:
                    start_old = source_filter * params_per_filter
                    end_old = (source_filter + 1) * params_per_filter
                    
                    start_new = target_idx * params_per_filter
                    end_new = (target_idx + 1) * params_per_filter
                    
                    # Bounds checking
                    if end_old <= fc_layer.weight.size(1) and end_new <= new_fc.weight.size(1):
                        new_fc.weight.data[:, start_new:end_new] = fc_layer.weight.data[:, start_old:end_old]
                    else:
                        print("Warning: Bounds error in FC layer update")
                        return model
                    
                    target_idx += 1
            
            # Copy bias
            new_fc.bias.data = fc_layer.bias.data.clone()
        
        # Initialize weights properly to prevent NaN
        with torch.no_grad():
            # Clamp weights to reasonable range
            new_fc.weight.data = torch.clamp(new_fc.weight.data, -1.0, 1.0)
            new_fc.bias.data = torch.clamp(new_fc.bias.data, -1.0, 1.0)
            
            # Check for NaN/inf values
            if torch.isnan(new_fc.weight.data).any() or torch.isinf(new_fc.weight.data).any():
                print("Warning: NaN/inf detected in FC weights, reinitializing")
                torch.nn.init.xavier_uniform_(new_fc.weight.data)
                torch.nn.init.zeros_(new_fc.bias.data)
        
        # Update model
        fc_modules[fc_idx] = new_fc # type: ignore
        model.net_2 = torch.nn.Sequential(*fc_modules)
        
        return model

