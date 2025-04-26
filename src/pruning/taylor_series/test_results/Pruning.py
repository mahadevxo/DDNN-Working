import torch
import numpy as np
import gc

class Pruning:
    def __init__(self, model):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
    
    def _clear_memory(self):
        """Helper method to clear memory efficiently"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def _replace_layers(self, model, i, indexes, layers):
        """Replace layers at specified indexes with new layers"""
        return layers[indexes.index(i)] if i in indexes else model[i]
    
    def _get_next_conv(self, model, layer_index):
        """Find the next convolutional layer after the specified layer index"""
        next_conv = None
        offset = 1
        
        while layer_index + offset < len(model.net_1._modules.items()):
            name, candidate = list(model.net_1._modules.items())[layer_index + offset]
            if isinstance(candidate, torch.nn.Conv2d):
                next_conv = candidate
                break
            offset += 1
        
        return next_conv, offset if next_conv is not None else None
    
    def _create_new_conv(self, conv, in_channels=None, out_channels=None):
        """Create a new convolutional layer with specified parameters"""
        in_channels = conv.in_channels if in_channels is None else in_channels
        out_channels = conv.out_channels - 1 if out_channels is None else out_channels

        return torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
        )
        
    def _prune_conv_layer(self, conv, new_conv, filter_index):
        """Remove a filter from a convolutional layer"""
        old_weights = conv.weight.data.cpu().numpy()
        new_weights = new_conv.weight.data.cpu().numpy()
        
        new_weights[:filter_index, :, :, :] = old_weights[:filter_index, :, :, :]
        new_weights[filter_index:, :, :, :] = old_weights[filter_index+1:, :, :, :]
        
        new_conv.weight.data = torch.from_numpy(new_weights).to(self.device)
        
        if conv.bias is not None:
            bias_numpy = conv.bias.data.cpu().numpy()
            bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
            bias[:filter_index] = bias_numpy[:filter_index]
            bias[filter_index:] = bias_numpy[filter_index+1:]
            new_conv.bias.data = torch.from_numpy(bias).to(self.device)
    
    def _prune_next_conv_layer(self, next_conv, new_next_conv, filter_index):
        """Update the next convolutional layer to account for the removed filter"""
        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = new_next_conv.weight.data.cpu().numpy()
        
        new_weights[:, :filter_index, :, :] = old_weights[:, :filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index+1:, :, :]
        
        new_next_conv.weight.data = torch.from_numpy(new_weights).to(self.device)
        
        if next_conv.bias is not None:
            new_next_conv.bias.data = next_conv.bias.data.clone().to(self.device)
    
    def _prune_last_conv_layer(self, model, conv, new_conv, layer_index, filter_index):
        """Prune the last convolutional layer and update the first fully connected layer"""
        # Calculate the actual feature map size
        with torch.no_grad():
            # Create dummy input (adjust size if needed for your model)
            dummy_input = torch.zeros(1, 3, 224, 224).to(self.device)
            
            # Forward through net_1 to get the feature maps
            feature_maps = model.net_1(dummy_input)
            
            # Calculate features per filter
            total_features = feature_maps.numel() // feature_maps.shape[0]  # Remove batch dimension
            filters = feature_maps.shape[1]  # Number of channels
            features_per_filter = total_features // filters
            
            print(f"Feature maps shape: {feature_maps.shape}, Features per filter: {features_per_filter}")
        
        # Replace conv layer in model
        modules = list(model.net_1)
        modules[layer_index] = new_conv
        model.net_1 = torch.nn.Sequential(*modules)
        
        # Find the first fully connected layer
        layer_index = 0
        old_linear_layer = None
        for name, module in model.net_2._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index += 1
        
        if old_linear_layer is None:
            raise ValueError("No linear layer found in net_2")
        
        # Create new linear layer with updated input size
        new_linear_layer = torch.nn.Linear(
            old_linear_layer.in_features - features_per_filter,
            old_linear_layer.out_features
        )
        
        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()
        
        # Copy weights, skipping the pruned filter's connections
        new_weights[:, :filter_index * features_per_filter] = \
            old_weights[:, :filter_index * features_per_filter]
        new_weights[:, filter_index * features_per_filter:] = \
            old_weights[:, (filter_index + 1) * features_per_filter:]
            
        new_linear_layer.weight.data = torch.from_numpy(new_weights).to(self.device)
        new_linear_layer.bias.data = old_linear_layer.bias.data.to(self.device)
        
        # Replace the linear layer in net_2
        modules = list(model.net_2)
        modules[layer_index] = new_linear_layer
        model.net_2 = torch.nn.Sequential(*modules)
        
        # Clean up
        del old_linear_layer
        
        self._clear_memory()
        return model
    
    def prune_vgg_conv_layer(self, model, layer_index, filter_index):
        """Prune a specific filter from a convolutional layer in a VGG-style network"""
        # Get the layer to prune
        try:
            _, conv = list(model.net_1._modules.items())[layer_index]
            if not isinstance(conv, torch.nn.Conv2d):
                raise ValueError(f"Layer at index {layer_index} is not a Conv2d layer")
        except IndexError as e:
            raise IndexError(f"Layer index {layer_index} out of bounds") from e

        # Create new conv layer with one less filter
        new_conv = self._create_new_conv(conv)

        # Remove the selected filter
        self._prune_conv_layer(conv, new_conv, filter_index)

        # Find the next conv layer (if any)
        next_conv, offset = self._get_next_conv(model, layer_index) if self._get_next_conv(model, layer_index)[0] else (None, None)

        if next_conv is not None:
            # Update the next conv layer to handle the removed filter
            next_new_conv = self._create_new_conv(
                next_conv, 
                in_channels=next_conv.in_channels - 1,
                out_channels=next_conv.out_channels
            )
            self._prune_next_conv_layer(next_conv, next_new_conv, filter_index)

            # Replace both layers in the model using the replacement approach
            net_1 = torch.nn.Sequential(
                *(self._replace_layers(model.net_1, i, [layer_index, layer_index + offset], 
                                      [new_conv, next_new_conv])
                  for i, _ in enumerate(model.net_1))
            )
            del model.net_1
            model.net_1 = net_1

            # Clean up references
            del next_conv
            del conv
        else:
            # This is the last conv layer - need to update the first FC layer
            model = self._prune_last_conv_layer(model, conv, new_conv, layer_index, filter_index)

        self._clear_memory()
        return model