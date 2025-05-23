import torch
import numpy as np
import gc

class Pruning:
    def __init__(self, model):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
    
    def _clear_memory(self):
        """
        Clears GPU/MPS memory by forcing garbage collection and emptying caches.
        
        This function helps prevent memory leaks during pruning by explicitly
        freeing unused memory after operations that might create large temporary tensors.
        
        Args:
            None
            
        Returns:
            None
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def _replace_layers(self, model, i, indices, layers):
        """
        Replaces specific layers in a sequential model.
        
        Substitutes the layer at index i with a corresponding layer from the layers list
        if i is in the indices list; otherwise, keeps the original layer.
        
        Args:
            model: The sequential model containing layers to replace
            i: Current layer index being evaluated
            indices: List of indices where layers should be replaced
            layers: List of new layers to use as replacements
            
        Returns:
            Either the replacement layer or the original layer
        """
        return layers[indices.index(i)] if i in indices else model[i]
    
    def _get_next_conv(self, model, layer_index):
        """
        Finds the next convolutional layer after the specified layer index.
        
        Searches through the model architecture starting from the given layer index
        and returns the first convolutional layer encountered.
        
        Args:
            model: The model to search through
            layer_index: Starting index for the search
            
        Returns:
            The next convolutional layer, or None if no conv layer is found
        """
        next_conv = None
        offset = 1
        modules = list(model.net_1)
        while (layer_index + offset) < len(modules):
            candidate = modules[layer_index + offset]
            if isinstance(candidate, torch.nn.Conv2d):
                next_conv = candidate
                break
            offset += 1
        return next_conv
    
    def _get_next_conv_offset(self, model, layer_index):
        """
        Calculates the offset to the next convolutional layer.
        
        Determines how many layers ahead the next convolutional layer is
        from the specified starting layer index.
        
        Args:
            model: The model to search through
            layer_index: Starting index for the search
            
        Returns:
            Integer representing the number of layers until the next conv layer
        """
        offset = 1
        modules = list(model.net_1)
        while (layer_index + offset) < len(modules):
            candidate = modules[layer_index + offset]
            if isinstance(candidate, torch.nn.Conv2d):
                break
            offset += 1
        return offset
    
    def _create_new_conv(self, conv, in_channels=None, out_channels=None):
        """
        Creates a new convolutional layer with specified dimensions.
        
        Constructs a new conv layer with the same parameters as the original
        but with potentially different input/output channel dimensions.
        
        Args:
            conv: Original convolutional layer to copy parameters from
            in_channels: Number of input channels for the new layer (if None, uses original)
            out_channels: Number of output channels for the new layer (if None, uses original - 1)
            
        Returns:
            New conv2d layer with the specified dimensions
        """
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
        """
        Transfers weights from original conv layer to new layer, removing the specified filter.
        
        Copies weights and biases from the original conv layer to the new one,
        skipping the filter at filter_index to effectively remove it.
        
        Args:
            conv: Original convolutional layer
            new_conv: New convolutional layer with one fewer filter
            filter_index: Index of the filter to remove
            
        Returns:
            None, but modifies new_conv in-place
        """
        old_weights = conv.weight.data.cpu().numpy()
        new_weights = new_conv.weight.data.cpu().numpy()
        
        new_weights[:filter_index, :, :, :] = old_weights[:filter_index, :, :, :]
        new_weights[filter_index:, :, :, :] = old_weights[filter_index+1:, :, :, :]
        
        new_conv.weight.data = torch.from_numpy(new_weights).to(self.device)
        bias_numpy = conv.bias.data.cpu().numpy()
        
        bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
        bias[:filter_index] = bias_numpy[:filter_index]
        bias[filter_index:] = bias_numpy[filter_index+1:]
        
        new_conv.bias.data = torch.from_numpy(bias).to(self.device)
    
    def _prune_next_conv_layer(self, next_conv, new_next_conv, filter_index):
        """
        Adjusts the next convolutional layer to account for removed input channels.
        
        When a filter is removed from one layer, the next layer must be adjusted
        to expect one fewer input channel. This function modifies the weights
        of the next layer accordingly.
        
        Args:
            next_conv: Original next convolutional layer
            new_next_conv: New next convolutional layer with one fewer input channel
            filter_index: Index of the filter/channel that was removed
            
        Returns:
            None, but modifies new_next_conv in-place
        """
        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = new_next_conv.weight.data.cpu().numpy()
        
        new_weights[:, :filter_index, :, :] = old_weights[:, :filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index+1:, :, :]
        
        new_next_conv.weight.data = torch.from_numpy(new_weights).to(self.device)
        new_next_conv.bias.data = next_conv.bias.data.to(self.device)
    
    def _prune_last_conv_layer(self, model, conv, new_conv, layer_index, filter_index):
        """
        Prunes the last convolutional layer and adjusts the first fully connected layer.
        
        When pruning the last convolutional layer, the first fully connected layer
        also needs to be modified to account for the reduced feature map dimensions.
        This function handles that special case.
        
        Args:
            model: The model being pruned
            conv: Original last convolutional layer
            new_conv: New convolutional layer with one fewer filter
            layer_index: Index of the layer in the model
            filter_index: Index of the filter to remove
            
        Returns:
            Modified model with updated layers
        """
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
        
        # Create new linear layer
        new_linear_layer = torch.nn.Linear(
            old_linear_layer.in_features - params_per_input_channel,
            old_linear_layer.out_features
        )
        
        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()
        
        new_weights[:, :filter_index * params_per_input_channel] = \
            old_weights[:, :filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel:] = \
            old_weights[:, (filter_index + 1) * params_per_input_channel:]
            
        new_linear_layer.weight.data = torch.from_numpy(new_weights).to(self.device)
        new_linear_layer.bias.data = old_linear_layer.bias.data.to(self.device)
        
        model.net_2 = torch.nn.Sequential(
            *(self._replace_layers(model.net_2, i, [layer_index], \
                [new_linear_layer]) for i, _ in enumerate(model.net_2)))
        
        self._clear_memory()
        return model
    
    def prune_vgg_conv_layer(self, model, layer_index, filter_index):
        """
        Prunes a single filter from a convolutional layer in a VGG-style network.
        
        Removes the specified filter from the convolutional layer at layer_index
        and makes all necessary adjustments to subsequent layers to maintain
        network functionality.
        
        Args:
            model: The model to prune
            layer_index: Index of the convolutional layer to prune
            filter_index: Index of the filter to remove
            
        Returns:
            Pruned model with the specified filter removed
        """
        _, conv = list(model.net_1._modules.items())[layer_index]
        next_conv = self._get_next_conv(model, layer_index)
        new_conv = self._create_new_conv(conv)

        self._prune_conv_layer(conv, new_conv, filter_index)

        if next_conv is not None:
            # Fix: explicitly pass out_channels to keep the same number of filters
            next_new_conv = self._create_new_conv(next_conv, in_channels=next_conv.in_channels - 1, out_channels=next_conv.out_channels)
            self._prune_next_conv_layer(next_conv, next_new_conv, filter_index)
            # Replace specific layers in the Sequential rather than building tuples
            modules = list(model.net_1)
            modules[layer_index] = new_conv
            offset = self._get_next_conv_offset(model, layer_index)
            modules[layer_index + offset] = next_new_conv
            model.net_1 = torch.nn.Sequential(*modules)
        else:
            # Use _prune_last_conv_layer to update classifier layers for the last conv layer
            model = self._prune_last_conv_layer(model, conv, new_conv, layer_index, filter_index)
        self._clear_memory()
        return model
