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
    
    def _replace_layers(self, model, i, indices, layers):
        return layers[indices.index(i)] if i in indices else model[i]
    
    def _get_next_conv(self, model, layer_index):
        next_conv = None
        offset = 1
        modules = list(model.features)
        while (layer_index + offset) < len(modules):
            candidate = modules[layer_index + offset]
            if isinstance(candidate, torch.nn.Conv2d):
                next_conv = candidate
                break
            offset += 1
        return next_conv
    
    def _get_next_conv_offset(self, model, layer_index):
        offset = 1
        modules = list(model.features)
        while (layer_index + offset) < len(modules):
            candidate = modules[layer_index + offset]
            if isinstance(candidate, torch.nn.Conv2d):
                break
            offset += 1
        return offset
    
    def _create_new_conv(self, conv, in_channels=None, out_channels=None):
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
        """Prune a convolutional layer directly with PyTorch operations"""
        # Copy weights for filters before the pruned filter
        if filter_index > 0:
            new_conv.weight.data[:filter_index] = conv.weight.data[:filter_index]
        
        # Copy weights for filters after the pruned filter
        if filter_index < conv.weight.data.size(0) - 1:
            new_conv.weight.data[filter_index:] = conv.weight.data[filter_index+1:]
        
        # Handle bias similarly
        if conv.bias is not None:
            if filter_index > 0:
                new_conv.bias.data[:filter_index] = conv.bias.data[:filter_index]
            if filter_index < conv.bias.data.size(0) - 1:
                new_conv.bias.data[filter_index:] = conv.bias.data[filter_index+1:]
    
    def _prune_next_conv_layer(self, next_conv, new_next_conv, filter_index):
        """Prune input channels of the next convolutional layer directly with PyTorch operations"""
        # Copy weights for input channels before the pruned channel
        if filter_index > 0:
            new_next_conv.weight.data[:, :filter_index] = next_conv.weight.data[:, :filter_index]
        
        # Copy weights for input channels after the pruned channel
        if filter_index < next_conv.weight.data.size(1) - 1:
            new_next_conv.weight.data[:, filter_index:] = next_conv.weight.data[:, filter_index+1:]
        
        # Copy bias directly (not affected by input channels)
        if next_conv.bias is not None:
            new_next_conv.bias.data = next_conv.bias.data.clone()
    
    def _prune_last_conv_layer(self, model, conv, new_conv, layer_index, filter_index):
        """Prune the last convolutional layer and update the first fully connected layer"""
        # Replace conv layer in model
        modules = list(model.features)
        modules[layer_index] = new_conv
        model.features = torch.nn.Sequential(*modules)
        
        # Find the first fully connected layer
        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
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
        
        # Directly copy weights without NumPy conversion
        if filter_index > 0:
            start_idx = 0
            end_idx = filter_index * params_per_input_channel
            new_linear_layer.weight.data[:, start_idx:end_idx] = old_linear_layer.weight.data[:, start_idx:end_idx]
        
        if filter_index < conv.out_channels - 1:
            start_idx = filter_index * params_per_input_channel
            old_start_idx = (filter_index + 1) * params_per_input_channel
            new_linear_layer.weight.data[:, start_idx:] = old_linear_layer.weight.data[:, old_start_idx:]
        
        # Copy bias directly
        new_linear_layer.bias.data = old_linear_layer.bias.data.clone()
        
        # Replace the classifier layer
        classifier_modules = list(model.classifier)
        classifier_modules[layer_index] = new_linear_layer
        model.classifier = torch.nn.Sequential(*classifier_modules)
        
        self._clear_memory()
        return model
    
    def prune_vgg_conv_layer(self, model, layer_index, filter_index):
        """Prune a conv layer from a VGG-like network"""
        try:
            # Get the conv layer to prune
            conv = list(model.features)[layer_index]
            next_conv = self._get_next_conv(model, layer_index)
            
            # Create new conv layer with one fewer filter
            new_conv = self._create_new_conv(conv)
            
            # Prune the selected filter
            self._prune_conv_layer(conv, new_conv, filter_index)
            
            if next_conv is not None:
                # Create and update the next conv layer
                next_new_conv = self._create_new_conv(next_conv, 
                                                     in_channels=next_conv.in_channels - 1, 
                                                     out_channels=next_conv.out_channels)
                self._prune_next_conv_layer(next_conv, next_new_conv, filter_index)
                
                # Replace the layers in the model
                modules = list(model.features)
                modules[layer_index] = new_conv
                offset = self._get_next_conv_offset(model, layer_index)
                modules[layer_index + offset] = next_new_conv
                model.features = torch.nn.Sequential(*modules)
            else:
                # This is the last conv layer, update the classifier as well
                model = self._prune_last_conv_layer(model, conv, new_conv, layer_index, filter_index)
            
            self._clear_memory()
            return model
            
        except Exception as e:
            print(f"Error during pruning: {str(e)}")
            import traceback
            traceback.print_exc()
            self._clear_memory()
            return model