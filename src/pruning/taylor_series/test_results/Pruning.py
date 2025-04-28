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
        modules = list(model.net_1)
        while (layer_index + offset) < len(modules):
            candidate = modules[layer_index + offset]
            if isinstance(candidate, torch.nn.Conv2d):
                next_conv = candidate
                break
            offset += 1
        return next_conv
    
    def _get_next_conv_offset(self, model, layer_index):
        offset = 1
        modules = list(model.net_1)
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
        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = new_next_conv.weight.data.cpu().numpy()
        
        new_weights[:, :filter_index, :, :] = old_weights[:, :filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index+1:, :, :]
        
        new_next_conv.weight.data = torch.from_numpy(new_weights).to(self.device)
        new_next_conv.bias.data = next_conv.bias.data.to(self.device)
    
    def _prune_last_conv_layer(self, model, conv, new_conv, layer_index, filter_index):
        """Prune the last convolutional layer and update the first fully connected layer"""
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